"""Train one subject's BrainVAE using only the internal NSD source split."""
from __future__ import annotations

import argparse
import json
import math
import random
import signal

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.scheme1 import (PROTOCOL, Scheme1NSDDataset, atomic_torch_save, check_prepared_config, check_source_files, data_directory,
                              file_sha256, load_prepared, load_scheme1_config, resolve_path, validate_tag)
from src.evaluation.voxel_metrics import cosine, mse, pearson
from src.models.brainvae_factory import build_brainvae
from src.utils.seed import seed_everything
from src.utils.mapping_precision import mapping_autocast, mapping_scaler
from src.utils.scheme1_runs import training_directory, training_identity
from src.utils.synbrain_training import configure_synbrain_execution, preload_shared_arrays


def capture_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG restoration is not possible")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_json_lines(rows: list, path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def nonfinite_loss_report(losses, batch, epoch: int, step: int, global_step: int, precision: str) -> dict:
    """Identify the failing loss and source records without retaining activations."""
    values = {key: float(value.detach()) for key, value in losses.items()}
    inputs = {}
    for key in ('fmri', 'clip_vision'):
        value = batch[key].detach()
        finite = torch.isfinite(value)
        valid = value[finite]
        inputs[key] = {'all_finite': bool(finite.all()), 'nonfinite_count': int((~finite).sum()),
                       'min': float(valid.min()) if valid.numel() else None,
                       'max': float(valid.max()) if valid.numel() else None}
    return {'epoch': epoch, 'batch': step, 'global_step_before_batch': global_step,
            'mixed_precision': precision,
            'invalid_losses': [key for key, value in values.items() if not math.isfinite(value)],
            'losses': {key: value if math.isfinite(value) else str(value) for key, value in values.items()},
            'image_ids': batch['image_id'].tolist(), 'dataset_indices': batch['index'].tolist(),
            'inputs': inputs}


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def kl_weight_for_epoch(epoch: int, config: dict) -> float:
    target = float(config["model"]["kl_weight"])
    warmup = int(config["training"].get("kl_warmup_epochs", 0))
    ramp = int(config["training"].get("kl_ramp_epochs", 0))
    if epoch < warmup:
        return 0.0
    return target if ramp <= 0 else target * min(1.0, (epoch - warmup) / ramp)


@torch.no_grad()
def evaluate(model, loader, device, max_steps: int = 0, mixed_precision: str = 'no') -> dict:
    """Sample-weighted streaming metrics; never retain the full token pool."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for step, batch in enumerate(loader, start=1):
        fmri = batch["fmri"].float().to(device)
        clip = batch["clip_vision"].float().to(device)
        with mapping_autocast(device, mixed_precision):
            latent, recon, losses = model(fmri, clip, sample_posterior=False)
        values = {key: float(value) for key, value in losses.items()}
        values.update({"latent_cosine": float(F.cosine_similarity(latent.flatten(1), clip.flatten(1), dim=-1).mean()),
                       "fmri_mse": mse(recon.squeeze(1), fmri), "fmri_pearson": pearson(recon.squeeze(1), fmri),
                       "fmri_cosine": cosine(recon.squeeze(1), fmri)})
        n = len(fmri)
        for key, value in values.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite validation metric: {key}")
            totals[key] = totals.get(key, 0.0) + value * n
        count += n
        if max_steps and step >= max_steps:
            break
    if not count:
        raise ValueError("Empty validation loader")
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brainvae_scheme1.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--run-tag", default="", help="Optional manual subdirectory; default: brainvae")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0, help="Debug only: limit training batches per epoch")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Debug only: limit validation batches")
    parser.add_argument("--dry-run", action="store_true", help="One real training forward/backward and validation batch; no checkpoint writes")
    parser.add_argument("--resume", action="store_true", help="Restore this run's last.pt, including optimizer, RNG and batch position")
    args = parser.parse_args()
    if args.dry_run and args.resume:
        parser.error("--resume cannot be combined with --dry-run")
    if min(args.epochs, args.batch_size, args.max_steps, args.max_val_steps) < 0 or args.threads <= 0:
        raise ValueError("Invalid epoch / batch / step / thread count")
    torch.set_num_threads(args.threads)
    config, paths = load_scheme1_config(args.config, ROOT)
    args.seed = args.seed if args.seed is not None else int(config["training"].get("seed", 42))
    epochs = args.epochs or int(config["training"]["epochs"])
    effective_batch_size = args.batch_size or int(config["training"]["batch_size"])
    if min(epochs, effective_batch_size) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    config["training"].update({"epochs": epochs, "batch_size": effective_batch_size, "seed": args.seed})
    seed_everything(args.seed)
    if args.run_tag:
        validate_tag(args.run_tag)
        run_dir = resolve_path(config["output_root"], ROOT) / f"subj{args.subject:02d}" / "runs" / args.run_tag
    else:
        run_dir = training_directory(config, ROOT, args.subject, max_steps=args.max_steps, max_val_steps=args.max_val_steps)
    if args.resume and not (run_dir / "checkpoints/last.pt").is_file():
        raise FileNotFoundError(f"No resumable last.pt in {run_dir}; run fresh training first")
    prepared = data_directory(config, ROOT, args.subject)
    manifest, split, normalizer = load_prepared(prepared)
    if manifest["subject"] != args.subject:
        raise ValueError("Prepared subject does not match requested subject")
    check_prepared_config(config, manifest)
    check_source_files(paths.nsd_root, args.subject, manifest, ("train_fmri", "train_ids", "train_clip"))
    train = Scheme1NSDDataset(paths.nsd_root, args.subject, "train", normalizer, split["train_indices"],
                             average_repeats=False, expected_ids=split["source_image_ids"])
    val = Scheme1NSDDataset(paths.nsd_root, args.subject, "train", normalizer, split["val_indices"],
                           average_repeats=bool(config["validation"]["average_repeats"]),
                           expected_ids=split["source_image_ids"])
    expected = (config["model"]["latent_tokens"], config["model"]["embed_dim"])
    if tuple(train.clip.shape[1:]) != expected:
        raise ValueError(f"Source CLIP dimensions do not match model {expected}")
    batch_size = effective_batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = choose_device(args.device)
    precision = config['training'].get('mixed_precision', 'no')
    scaler = mapping_scaler(device, precision)
    if scaler.is_enabled() and 'initial_loss_scale' in config['training']:
        scaler = torch.amp.GradScaler('cuda', init_scale=float(config['training']['initial_loss_scale']))
    if 'allow_tf32' in config['training']:
        torch.backends.cuda.matmul.allow_tf32 = bool(config['training']['allow_tf32'])
        torch.backends.cudnn.allow_tf32 = bool(config['training']['allow_tf32'])
    if 'cudnn_benchmark' in config['training']:
        torch.backends.cudnn.benchmark = bool(config['training']['cudnn_benchmark'])
    if config['training'].get('preload_ram', False):
        loaded_bytes = preload_shared_arrays(train, val)
        print(f'[数据预载] shared source arrays={loaded_bytes / 1024**3:.2f} GiB; individual training repeats retained', flush=True)
    loader_kwargs = {"batch_size": batch_size, "num_workers": int(config["training"].get("num_workers", 0)),
                     "pin_memory": device.type == "cuda"}
    if loader_kwargs['num_workers']:
        loader_kwargs.update(persistent_workers=bool(config['training'].get('persistent_workers', False)),
                             prefetch_factor=int(config['training'].get('prefetch_factor', 2)))
    train_generator = torch.Generator()
    train_loader = DataLoader(train, shuffle=True, generator=train_generator, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, generator=torch.Generator().manual_seed(args.seed), **loader_kwargs)
    model = build_brainvae(config["model"]).to(device)
    attention_backend = configure_synbrain_execution(model, config['training'])
    loss_definition = {"semantic_source": model.semantic_source, "clip_reg_weight": model.clip_reg_weight,
                       "clip_reg_reduction": "mean_raw_token_coordinates", "clip_weight": model.clip_weight,
                       "recon_weight": model.recon_weight, "recon_reduction": "sum_voxels_mean_batch",
                       "kl_weight_target": model.kl_weight, "kl_reduction": "sum_dimensions_mean_batch",
                       "reconstruction_training_source": "posterior_sample", "generation_source": "posterior_mean",
                       "architecture": config["model"].get("architecture", "project"),
                       "free_bits": model.free_bits, "normalization": normalizer.description}
    print(json.dumps({"protocol": PROTOCOL, "subject": args.subject, "device": str(device),
                      "train_images": len(split["train_indices"]), "train_trials": len(train),
                      "val_images": len(split["val_indices"]), "val_items": len(val),
                      "test_used_for_selection": False, "imagery_used_for_training": False,
                      "loss_definition": loss_definition}, indent=2), flush=True)
    if args.dry_run:
        batch = next(iter(train_loader))
        model.train()
        with mapping_autocast(device, precision):
            latent, recon, losses = model(batch["fmri"].to(device), batch["clip_vision"].to(device))
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Non-finite dry-run training loss")
        scaler.scale(losses["loss"]).backward()
        finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                               for parameter in model.parameters())
        if not finite_gradients:
            raise FloatingPointError('Dry-run produced non-finite gradients; reduce initial_loss_scale')
        print("[dry-run] real batch:", tuple(batch["fmri"].shape), "latent:", tuple(latent.shape), "recon:", tuple(recon.shape))
        print("[dry-run] losses:", {key: float(value.detach()) for key, value in losses.items()}, flush=True)
        print("[dry-run] validation:", evaluate(model, val_loader, device, max_steps=1, mixed_precision=precision), flush=True)
        print("[已完成] stage=smoke", flush=True)
        return
    run_tag = run_dir.name
    metric = config["validation"]["selection_metric"]
    mode = config["validation"].get("selection_mode", "max")
    if mode not in {"min", "max"}:
        raise ValueError("selection_mode must be min or max")
    protocol = {"protocol": PROTOCOL, "subject": args.subject, "seed": args.seed, "run_tag": run_tag,
                "config": config, "prepared_data": str(prepared), "data_fingerprint": manifest["data_fingerprint"],
                "debug_run": bool(args.max_steps or args.max_val_steps),
                "debug_steps": [args.max_steps, args.max_val_steps],
                "source_train_images": len(split["train_indices"]), "source_val_images": len(split["val_indices"]),
                "selection_metric": metric, "selection_mode": mode, "validation_source": "NSD_train_internal_image_holdout",
                "imagery_supervision": False, "anchors": False, "adapter": False,
                "loss_definition": loss_definition,
                "preprocessing_sha256": manifest["artifact_sha256"]["preprocessing.npz"]}
    numerical_policy = {'mixed_precision': precision}
    if config['model'].get('architecture') == 'synbrain':
        numerical_policy['synbrain_attention_core'] = ('sdpa_single_head_v1' if attention_backend == 'sdpa'
            else 'fp32_qk_softmax_value_v1' if precision in ('fp16', 'bf16') else 'original_fp32')
    protocol['numerical_policy'] = numerical_policy
    checkpoint_dir = run_dir / "checkpoints"
    optimizer_options = {}
    if config['training'].get('fused_optimizer', False):
        if device.type != 'cuda':
            raise ValueError('fused_optimizer requires CUDA')
        optimizer_options['fused'] = True
    optimizer = AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                      weight_decay=float(config["training"]["weight_decay"]), **optimizer_options)
    min_delta = float(config["validation"].get("min_delta", 0.0))
    patience = int(config["validation"].get("patience", 0))
    save_every = int(config["training"].get("checkpoint_every_steps", 100))
    if save_every < 1:
        raise ValueError("checkpoint_every_steps must be positive")
    state = {"version": 1, "epoch_index": 0, "next_batch": 0, "global_step": 0,
             "epoch_totals": {}, "epoch_count": 0, "best_score": -math.inf if mode == "max" else math.inf,
             "best_epoch": 0, "stale_epochs": 0, "finished": False, "history": [], "device": str(device)}
    rng_to_restore = None
    if args.resume:
        saved = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False)
        previous = saved.get("protocol_metadata", {})
        state = saved.get("training_state", {})
        if state.get("version") != 1 or "optimizer" not in saved or "rng_state" not in state:
            raise ValueError("Old checkpoint lacks optimizer/RNG/progress; cannot resume it. Start a fresh training run.")
        previous_debug = previous.get("debug_steps", [0, 0])
        if (previous.get("protocol") != PROTOCOL or previous.get("subject") != args.subject
                or previous.get("data_fingerprint") != manifest["data_fingerprint"]
                or previous.get("preprocessing_sha256") != protocol["preprocessing_sha256"]
                or previous.get("run_tag") != run_tag
                or training_identity(previous["config"], *previous_debug) != training_identity(config, args.max_steps, args.max_val_steps)):
            raise ValueError("Resume subject / training configuration / source data / normalization mismatch")
        for key, value in split.items():
            if key not in saved["split"] or not np.array_equal(saved["split"][key].numpy(), value):
                raise ValueError("Resume image split mismatch")
        if not 0 <= state["epoch_index"] <= epochs or state["next_batch"] < 0:
            raise ValueError("Invalid checkpoint progress")
        if state["best_epoch"] and not (checkpoint_dir / "best.pt").is_file():
            raise FileNotFoundError("The run's selected best.pt is missing; cannot safely resume")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scaler.is_enabled():
            if not saved.get('grad_scaler'):
                raise ValueError('AMP BrainVAE resume requires saved GradScaler state')
            scaler.load_state_dict(saved['grad_scaler'])
        rng_to_restore = state["rng_state"]
        protocol = dict(previous)  # Keep original training provenance and record numerical fixes.
        previous_policy = previous.get('numerical_policy')
        if previous_policy is None and config['model'].get('architecture') == 'synbrain' and precision == 'fp16':
            previous_policy = {'mixed_precision': precision, 'synbrain_attention_core': 'legacy_fp16'}
        if previous_policy is not None and previous_policy != numerical_policy:
            recovery = {'from_global_step': state['global_step'], 'epoch': state['epoch_index'] + 1,
                        'saved_batch': state['next_batch'], 'previous_policy': previous_policy,
                        'new_policy': numerical_policy,
                        'source_checkpoint_sha256': file_sha256(checkpoint_dir / 'last.pt')}
            protocol['numerical_recovery_history'] = [*protocol.get('numerical_recovery_history', []), recovery]
            print('[数值修复恢复] ' + json.dumps(recovery, ensure_ascii=False), flush=True)
        protocol['numerical_policy'] = numerical_policy
        temporary = run_dir / 'protocol.json.tmp'
        temporary.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(run_dir / 'protocol.json')
        if state["device"] != str(device):
            print("[警告] 恢复设备改变，随机流与数值结果不保证逐位一致", flush=True)
        atomic_json_lines(state["history"], run_dir / "history.jsonl")
        if state["finished"]:
            print(f"[断点恢复] 检查点已完成，核验并补写完成标记，不再训练 global_step={state['global_step']}", flush=True)
        else:
            print(f"[断点恢复] epoch={state['epoch_index'] + 1} 已保存批次={state['next_batch']} "
                  f"global_step={state['global_step']} checkpoint={checkpoint_dir / 'last.pt'}", flush=True)
        del saved
    else:
        if (config["model"].get("architecture") == "synbrain" and
                ((checkpoint_dir / "last.pt").exists() or (checkpoint_dir / "best.pt").exists())):
            raise FileExistsError(f"BrainVAE checkpoints already exist in {run_dir}; use --resume or a new run_name")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # A new training run replaces this fixed run's checkpoints. Remove the
        # previous completion marker before publishing its first checkpoint.
        completion_marker = run_dir / "training_complete.json"
        if completion_marker.exists():
            completion_marker.unlink()
        (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        atomic_json_lines([], run_dir / "history.jsonl")

    def payload() -> dict:
        return {"model": model.state_dict(), "config": config["model"],
                "epoch": state["history"][-1]["epoch"] if state["history"] else 0,
                "protocol_metadata": protocol, "data_manifest": manifest,
                "preprocessing": normalizer.checkpoint_state(),
                "split": {key: torch.from_numpy(value.copy()) for key, value in split.items()},
                "metrics": state["history"][-1] if state["history"] else {},
                "best_score": state["best_score"], "best_epoch": state["best_epoch"]}

    def save_last() -> None:
        state["rng_state"] = capture_rng_state()
        atomic_torch_save({**payload(), "optimizer": optimizer.state_dict(), "training_state": state,
                           'grad_scaler': scaler.state_dict(),
                           "checkpoint_kind": "last_resumable"}, checkpoint_dir / "last.pt")

    if not args.resume:
        save_last()  # Even a first-epoch interruption has a recoverable start.
    stop_request = {'signal': None}

    def request_stop(signum, frame):
        stop_request['signal'] = signum

    if config['training'].get('save_on_signal', False):
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    total_batches = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    if state["next_batch"] > total_batches:
        raise ValueError("Saved batch offset exceeds the current epoch length")
    first_step_to_report = state['global_step'] + 1
    for epoch in range(state["epoch_index"], epochs):
        if state["finished"]:
            break
        model.train()
        train_generator.manual_seed(args.seed + epoch)
        iterator = iter(train_loader)
        skip = state["next_batch"]
        for _ in range(skip):
            next(iterator)  # Deterministic dataset; rebuild the same epoch order.
        if rng_to_restore is not None:
            restore_rng_state(rng_to_restore)
            rng_to_restore = None
        totals, count = state["epoch_totals"], state["epoch_count"]
        active_kl = kl_weight_for_epoch(epoch, config)
        for step in range(skip + 1, total_batches + 1):
            batch = next(iterator)
            fmri, clip = batch["fmri"].to(device, non_blocking=True), batch["clip_vision"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with mapping_autocast(device, precision):
                _, _, losses = model(fmri, clip, sample_posterior=True, kl_weight_override=active_kl)
            if not all(torch.isfinite(value).item() for value in losses.values()):
                report = nonfinite_loss_report(losses, batch, epoch + 1, step, state['global_step'], precision)
                report_path = run_dir / 'numerical_failure.json'
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}, step {step}; "
                                         f"invalid={report['invalid_losses']}; losses={report['losses']}; "
                                         f"diagnostics={report_path}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            if config['training'].get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'],
                                               error_if_nonfinite=not scaler.is_enabled())
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < previous_scale:
                state['amp_skipped_updates'] = state.get('amp_skipped_updates', 0) + 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(fmri)
            count += len(fmri)
            state.update({"epoch_index": epoch, "next_batch": step, "epoch_totals": totals, "epoch_count": count,
                          "global_step": state["global_step"] + 1})
            if state['global_step'] == first_step_to_report:
                print(f"[训练已启动] epoch={epoch + 1}/{epochs} batch={step}/{total_batches} "
                      f"global_step={state['global_step']} "
                      f"loss={float(losses['loss'].detach()):.6g} precision={precision} "
                      f"amp_skipped_updates={state.get('amp_skipped_updates', 0)}", flush=True)
            if state["global_step"] % save_every == 0:
                save_last()
                details = ""
                if "clip_reg_loss" in losses:
                    details = (f" raw_token_mse={float(losses['clip_reg_loss'].detach()):.6g}"
                               f" mean_cos={float(losses['mean_clip_cosine']):.6g}"
                               f" norm_ratio={float(losses['mean_norm_ratio']):.6g}"
                               f" logvar_mean={float(losses['logvar_mean']):.6g}")
                print(f"[断点已保存] epoch={epoch + 1}/{epochs} batch={step}/{total_batches} "
                      f"global_step={state['global_step']} loss={float(losses['loss'].detach()):.6g}{details}", flush=True)
            if stop_request['signal'] is not None:
                save_last()
                print('[已保存中断状态] last.pt is resumable', flush=True)
                raise SystemExit(128 + stop_request['signal'])
        if state.get('amp_skipped_updates', 0) == state['global_step']:
            raise FloatingPointError('All BrainVAE AMP optimizer updates were skipped')
        val_metrics = evaluate(model, val_loader, device, args.max_val_steps, mixed_precision=precision)
        if metric not in val_metrics:
            raise ValueError(f"Unknown selection metric: {metric}")
        score = val_metrics[metric]
        improved = score > state["best_score"] + min_delta if mode == "max" else score < state["best_score"] - min_delta
        if improved:
            state.update({"best_score": score, "best_epoch": epoch + 1, "stale_epochs": 0})
        else:
            state["stale_epochs"] += 1
        message = {"epoch": epoch + 1, "train": {k: v / count for k, v in totals.items()},
                   "val": val_metrics, "kl_weight": active_kl, "best_score": state["best_score"],
                   "best_epoch": state["best_epoch"], "stale_epochs": state["stale_epochs"]}
        state["history"].append(message)
        state.update({"epoch_index": epoch + 1, "next_batch": 0, "epoch_totals": {}, "epoch_count": 0,
                      "finished": epoch + 1 >= epochs or (patience > 0 and state["stale_epochs"] >= patience)})
        if improved:
            atomic_torch_save({**payload(), "checkpoint_kind": "best_inference"}, checkpoint_dir / "best.pt")
        save_last()
        atomic_json_lines(state["history"], run_dir / "history.jsonl")
        print(json.dumps(message), flush=True)
        if patience > 0 and state["stale_epochs"] >= patience:
            print(f"Early stopping; best epoch={state['best_epoch']}", flush=True)
            break
    if not state["finished"]:
        raise RuntimeError("Training did not reach a terminal state")
    completed = {"best_epoch": state["best_epoch"], "best_score": state["best_score"],
                 "last_epoch": state["epoch_index"], "global_step": state["global_step"]}
    temporary = run_dir / "training_complete.json.tmp"
    temporary.write_text(json.dumps(completed), encoding="utf-8")
    temporary.replace(run_dir / "training_complete.json")
    print(f"Saved scheme 1 checkpoints: {checkpoint_dir}", flush=True)
    print("[已完成] stage=train", flush=True)


if __name__ == "__main__":
    main()
