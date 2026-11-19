"""Train perception heads using one shared frozen-source/cache/resume contract."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.scheme1 import atomic_torch_save, check_source_files, resolve_path
from src.utils.clip_mapping import (
    CachedMappingDataset, atomic_json, cache_source_means,
    frozen_brain, load_brain_source, load_mapping_config,
    mapping_lock, training_clip_mean, verified_completed_mean_cache,
    source_nsd_dataset,
)
from src.utils.scheme1_runs import training_directory
from src.utils.seed import seed_everything
from src.utils.mapping_precision import mapping_scaler
from src.utils.clip_head_tasks import MappingTrainingTask
from train_brainvae_scheme1 import atomic_json_lines, capture_rng_state, choose_device, restore_rng_state


def mapping_training_identity(config: dict, max_steps: int, max_val_steps: int) -> dict:
    return MappingTrainingTask(config).identity(max_steps, max_val_steps)


def train(args, config, brain_config, paths, device, run_dir, task=None):
    task = task or MappingTrainingTask(config)
    checkpoint_dir = run_dir / "checkpoints"
    mixed_precision = config["training"].get("mixed_precision", "no")
    micro_batch_size = int(config["training"].get("micro_batch_size", 0))
    if not args.cache_only:
        scaler = mapping_scaler(device, mixed_precision)

    def configure_head_backend():
        # Only change precision policy after frozen means have been computed.
        # A shared cache must keep the original BrainVAE's FP32 encoding policy.
        if "allow_tf32" in config["training"]:
            torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["allow_tf32"])
            torch.backends.cudnn.allow_tf32 = bool(config["training"]["allow_tf32"])

    if not args.dry_run and not args.cache_only:
        if args.resume:
            if not (checkpoint_dir / "last.pt").is_file():
                raise FileNotFoundError("No resumable mapping last.pt; run initial training without --resume")
        elif (checkpoint_dir / "last.pt").exists() or (checkpoint_dir / "best.pt").exists():
            raise FileExistsError(f"Mapping checkpoints already exist in {run_dir}; use --resume")
    source = load_brain_source(config, brain_config, ROOT, args.subject, args.brainvae_checkpoint)
    manifest, split, normalizer = source["manifest"], source["split"], source["normalizer"]
    check_source_files(paths.nsd_root, args.subject, manifest, ("train_fmri", "train_ids", "train_clip"))
    source_pool = source_nsd_dataset(source, paths.nsd_root, args.subject, "train",
                                   average_repeats=True, load_targets=False,
                                   expected_ids=split["source_image_ids"])
    shape = (config["model"]["latent_tokens"], config["model"]["embed_dim"])
    clip_path = paths.nsd_root / f"subj{args.subject:02d}" / f"nsd_sdxl_clip_train_sub{args.subject}.npy"
    cache_directory = run_dir / "features"
    if config.get("mean_cache_run"):
        cache_directory = training_directory({**config, "run_name": config["mean_cache_run"]},
                                             ROOT, args.subject) / "features"
    if config.get('mean_cache_directory'):
        cache_directory = resolve_path(config['mean_cache_directory'], ROOT)
    if args.dry_run:
        cached = verified_completed_mean_cache(cache_directory,source['identity'],len(source_pool),shape)
        brain = None if cached else frozen_brain(source, device)
        batches = []
        for indices in (split["train_indices"], split["val_indices"]):
            dataset = (CachedMappingDataset(cached,clip_path,indices,split['source_image_ids']) if cached else
                       source_nsd_dataset(source, paths.nsd_root, args.subject, "train", indices,
                                         average_repeats=True, expected_ids=split["source_image_ids"]))
            batch = next(iter(DataLoader(dataset, batch_size=config["training"]["batch_size"])))
            if cached:
                mean, target = batch['mean'].to(device), batch['clip'].to(device)
            else:
                width = int(config['training']['encode_batch_size'])
                with torch.no_grad():
                    mean = torch.cat([brain.encode(part.to(device)).mode() for part in batch['fmri'].split(width)])
                target = batch['clip_vision'].to(device)
            batches.append({"mean": mean.detach(), "clip": target,
                            "image_id": batch["image_id"]})
        if brain is not None and any(p.grad is not None for p in brain.parameters()):
            raise RuntimeError("Frozen BrainVAE unexpectedly received a gradient")
        del brain, source["weights"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
        configure_head_backend()
        raw_clip = np.load(clip_path, mmap_mode="r", allow_pickle=False)
        task.prepare(raw_clip, split, source["identity"], device)
        seed_everything(int(config['training']['seed']))
        mapper = task.build().to(device)
        if "forward_seed" in config["training"]:
            seed_everything(int(config["training"]["forward_seed"]))
        mapper.train()
        components = task.backward(mapper, batches[0]["mean"], batches[0]["clip"], batches[0],
                                   scaler, mixed_precision, micro_batch_size)
        loss = components["total"]
        scaler.unscale_(AdamW(mapper.parameters()))
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mapper.parameters()):
            raise FloatingPointError("Non-finite mapping dry-run loss/gradient")
        control = training_clip_mean(raw_clip, split["train_indices"])
        print(json.dumps({"loss": float(loss), "components":{key:float(value) for key,value in components.items()},
                          "mixed_precision": mixed_precision,
                          "validation": task.evaluate(mapper, [batches[1]], device, control, 0,
                                                      mixed_precision, micro_batch_size)}, indent=2))
        print(f"[已完成] stage={task.stage}_smoke", flush=True)
        return
    with mapping_lock(cache_directory):
        mean_path = cache_source_means(lambda: frozen_brain(source, device), source_pool,
                                      cache_directory, source["identity"], shape,
                                      int(config["training"]["encode_batch_size"]), device)
    # The head is the only model kept on the accelerator during optimization.
    del source["weights"], source_pool
    if args.cache_only:
        print(f"[已完成] stage=mapping_cache path={mean_path}", flush=True)
        return
    train_data = CachedMappingDataset(mean_path, clip_path, split["train_indices"], split["source_image_ids"])
    val_data = CachedMappingDataset(mean_path, clip_path, split["val_indices"], split["source_image_ids"])
    if tuple(train_data.means.shape[1:]) != shape:
        raise ValueError("Mapping cache dimensions differ from the head")
    mean_clip = training_clip_mean(train_data.clip, split["train_indices"])
    task.prepare(train_data.clip, split, source["identity"], device)
    seed = int(config["training"]["seed"])
    seed_everything(seed)  # Cache construction must not affect head initialization.
    configure_head_backend()
    mapper = task.build().to(device)
    if "forward_seed" in config["training"]:
        seed_everything(int(config["training"]["forward_seed"]))
    optimizer = AdamW(mapper.parameters(), lr=float(config["training"]["learning_rate"]),
                      weight_decay=float(config["training"]["weight_decay"]),
                      **({"fused": True} if config["training"].get("fused_adamw") and device.type == "cuda" else {}))
    train_generator = torch.Generator()
    loader_kwargs = {"num_workers": int(config["training"].get("num_workers", 0)),
                     "pin_memory": device.type == "cuda"}
    if loader_kwargs['num_workers']:
        loader_kwargs['persistent_workers'] = bool(config['training'].get('persistent_workers',False))
        loader_kwargs['prefetch_factor'] = int(config['training'].get('prefetch_factor',2))
    persistent_workers = bool(loader_kwargs.get('persistent_workers', False))
    if persistent_workers:
        # A reused iterator draws its worker seed only in the first epoch. Keep
        # that generator separate so a fresh resumed loader has the same order.
        train_loader = DataLoader(train_data, batch_size=int(config["training"]["batch_size"]),
                                  sampler=RandomSampler(train_data, generator=train_generator),
                                  generator=torch.Generator().manual_seed(seed),
                                  drop_last=task.drop_last, **loader_kwargs)
    else:
        train_loader = DataLoader(train_data, batch_size=int(config["training"]["batch_size"]), shuffle=True,
                                  generator=train_generator, drop_last=task.drop_last, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=int(config["validation"]["batch_size"]), shuffle=False,
                            generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    if not len(train_loader):
        raise ValueError("No complete training batch; reduce batch_size")
    identity = task.identity(args.max_steps, args.max_val_steps)
    protocol = {"protocol": task.protocol, "subject": args.subject, "run_name": run_dir.name,
                "source_identity": source["identity"], "config": config, "training_identity": identity,
                "data_fingerprint": manifest["data_fingerprint"], "prepared_data": str(source["prepared"]),
                "source_train_images": len(train_data), "source_val_images": len(val_data),
                "validation_source": "NSD_train_internal_image_holdout",
                "selection_metric": config["validation"]["selection_metric"], "selection_mode": "min",
                "loss_definition": task.loss_definition,
                "aggregation": "mean_fmri", "brainvae_frozen": True,
                "mean_cache": str(mean_path),
                "imagery_supervision": False, "test_used_for_selection": False,
                "debug_training": bool(args.max_steps or args.max_val_steps or source["identity"]["source_debug_training"])}
    protocol.update(task.protocol_fields())
    epochs, save_every = int(config["training"]["epochs"]), int(config["training"]["checkpoint_every_steps"])
    state = {"version": 1, "epoch_index": 0, "next_batch": 0, "global_step": 0,
             "epoch_loss_total": 0.0, "epoch_count": 0, "best_score": math.inf, "best_epoch": 0,
             "stale_epochs": 0, "finished": False, "history": [], "device": str(device)}
    rng_to_restore = None
    if args.resume:
        saved = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False)
        previous = saved["protocol_metadata"]
        if (previous["protocol"] != task.protocol or previous["subject"] != args.subject
                or previous["run_name"] != run_dir.name or previous["source_identity"] != source["identity"]
                or previous["training_identity"] != identity):
            raise ValueError("Mapping resume config / source BrainVAE / subject / data mismatch")
        state = saved["training_state"]
        if state.get("version") != 1 or not 0 <= state["epoch_index"] <= epochs:
            raise ValueError("Invalid mapping resume progress")
        if state["best_epoch"] and not (checkpoint_dir / "best.pt").is_file():
            raise FileNotFoundError("Selected mapping best.pt is missing")
        mapper.load_state_dict(saved["model"], strict=True)
        task.restore(saved)
        optimizer.load_state_dict(saved["optimizer"])
        if scaler.is_enabled():
            if not saved.get("grad_scaler"):
                raise ValueError("FP16 mapping checkpoint lacks resumable GradScaler state")
            scaler.load_state_dict(saved["grad_scaler"])
        rng_to_restore = state["rng_state"]
        protocol = previous
        if state["device"] != str(device):
            print("[警告] 恢复设备改变，不保证逐位一致", flush=True)
        del saved
        print(f"[断点恢复] epoch={state['epoch_index'] + 1} 已保存批次={state['next_batch']} global_step={state['global_step']}", flush=True)
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(protocol, run_dir / "protocol.json")
    atomic_json_lines(state["history"], run_dir / "history.jsonl")
    print(json.dumps({"head": config["model"], "trainable_parameters": sum(p.numel() for p in mapper.parameters() if p.requires_grad),
                      "source": source["identity"], "train_images": len(train_data), "val_images": len(val_data),
                      "mixed_precision": mixed_precision, "logical_batch_size": config["training"]["batch_size"],
                      "micro_batch_size": micro_batch_size or config["training"]["batch_size"],
                      "mean_cache": str(mean_path),
                      "loss": protocol["loss_definition"]}, indent=2), flush=True)
    if scaler.is_enabled():
        state.setdefault("amp_skipped_updates", 0)
    state.setdefault("epoch_components", {})

    def payload():
        return {"model": mapper.state_dict(), "config": config["model"], "protocol_metadata": protocol,
                "preprocessing": normalizer.checkpoint_state(),
                "split": {key: torch.from_numpy(value.copy()) for key, value in split.items()},
                "training_mean_clip": mean_clip, "best_epoch": state["best_epoch"], "best_score": state["best_score"],
                "metrics": state["history"][-1] if state["history"] else {}, **task.checkpoint_fields()}

    def save_last():
        state["rng_state"] = capture_rng_state()
        atomic_torch_save({**payload(), "optimizer": optimizer.state_dict(), "training_state": state,
                           "grad_scaler": scaler.state_dict(),
                           "checkpoint_kind": "last_resumable"}, checkpoint_dir / "last.pt")

    if not args.resume:
        save_last()
    total_batches = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
    if not 0 <= state["next_batch"] <= total_batches:
        raise ValueError("Mapping saved batch offset exceeds the epoch length")
    patience, delta = int(config["validation"]["patience"]), float(config["validation"]["min_delta"])
    for epoch in range(state["epoch_index"], epochs):
        if state["finished"]:
            break
        mapper.train()
        train_generator.manual_seed(seed + epoch)
        if persistent_workers:
            # Match the legacy fresh-iterator seed draw before every permutation,
            # including when restarting partway through a later epoch.
            torch.empty((), dtype=torch.int64).random_(generator=train_generator)
        iterator = iter(train_loader)
        skip = state["next_batch"]
        for _ in range(skip):
            next(iterator)
        if rng_to_restore is not None:
            restore_rng_state(rng_to_restore)
            rng_to_restore = None
        for step in range(skip + 1, total_batches + 1):
            batch = next(iterator)
            mean, target = batch["mean"].to(device, non_blocking=True), batch["clip"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            components = task.backward(mapper, mean, target, batch, scaler, mixed_precision, micro_batch_size)
            loss = components["total"]
            previous_scale = scaler.get_scale()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0,
                                           error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < previous_scale:
                state["amp_skipped_updates"] += 1
                print(f"[FP16溢出跳过] epoch={epoch + 1} batch={step} scale={scaler.get_scale():.6g}", flush=True)
            state.update({"epoch_index": epoch, "next_batch": step, "global_step": state["global_step"] + 1,
                          "epoch_loss_total": state["epoch_loss_total"] + float(loss.detach()) * len(mean),
                          "epoch_count": state["epoch_count"] + len(mean)})
            for key, value in components.items():
                state["epoch_components"][key] = state["epoch_components"].get(key, 0.0)+float(value)*len(mean)
            if state["global_step"] % save_every == 0:
                save_last()
                print(f"[断点已保存] epoch={epoch + 1}/{epochs} batch={step}/{total_batches} "
                      f"global_step={state['global_step']} loss={float(loss.detach()):.6g}", flush=True)
        if scaler.is_enabled() and state["amp_skipped_updates"] == state["global_step"]:
            raise FloatingPointError("All FP16 optimizer updates were skipped; no trained head to select")
        validation = task.evaluate(mapper, val_loader, device, mean_clip, args.max_val_steps,
                                   mixed_precision, micro_batch_size)
        score = task.score(validation)
        improved = score < state["best_score"] - delta
        if improved:
            state.update({"best_score": score, "best_epoch": epoch + 1, "stale_epochs": 0})
        else:
            state["stale_epochs"] += 1
        averages = {key:value/state["epoch_count"] for key,value in state["epoch_components"].items()}
        if isinstance(task, MappingTrainingTask) and not task.joint:
            averages["gen_mse"] = state["epoch_loss_total"]/state["epoch_count"]
        message = {"epoch": epoch + 1, **task.epoch_fields(averages, validation),
                   "val": validation, "best_epoch": state["best_epoch"], "best_score": state["best_score"],
                   "stale_epochs": state["stale_epochs"]}
        if scaler.is_enabled():
            message.update({"amp_scale": scaler.get_scale(),
                            "amp_skipped_updates": state["amp_skipped_updates"]})
        state["history"].append(message)
        state.update({"epoch_index": epoch + 1, "next_batch": 0, "epoch_loss_total": 0.0, "epoch_count": 0,
                      "epoch_components": {},
                      "finished": epoch + 1 >= epochs or (patience > 0 and state["stale_epochs"] >= patience)})
        if improved:
            atomic_torch_save({**payload(), "checkpoint_kind": "best_denoising_monitor" if task.stage == 'prior' else 'best_inference'}, checkpoint_dir / "best.pt")
        if epoch+1 in config["validation"].get("candidate_epochs", []):
            atomic_torch_save({**payload(), "checkpoint_kind": "predeclared_generation_candidate"},
                              checkpoint_dir / f"epoch{epoch+1:03d}.pt")
        save_last()
        atomic_json_lines(state["history"], run_dir / "history.jsonl")
        print(json.dumps(message), flush=True)
    if not state["finished"]:
        raise RuntimeError("Mapping training did not reach a terminal state")
    atomic_json({"best_epoch": state["best_epoch"], "best_score": state["best_score"],
                 "last_epoch": state["epoch_index"], "global_step": state["global_step"],
                 "selection_metric": "internal_val_"+config["validation"]["selection_metric"],
                 **task.protocol_fields()}, run_dir / "training_complete.json")
    print(f"Saved mapping checkpoints: {checkpoint_dir}", flush=True)
    print(f"[已完成] stage={task.stage}_train", flush=True)


def main(config_loader=load_mapping_config, task_factory=MappingTrainingTask):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--brainvae-checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Split each logical batch; 0 disables splitting")
    parser.add_argument("--mixed-precision", choices=("no", "fp16"), default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="One real frozen-encoder / head backward batch; no writes")
    parser.add_argument("--cache-only", action="store_true", help="Prepare/resume source posterior means without training the head")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_steps, args.max_val_steps) < 0 or args.threads <= 0 or args.subject <= 0:
        parser.error("Invalid subject / epoch / batch / step / thread count")
    if args.micro_batch_size is not None and args.micro_batch_size < 0:
        parser.error("micro-batch-size must be nonnegative")
    if args.dry_run and (args.resume or args.cache_only):
        parser.error("--dry-run cannot be combined with --resume or --cache-only")
    torch.set_num_threads(args.threads)
    config, brain_config, paths = config_loader(args.config, ROOT)
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.micro_batch_size is not None:
        config["training"]["micro_batch_size"] = args.micro_batch_size
    if args.mixed_precision is not None:
        config["training"]["mixed_precision"] = args.mixed_precision
    task = task_factory(config)  # Revalidate CLI batch/precision/objective overrides.
    device = choose_device(args.device)
    seed_everything(int(config["training"]["seed"]))
    run_dir = training_directory(config, ROOT, args.subject, max_steps=args.max_steps, max_val_steps=args.max_val_steps)
    if args.dry_run:
        train(args, config, brain_config, paths, device, run_dir, task)
    else:
        with mapping_lock(run_dir):
            train(args, config, brain_config, paths, device, run_dir, task)


if __name__ == "__main__":
    main()
