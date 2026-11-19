#!/usr/bin/env python3
"""Train a subj01 frozen-BrainVAE mean -> SDXL CLIP-L text-token head.

The first COCO caption of each image is the predeclared target. The head is
selected only on the existing internal train/validation split. NSD test text
is used only by the separate evaluate action, never by prediction or training.

Examples:
  python -u output_1104/scripts/train_sdxl_clipl_head.py --smoke --batch-size 2
  python -u output_1104/scripts/train_sdxl_clipl_head.py
  python -u output_1104/scripts/train_sdxl_clipl_head.py --resume
  python -u output_1104/scripts/train_sdxl_clipl_head.py --action predict
  python -u output_1104/scripts/train_sdxl_clipl_head.py --action evaluate
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.scheme1 import check_source_files, file_sha256, file_signature, validate_tag
from src.models.sdxl_text_head import MeanToCLIPTextHead
from src.utils.clip_mapping import (atomic_json, frozen_brain, load_brain_source,
                                    load_head_source_config, mapping_lock, source_nsd_dataset)
from train_brainvae_scheme1 import capture_rng_state, restore_rng_state


TARGET_DIR = ROOT / 'output_1104/nsd_subj01/sdxl_clip_l11_all_both_v1'
SOURCE_CONFIG = ROOT / 'configs/clip_mapping_synbrain_local_adatoken.yaml'
FORMAL_ROOT = ROOT / 'output_1104/nsd_subj01/text_prediction'
DEBUG_ROOT = ROOT / 'output_1104/debug/text_prediction'
TARGET_REPRESENTATION = 'sdxl_clip_l_hidden_states_11_pre_final_layernorm_unprojected'
STOP_SIGNAL = 0


def on_signal(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum
    print('[signal] {}: saving the latest completed batch'.format(signum), flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_torch(payload, path):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    os.replace(str(temp), str(path))


def load_brain_context():
    mapping, brain_cfg, paths = load_head_source_config(SOURCE_CONFIG, ROOT)
    source = load_brain_source(mapping, brain_cfg, ROOT, 1)
    if source['identity']['subject'] != 1 or source['identity'].get('source_debug_training'):
        raise ValueError('The source must be the completed subj01 BrainVAE')
    check_source_files(paths.nsd_root, 1, source['manifest'], ('train_fmri', 'train_ids', 'test_fmri', 'test_ids'))
    return source, paths


def load_targets(directory, split, expected_ids, verify_hash=True):
    """Load caption-aligned labels for a specified split, validating provenance."""
    identity_path = directory / 'source_identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    path = directory / '{}_clip_l11_tokens.npy'.format(split)
    metadata = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    if (identity.get('subject') != 1 or identity.get('max_images_per_split') != 0
            or identity.get('representation') != TARGET_REPRESENTATION
            or metadata.get('representation') != TARGET_REPRESENTATION
            or metadata.get('split') != split or metadata.get('caption_policy') != 'all'):
        raise ValueError('Not the complete, per-caption subj01 CLIP-L hidden-11 target')
    from src.data.scheme1 import digest_json
    if metadata.get('identity_sha256') != digest_json(identity):
        # The extractor uses compact JSON for identity. Recompute its exact digest.
        import hashlib
        encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':')).encode()
        if metadata.get('identity_sha256') != hashlib.sha256(encoded).hexdigest():
            raise ValueError('CLIP-L source identity digest mismatch')
    if verify_hash and file_sha256(path) != metadata['features_sha256']:
        raise ValueError('CLIP-L target SHA-256 mismatch: {}'.format(path))
    ids = np.load(directory / '{}_image_ids.npy'.format(split), allow_pickle=False)
    offsets = np.load(directory / '{}_caption_offsets.npy'.format(split), allow_pickle=False)
    caption_ids = np.load(directory / '{}_caption_image_ids.npy'.format(split), allow_pickle=False)
    tokens = np.load(path, mmap_mode='r', allow_pickle=False)
    if (not np.array_equal(ids, expected_ids) or offsets.shape != (len(ids) + 1,)
            or offsets[0] != 0 or offsets[-1] != len(tokens) or np.any(np.diff(offsets) < 1)
            or tokens.shape != (int(offsets[-1]), 77, 768) or tokens.dtype != np.float16
            or not np.array_equal(caption_ids, np.repeat(ids, np.diff(offsets)))):
        raise ValueError('CLIP-L target rows, IDs, caption offsets, shape or dtype do not match')
    return {'tokens': tokens, 'offsets': offsets, 'ids': ids, 'metadata': metadata,
            'identity': identity, 'identity_sha256': file_sha256(identity_path),
            'path': path, 'features_sha256': metadata['features_sha256']}


class TextTargetDataset(Dataset):
    def __init__(self, brain_dataset, targets):
        self.brain_dataset = brain_dataset
        self.targets = targets

    def __len__(self):
        return len(self.brain_dataset)

    def __getitem__(self, index):
        item = self.brain_dataset[index]
        row = self.brain_dataset.image_index(index)
        target = np.array(self.targets['tokens'][int(self.targets['offsets'][row])],
                          dtype=np.float32, copy=True)
        if not np.isfinite(target).all():
            raise FloatingPointError('Nonfinite CLIP-L target at source row {}'.format(row))
        item['target'] = torch.from_numpy(target)
        item['source_row'] = row
        return item


def train_target_mean(targets, train_rows, chunk_size=32):
    total = np.zeros((77, 768), dtype=np.float64)
    for start in range(0, len(train_rows), chunk_size):
        rows = train_rows[start:start + chunk_size]
        values = np.asarray(targets['tokens'][targets['offsets'][rows]], dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite CLIP-L training targets')
        total += values.sum(axis=0, dtype=np.float64)
    return (total / len(train_rows)).astype(np.float32)


def autocast_for(device, precision):
    if device.type == 'cuda' and precision == 'fp16':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    if device.type == 'cuda' and precision == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def head_settings(args):
    return {'width': args.width, 'heads': args.heads, 'depth': args.depth,
            'mlp_ratio': args.mlp_ratio, 'dropout': args.dropout}


def training_settings(args):
    return {'epochs': 1 if args.smoke else args.epochs, 'batch_size': args.batch_size,
            'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
            'precision': args.precision, 'seed': args.seed, 'patience': args.patience,
            'checkpoint_every': args.checkpoint_every,
            'debug_train_batches': 2 if args.smoke else 0,
            'debug_val_batches': 2 if args.smoke else 0}


def protocol_for(args, source, targets, train_rows, val_rows):
    import hashlib
    def array_sha(value):
        return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()
    return {
        'protocol': 'subj01_frozen_brainvae_mu_to_sdxl_clipl11_first_caption_v1',
        'subject': 1, 'brain_source': source['identity'],
        'target_representation': TARGET_REPRESENTATION,
        'target_train_path': str(targets['path'].resolve()),
        'target_train_sha256': targets['features_sha256'],
        'target_source_identity_sha256': targets['identity_sha256'],
        'target_offsets_sha256': file_sha256(TARGET_DIR / 'train_caption_offsets.npy'),
        'caption_policy': 'first_nonempty_original_coco_caption_in_source_order',
        'fMRI_aggregation': 'mean_of_repeats_before_frozen_brainvae',
        'brainvae_frozen': True, 'input': 'brainvae_posterior_mean_no_w010',
        'target_train_mean': 'first-caption token mean from internal train images only',
        'train_images': len(train_rows), 'validation_images': len(val_rows),
        'train_rows_sha256': array_sha(train_rows), 'validation_rows_sha256': array_sha(val_rows),
        'validation_source': 'existing_w010_internal_nsd_train_holdout',
        'upstream_validation_note': 'the frozen BrainVAE was previously selected using this same internal split',
        'selection_metric': 'validation_token_mse', 'test_used_for_selection': False,
        'head': head_settings(args), 'training': training_settings(args),
        'debug_training': bool(args.smoke), 'torch_version': str(torch.__version__),
        'code_sha256': {'runner': file_sha256(Path(__file__)),
                        'head': file_sha256(ROOT / 'src/models/sdxl_text_head.py')},
    }


def load_datasets(source, paths, targets, train_rows, val_rows):
    result = {}
    for name, rows in (('train', train_rows), ('val', val_rows)):
        dataset = source_nsd_dataset(source, paths.nsd_root, 1, 'train', rows,
                                     average_repeats=True, load_targets=False,
                                     expected_ids=source['split']['source_image_ids'])
        result[name] = TextTargetDataset(dataset, targets)
    return result


def metrics_update(totals, prediction, target, baseline):
    pred, true = prediction.float(), target.float()
    n = len(pred)
    totals['count'] += n
    totals['mse_sum'] += float((pred - true).square().flatten(1).mean(1).sum())
    totals['baseline_mse_sum'] += float((baseline - true).square().flatten(1).mean(1).sum())
    p = F.normalize((pred - baseline).flatten(1), dim=1, eps=1e-8)
    t = F.normalize((true - baseline).flatten(1), dim=1, eps=1e-8)
    similarity = p @ t.T
    totals['centered_cosine_sum'] += float(similarity.diagonal().sum())
    if n > 1:
        correct = (similarity.diagonal()[:, None] > similarity).float()
        ties = (similarity.diagonal()[:, None] == similarity).float()
        off_diag = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
        totals['pair_wins'] += float((correct + 0.5 * ties)[off_diag].sum())
        totals['pair_count'] += n * (n - 1)


def metric_values(totals):
    n = totals['count']
    if not n:
        raise ValueError('No validation examples')
    return {'images': n, 'token_mse': totals['mse_sum'] / n,
            'train_mean_baseline_mse': totals['baseline_mse_sum'] / n,
            'centered_flat_cosine': totals['centered_cosine_sum'] / n,
            'within_batch_2way': (totals['pair_wins'] / totals['pair_count']
                                  if totals['pair_count'] else None),
            'within_batch_pairs': totals['pair_count']}


def empty_totals():
    return {'count': 0, 'mse_sum': 0., 'baseline_mse_sum': 0., 'centered_cosine_sum': 0.,
            'pair_wins': 0., 'pair_count': 0}


@torch.no_grad()
def validate(brain, head, dataset, device, args):
    brain.eval()
    head.eval()
    totals = empty_totals()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == 'cuda')
    for step, batch in enumerate(loader, 1):
        if STOP_SIGNAL:
            return None
        fmri = batch['fmri'].float().to(device, non_blocking=True)
        target = batch['target'].float().to(device, non_blocking=True)
        mean = brain.encode(fmri).mode().float()
        with autocast_for(device, args.precision):
            prediction = head(mean)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite validation prediction')
        metrics_update(totals, prediction, target, head.target_mean)
        if STOP_SIGNAL:
            return None
        if args.smoke and step >= 2:
            break
    return metric_values(totals)


def run_training(args, device, run):
    source, paths = load_brain_context()
    split = source['split']
    targets = load_targets(TARGET_DIR, 'train', split['source_image_ids'])
    train_rows = split['train_indices'][:8] if args.smoke else split['train_indices']
    val_rows = split['val_indices'][:4] if args.smoke else split['val_indices']
    if np.intersect1d(train_rows, val_rows).size:
        raise ValueError('Training and validation image sets overlap')
    protocol = protocol_for(args, source, targets, train_rows, val_rows)
    datasets = load_datasets(source, paths, targets, train_rows, val_rows)
    checkpoint_dir = run / 'checkpoints'
    last, best = checkpoint_dir / 'last.pt', checkpoint_dir / 'best.pt'
    protocol_path, target_mean_path = run / 'protocol.json', run / 'train_target_mean.npy'
    with mapping_lock(run):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not (last.is_file() and protocol_path.is_file() and target_mean_path.is_file()):
                raise FileNotFoundError('Resume requires last.pt, protocol.json and train_target_mean.npy')
            old = json.loads(protocol_path.read_text(encoding='utf-8'))
            mean_hash = old.pop('train_target_mean_sha256')
            if old != protocol or file_sha256(target_mean_path) != mean_hash:
                raise ValueError('Training data, settings or train-only target mean changed')
            target_mean = np.load(target_mean_path, allow_pickle=False)
        else:
            if last.exists() or best.exists() or protocol_path.exists() or target_mean_path.exists():
                raise FileExistsError('Run already exists; use --resume or choose a new --run-name')
            print('[prepare] computing target mean using {} internal training images'.format(len(train_rows)), flush=True)
            target_mean = train_target_mean(targets, train_rows)
            temp = target_mean_path.with_name(target_mean_path.name + '.tmp')
            with temp.open('wb') as handle:
                np.save(handle, target_mean, allow_pickle=False)
            os.replace(str(temp), str(target_mean_path))
            atomic_json(dict(protocol, train_target_mean_sha256=file_sha256(target_mean_path)), protocol_path)

        seed_all(args.seed)
        brain = frozen_brain(source, device)
        del source['weights']
        head = MeanToCLIPTextHead(torch.from_numpy(target_mean), **head_settings(args)).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                     **({'fused': True} if device.type == 'cuda' else {}))
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and args.precision == 'fp16'),
                                      init_scale=128.)
        state = {'epoch': 0, 'next_batch': 0, 'global_step': 0, 'sum_loss': 0., 'samples': 0,
                 'best_score': float('inf'), 'best_epoch': 0, 'stale': 0, 'history': [],
                 'amp_skipped_updates': 0, 'finished': False}
        rng = None
        if args.resume:
            saved = torch.load(last, map_location='cpu', weights_only=False)
            if saved['protocol'] != protocol or saved['train_target_mean_sha256'] != file_sha256(target_mean_path):
                raise ValueError('Checkpoint protocol differs from current run')
            head.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            state, rng = saved['state'], saved['rng']
            if state['best_epoch'] and not best.is_file():
                raise FileNotFoundError('Resume requires the selected best.pt')
            del saved

        def model_payload():
            return {'model': head.state_dict(), 'protocol': protocol,
                    'head_config': head_settings(args),
                    'train_target_mean_sha256': file_sha256(target_mean_path)}

        def save_last():
            save_torch(dict(model_payload(), state=state, optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), rng=capture_rng_state()), last)

        if not args.resume:
            save_last()
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        for epoch in range(state['epoch'], training_settings(args)['epochs']):
            if state['finished']:
                break
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True,
                                generator=generator, num_workers=0, pin_memory=device.type == 'cuda')
            iterator = iter(loader)
            limit = min(2, len(loader)) if args.smoke else len(loader)
            for _ in range(state['next_batch']):
                next(iterator)
            if rng is not None:
                restore_rng_state(rng)
                rng = None
            head.train()
            for step in range(state['next_batch'], limit):
                batch = next(iterator)
                fmri = batch['fmri'].float().to(device, non_blocking=True)
                target = batch['target'].float().to(device, non_blocking=True)
                with torch.no_grad():
                    mean = brain.encode(fmri).mode().float()
                if tuple(mean.shape[1:]) != (256, 1664) or not torch.isfinite(mean).all():
                    raise FloatingPointError('Invalid frozen BrainVAE posterior mean')
                optimizer.zero_grad(set_to_none=True)
                with autocast_for(device, args.precision):
                    prediction = head(mean)
                loss = F.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0,
                                               error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                state['amp_skipped_updates'] += int(scaler.get_scale() < previous_scale)
                if any(p.grad is not None for p in brain.parameters()):
                    raise RuntimeError('Frozen BrainVAE unexpectedly received gradients')
                n = len(fmri)
                state.update(next_batch=step + 1, global_step=state['global_step'] + 1,
                             sum_loss=state['sum_loss'] + float(loss.detach()) * n,
                             samples=state['samples'] + n)
                if state['global_step'] % args.checkpoint_every == 0 or STOP_SIGNAL:
                    save_last()
                    print('[train] epoch {} batch {}/{} loss {:.6f}'.format(
                        epoch + 1, step + 1, limit, float(loss.detach())), flush=True)
                if STOP_SIGNAL:
                    return 128 + STOP_SIGNAL
            if state['amp_skipped_updates'] == state['global_step']:
                raise FloatingPointError('All optimizer updates were skipped')
            val = validate(brain, head, datasets['val'], device, args)
            if val is None:
                # Do not select a model from a partial validation pass. The
                # completed training batches will be skipped after resuming.
                save_last()
                print('[paused] partial validation discarded; resume to rerun validation', flush=True)
                return 128 + STOP_SIGNAL
            improved = val['token_mse'] < state['best_score']
            if improved:
                state.update(best_score=val['token_mse'], best_epoch=epoch + 1, stale=0)
                save_torch(model_payload(), best)
            else:
                state['stale'] += 1
            row = {'epoch': epoch + 1, 'train_mse': state['sum_loss'] / state['samples'],
                   'validation': val, 'best_epoch': state['best_epoch'],
                   'amp_skipped_updates': state['amp_skipped_updates']}
            state['history'].append(row)
            state.update(epoch=epoch + 1, next_batch=0, sum_loss=0., samples=0,
                         finished=(epoch + 1 >= training_settings(args)['epochs'] or
                                   (args.patience > 0 and state['stale'] >= args.patience)))
            save_last()
            history_temp = run / 'history.jsonl.tmp'
            history_temp.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in state['history']),
                                    encoding='utf-8')
            os.replace(str(history_temp), str(run / 'history.jsonl'))
            print('[epoch] {}'.format(json.dumps(row, ensure_ascii=False)), flush=True)
        if not state['finished'] or not best.is_file():
            raise RuntimeError('Training ended without a selected model')
        atomic_json({'status': 'complete', 'best_epoch': state['best_epoch'],
                     'best_validation_mse': state['best_score'], 'best_sha256': file_sha256(best),
                     'test_used_for_selection': False, 'debug_training': bool(args.smoke)},
                    run / 'training_complete.json')
        print('[complete] {}'.format(run), flush=True)
    return 0


def load_selected_head(run, device):
    path, done_path = run / 'checkpoints/best.pt', run / 'training_complete.json'
    done = json.loads(done_path.read_text(encoding='utf-8'))
    if done.get('status') != 'complete' or file_sha256(path) != done.get('best_sha256'):
        raise ValueError('Selected text head is missing or differs from its completion record')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    protocol = saved['protocol']
    if protocol['code_sha256']['head'] != file_sha256(ROOT / 'src/models/sdxl_text_head.py'):
        raise ValueError('Text-head implementation differs from the one used for training')
    if bool(done['debug_training']) != bool(protocol['debug_training']):
        raise ValueError('Debug/formal training identities do not match')
    recorded_protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    if (recorded_protocol != dict(protocol, train_target_mean_sha256=saved['train_target_mean_sha256'])
            or file_sha256(run / 'train_target_mean.npy') != saved['train_target_mean_sha256']):
        raise ValueError('Training protocol or train-only target mean differs from the checkpoint')
    target_mean = saved['model']['target_mean']
    head = MeanToCLIPTextHead(target_mean, **saved['head_config'])
    head.load_state_dict(saved['model'], strict=True)
    return head.to(device).eval().requires_grad_(False), protocol, done, path


@torch.no_grad()
def run_prediction(args, device, run):
    head, protocol, completion, path = load_selected_head(run, device)
    source, paths = load_brain_context()
    if source['identity'] != protocol['brain_source']:
        raise ValueError('Current frozen BrainVAE differs from the trained head source')
    indices = np.arange(len(source['split']['test_image_ids']))
    if args.smoke:
        indices = indices[:2]
    dataset = source_nsd_dataset(source, paths.nsd_root, 1, 'test', indices,
                                 average_repeats=True, load_targets=False,
                                 expected_ids=source['split']['test_image_ids'])
    brain = frozen_brain(source, device)
    del source['weights']
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')
    out = run / 'nsd_test'
    out.mkdir(exist_ok=True)
    final = out / 'predicted_clip_l11.npy'
    partial = out / 'predicted_clip_l11.npy.partial'
    if final.exists() or partial.exists() or (out / 'prediction_complete.json').exists():
        raise FileExistsError('Prediction output already exists; use a new run directory')
    values = np.lib.format.open_memmap(str(partial), mode='w+', dtype=np.float32,
                                       shape=(len(dataset), 77, 768))
    collected, pos = [], 0
    for batch in loader:
        mean = brain.encode(batch['fmri'].float().to(device, non_blocking=True)).mode().float()
        with autocast_for(device, args.precision):
            pred = head(mean)
        pred = pred.float().cpu().numpy()
        if pred.shape[1:] != (77, 768) or not np.isfinite(pred).all():
            raise FloatingPointError('Invalid predicted test text tokens')
        values[pos:pos + len(pred)] = pred
        pos += len(pred)
        collected.extend(map(int, batch['image_id']))
        print('[predict] {}/{}'.format(pos, len(dataset)), flush=True)
    values.flush()
    del values
    os.replace(str(partial), str(final))
    ids_path = out / 'image_ids.npy'
    np.save(ids_path, np.asarray(collected, dtype=np.int64), allow_pickle=False)
    atomic_json({'status': 'complete', 'subject': 1, 'split': 'nsd_test', 'images': pos,
                 'shape': [pos, 77, 768], 'dtype': 'float32',
                 'source_brainvae_sha256': source['identity']['brainvae_checkpoint_sha256'],
                 'head_checkpoint': str(path.resolve()), 'head_sha256': completion['best_sha256'],
                 'features_sha256': file_sha256(final), 'debug_prediction': bool(args.smoke),
                 'image_ids_sha256': file_sha256(ids_path),
                 'test_fmri_source': file_signature(paths.nsd_root / 'subj01/nsd_test_fmri_scale_sub1.npy'),
                 'test_image_id_source': file_signature(paths.nsd_root / 'subj01/nsd_test_image_id_sub1.npy'),
                 'head_precision': args.precision, 'brainvae_encoding_precision': 'fp32',
                 'prediction_code_sha256': file_sha256(Path(__file__)),
                 'test_text_used_as_input': False}, out / 'prediction_complete.json')
    print('[complete] {}'.format(final), flush=True)
    return 0


def run_evaluation(args, device, run):
    head, protocol, completion, path = load_selected_head(run, device)
    source, _ = load_brain_context()
    if source['identity'] != protocol['brain_source']:
        raise ValueError('Current BrainVAE differs from the trained head source')
    targets = load_targets(TARGET_DIR, 'test', source['split']['test_image_ids'])
    out = run / 'nsd_test'
    prediction_path = out / 'predicted_clip_l11.npy'
    info = json.loads((out / 'prediction_complete.json').read_text(encoding='utf-8'))
    if (info.get('status') != 'complete' or info.get('test_text_used_as_input') is not False
            or bool(info.get('debug_prediction')) != bool(args.smoke)
            or info['head_sha256'] != completion['best_sha256']
            or file_sha256(prediction_path) != info['features_sha256']):
        raise ValueError('Prediction identity or hash mismatch')
    pred = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
    ids = np.load(out / 'image_ids.npy', allow_pickle=False)
    if (file_sha256(out / 'image_ids.npy') != info['image_ids_sha256']
            or not np.array_equal(ids, targets['ids'][:len(ids)]) or pred.shape != (len(ids), 77, 768)):
        raise ValueError('Prediction IDs or shape differ from held-out targets')
    totals = empty_totals()
    with torch.no_grad():
        for start in range(0, len(ids), args.batch_size):
            end = min(start + args.batch_size, len(ids))
            p = torch.from_numpy(np.array(pred[start:end], dtype=np.float32, copy=True)).to(device)
            t = torch.from_numpy(np.array(targets['tokens'][targets['offsets'][start:end]],
                                          dtype=np.float32, copy=True)).to(device)
            metrics_update(totals, p, t, head.target_mean)
    results = {'protocol': 'nsd_test_first_caption_text_feature_evaluation_v1',
               'metrics': metric_values(totals), 'debug_evaluation': bool(args.smoke),
               'head_sha256': completion['best_sha256'], 'prediction_sha256': info['features_sha256'],
               'target_test_sha256': targets['features_sha256'], 'test_used_for_selection': False}
    destination = out / 'text_feature_metrics.json'
    if destination.exists():
        raise FileExistsError('Test evaluation already exists: {}'.format(destination))
    atomic_json(results, destination)
    print('[evaluate] {}'.format(json.dumps(results, ensure_ascii=False)), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--action', choices=('train', 'predict', 'evaluate'), default='train')
    parser.add_argument('--run-name', default='clipl11_first_caption_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp16', 'bf16', 'fp32'), default='fp16')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--mlp-ratio', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='use isolated tiny debug train/val/test subsets')
    args = parser.parse_args(argv)
    validate_tag(args.run_name)
    if min(args.batch_size, args.epochs, args.checkpoint_every, args.width, args.heads,
           args.depth, args.mlp_ratio, args.threads) < 1 or args.patience < 0:
        parser.error('Invalid positive integer argument')
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        parser.error('Invalid learning rate, weight decay or dropout')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on the GPU host')
    if device.type == 'cpu' and args.precision != 'fp32':
        raise ValueError('CPU execution requires --precision fp32')
    if device.type == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support BF16')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = (DEBUG_ROOT if args.smoke else FORMAL_ROOT) / args.run_name
    if args.action == 'train':
        return run_training(args, device, run)
    if args.resume:
        parser.error('--resume only applies to training')
    if args.action == 'predict':
        return run_prediction(args, device, run)
    return run_evaluation(args, device, run)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
