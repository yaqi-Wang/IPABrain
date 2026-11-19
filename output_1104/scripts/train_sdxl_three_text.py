#!/usr/bin/env python3
"""Train three matched SDXL text conditions from one frozen BrainVAE mean.

Outputs: CLIP-L [77,768], OpenCLIP-bigG penultimate [77,1280], and bigG
projected pooled [1280]. All labels use the same first COCO caption per image.
NSD test targets are accessed only by the explicit evaluate action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

import extract_sdxl_clip_l11 as extract_common
import extract_sdxl_bigg_text as bigg_extract
import train_sdxl_clipl_head as common
from src.data.scheme1 import file_sha256, file_signature, validate_tag
from src.models.sdxl_three_text_head import ThreeConditionTextHead
from src.utils.clip_mapping import atomic_json, frozen_brain, mapping_lock, source_nsd_dataset
from train_brainvae_scheme1 import capture_rng_state, restore_rng_state

CLIPL_DIR = ROOT / 'output_1104/nsd_subj01/sdxl_clip_l11_all_both_v1'
BIGG_DIR = ROOT / 'output_1104/nsd_subj01/sdxl_bigg_text_first_both_v1'
FORMAL_ROOT = ROOT / 'output_1104/nsd_subj01/three_text_prediction'
DEBUG_ROOT = ROOT / 'output_1104/debug/three_text_prediction'
BRANCHES = ('clip_l', 'bigg_tokens', 'bigg_pooled')
SHAPES = {'clip_l': (77, 768), 'bigg_tokens': (77, 1280), 'bigg_pooled': (1280,)}
STOP_SIGNAL = 0


def on_signal(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum
    print('[signal] {}: stopping at a safe batch boundary'.format(signum), flush=True)


def load_targets(split, expected_ids, bigg_dir, smoke):
    """Verify both encoders refer to the same NSD image and annotation ID."""
    clipl = common.load_targets(CLIPL_DIR, split, expected_ids, verify_hash=True)
    identity_path = bigg_dir / 'source_identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    meta = json.loads((bigg_dir / '{}_bigg.meta.json'.format(split)).read_text(encoding='utf-8'))
    if (identity.get('subject') != 1 or identity.get('caption_policy') != 'first_nonempty_coco_per_image'
            or identity.get('tokens_representation') != bigg_extract.TOKEN_REPR
            or identity.get('pooled_representation') != bigg_extract.POOLED_REPR
            or meta.get('identity_sha256') != extract_common.digest_json(identity)
            or meta.get('clip_l_source_identity_sha256') != clipl['identity_sha256']
            or (not smoke and identity.get('max_images_per_split') != 0)):
        raise ValueError('BigG targets are incomplete, wrong-layer, or not paired with this CLIP-L source')
    ids_path = bigg_dir / '{}_image_ids.npy'.format(split)
    annotation_path = bigg_dir / '{}_caption_annotation_ids.npy'.format(split)
    if (file_sha256(ids_path) != meta['image_ids_sha256']
            or file_sha256(annotation_path) != meta['caption_annotation_ids_sha256']):
        raise ValueError('BigG IDs or annotation IDs changed')
    ids = np.load(ids_path, allow_pickle=False)
    annotations = np.load(annotation_path, allow_pickle=False)
    clip_annotations = np.load(CLIPL_DIR / '{}_caption_annotation_ids.npy'.format(split), allow_pickle=False)
    expected_annotations = clip_annotations[clipl['offsets'][:-1]][:len(ids)]
    if (not np.array_equal(ids, expected_ids[:len(ids)])
            or not np.array_equal(annotations, expected_annotations)
            or (not smoke and len(ids) != len(expected_ids))):
        raise ValueError('BigG / CLIP-L image or first-caption annotation IDs do not match')
    paths = {name: bigg_dir / '{}_bigg_{}.npy'.format(split, name)
             for name in ('tokens', 'pooled')}
    data = {}
    for name, shape in [('tokens', (len(ids), 77, 1280)), ('pooled', (len(ids), 1280))]:
        if file_sha256(paths[name]) != meta[name + '_sha256']:
            raise ValueError('BigG {} SHA-256 mismatch'.format(name))
        data[name] = np.load(paths[name], mmap_mode='r', allow_pickle=False)
        if data[name].shape != shape or data[name].dtype != np.float16:
            raise ValueError('BigG {} shape/dtype mismatch'.format(name))
    return {'clip_l': clipl, 'bigg_tokens': data['tokens'], 'bigg_pooled': data['pooled'],
            'ids': ids, 'annotation_ids': annotations, 'bigg_meta': meta,
            'bigg_identity_sha256': file_sha256(identity_path), 'bigg_paths': paths}


def target_rows(targets, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= len(targets['ids'])):
        raise ValueError('Target row outside the matched feature range')
    l_indices = targets['clip_l']['offsets'][rows]
    return {'clip_l': np.asarray(targets['clip_l']['tokens'][l_indices], dtype=np.float32),
            'bigg_tokens': np.asarray(targets['bigg_tokens'][rows], dtype=np.float32),
            'bigg_pooled': np.asarray(targets['bigg_pooled'][rows], dtype=np.float32)}


def training_statistics(targets, rows, chunk_size=16):
    sums = {k: np.zeros(SHAPES[k], dtype=np.float64) for k in BRANCHES}
    squares = {k: 0.0 for k in BRANCHES}
    for start in range(0, len(rows), chunk_size):
        batch = target_rows(targets, rows[start:start + chunk_size])
        for name in BRANCHES:
            value = batch[name]
            if not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite {} training target'.format(name))
            sums[name] += value.sum(axis=0, dtype=np.float64)
            squares[name] += float(np.square(value, dtype=np.float64).sum(dtype=np.float64))
    means, scales = {}, {}
    for name in BRANCHES:
        mean64 = sums[name] / len(rows)
        variance = squares[name] / (len(rows) * np.prod(SHAPES[name])) - float(np.square(mean64).mean())
        if not np.isfinite(variance) or variance <= 1e-8:
            raise ValueError('Invalid train-only target variance for {}: {}'.format(name, variance))
        means[name], scales[name] = mean64.astype(np.float32), float(variance)
    return means, scales


class ThreeDataset(Dataset):
    def __init__(self, brain_dataset, targets, mean_cache=None, lookup=None):
        self.brain_dataset, self.targets = brain_dataset, targets
        self.mean_cache, self.lookup = mean_cache, lookup

    def __len__(self):
        return len(self.brain_dataset)

    def __getitem__(self, index):
        row = self.brain_dataset.image_index(index)
        if self.mean_cache is None:
            item = self.brain_dataset[index]
        else:
            cache_index = int(self.lookup[row])
            if cache_index < 0:
                raise ValueError('Brain mean cache does not cover source row {}'.format(row))
            item = {'mean': torch.from_numpy(np.array(self.mean_cache[cache_index], copy=True)),
                    'image_id': int(self.targets['ids'][row])}
        labels = target_rows(self.targets, [row])
        for key in BRANCHES:
            if not np.isfinite(labels[key]).all():
                raise FloatingPointError('Nonfinite {} target for row {}'.format(key, row))
            item[key] = torch.from_numpy(np.array(labels[key][0], copy=True))
        item['source_row'] = row
        return item


def available_memory():
    with open('/proc/meminfo', encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot determine available RAM for brain-mean caching')


@torch.no_grad()
def cache_brain_means(brain, source, paths, rows, batch_size, device):
    needed = len(rows) * 256 * 1664 * 4
    free = available_memory()
    if free < needed + 6 * 1024**3:
        raise RuntimeError('RAM cache needs {:.2f} GiB plus 6 GiB reserve; available {:.2f} GiB. '
                           'Use --cache-means none if needed.'.format(needed / 1024**3, free / 1024**3))
    print('[cache] allocating {:.2f} GiB RAM for {} frozen μ values'.format(needed / 1024**3, len(rows)), flush=True)
    cache = np.empty((len(rows), 256, 1664), dtype=np.float32)
    lookup = np.full(len(source['split']['source_image_ids']), -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows), dtype=np.int64)
    dataset = source_nsd_dataset(source, paths.nsd_root, 1, 'train', rows,
                                 average_repeats=True, load_targets=False,
                                 expected_ids=source['split']['source_image_ids'])
    generator = torch.Generator().manual_seed(12345)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda', generator=generator)
    start_time, pos = time.monotonic(), 0
    for step, batch in enumerate(loader, 1):
        value = brain.encode(batch['fmri'].float().to(device, non_blocking=True)).mode().float()
        if tuple(value.shape[1:]) != (256, 1664) or not torch.isfinite(value).all():
            raise FloatingPointError('Invalid cached BrainVAE posterior mean')
        array = value.cpu().numpy()
        cache[pos:pos + len(array)] = array
        pos += len(array)
        if step % 50 == 0 or pos == len(rows) or STOP_SIGNAL:
            print('[cache] {}/{} {:.1f} images/s'.format(
                pos, len(rows), pos / max(time.monotonic() - start_time, 1e-9)), flush=True)
        if STOP_SIGNAL:
            return None, None
    if pos != len(rows):
        raise RuntimeError('Incomplete in-memory brain-mean cache')
    return cache, lookup


def scales_and_means(path):
    with np.load(path, allow_pickle=False) as data:
        means = {name: data['mean_' + name].astype(np.float32) for name in BRANCHES}
        scales = {name: float(data['var_' + name]) for name in BRANCHES}
    if any(means[k].shape != SHAPES[k] or not np.isfinite(means[k]).all()
           or not np.isfinite(scales[k]) or scales[k] <= 1e-8 for k in BRANCHES):
        raise ValueError('Invalid training-only target statistics')
    return means, scales


def config_head(args):
    return {'width': args.width, 'heads': args.heads, 'depth': args.depth,
            'mlp_ratio': args.mlp_ratio, 'dropout': args.dropout}


def config_training(args):
    return {'epochs': 1 if args.smoke else args.epochs, 'batch_size': args.batch_size,
            'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
            'seed': args.seed, 'precision': args.precision, 'patience': args.patience,
            'checkpoint_every': args.checkpoint_every, 'cache_means': args.cache_means,
            'encode_batch_size': args.encode_batch_size,
            'debug_train_batches': 2 if args.smoke else 0, 'debug_val_batches': 2 if args.smoke else 0}


def protocol_for(args, source, targets, train_rows, val_rows):
    def rows_hash(rows):
        return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()
    return {'protocol': 'subj01_frozen_mu_sdxl_three_text_first_caption_v1', 'subject': 1,
            'brain_source': source['identity'], 'input': 'brainvae_posterior_mean_no_w010',
            'fmri_aggregation': 'mean_of_repeats_before_frozen_brainvae', 'brainvae_frozen': True,
            'caption_policy': 'first_nonempty_coco_annotation_per_image',
            'clip_l_train_sha256': targets['clip_l']['features_sha256'],
            'clip_l_source_identity_sha256': targets['clip_l']['identity_sha256'],
            'bigg_train_tokens_sha256': targets['bigg_meta']['tokens_sha256'],
            'bigg_train_pooled_sha256': targets['bigg_meta']['pooled_sha256'],
            'bigg_source_identity_sha256': targets['bigg_identity_sha256'],
            'train_images': len(train_rows), 'validation_images': len(val_rows),
            'train_rows_sha256': rows_hash(train_rows), 'validation_rows_sha256': rows_hash(val_rows),
            'validation_source': 'existing_w010_internal_nsd_train_holdout',
            'upstream_validation_note': 'frozen BrainVAE was previously selected using this internal split',
            'test_used_for_selection': False, 'head': config_head(args), 'training': config_training(args),
            'loss': 'equal mean of per-branch raw MSE divided by train-only branch variance',
            'debug_training': bool(args.smoke), 'torch_version': str(torch.__version__),
            'code_sha256': {'runner': file_sha256(Path(__file__)),
                            'head': file_sha256(ROOT / 'src/models/sdxl_three_text_head.py'),
                            'common_head': file_sha256(ROOT / 'src/models/sdxl_text_head.py'),
                            'common_runner': file_sha256(Path(common.__file__)),
                            'bigg_extractor': file_sha256(Path(bigg_extract.__file__))}}


def make_datasets(source, paths, targets, rows, cache=None, lookup=None):
    result = {}
    for name in ('train', 'val'):
        brain_data = source_nsd_dataset(source, paths.nsd_root, 1, 'train', rows[name],
                                        average_repeats=True, load_targets=False,
                                        expected_ids=source['split']['source_image_ids'])
        result[name] = ThreeDataset(brain_data, targets, cache, lookup)
    return result


def output_loss(pred, target, scales):
    raw = {k: F.mse_loss(pred[k].float(), target[k].float()) for k in BRANCHES}
    return sum(raw[k] / scales[k] for k in BRANCHES) / len(BRANCHES), raw


def head_input(batch, brain, device):
    if 'mean' in batch:
        return batch['mean'].float().to(device, non_blocking=True)
    with torch.no_grad():
        return brain.encode(batch['fmri'].float().to(device, non_blocking=True)).mode().float()


@torch.no_grad()
def validate(brain, head, dataset, scales, args, device):
    if brain is not None:
        brain.eval()
    head.eval()
    totals = {name: common.empty_totals() for name in BRANCHES}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda', generator=torch.Generator().manual_seed(54321))
    for step, batch in enumerate(loader, 1):
        if STOP_SIGNAL:
            return None
        mean = head_input(batch, brain, device)
        targets = {key: batch[key].float().to(device, non_blocking=True) for key in BRANCHES}
        with common.autocast_for(device, args.precision):
            prediction = head(mean)
        for name in BRANCHES:
            if not torch.isfinite(prediction[name]).all():
                raise FloatingPointError('Nonfinite {} validation prediction'.format(name))
            common.metrics_update(totals[name], prediction[name], targets[name], getattr(head, 'mean_' + name))
        if STOP_SIGNAL:
            return None
        if args.smoke and step >= 2:
            break
    metrics = {name: common.metric_values(totals[name]) for name in BRANCHES}
    score = sum(metrics[name]['token_mse'] / scales[name] for name in BRANCHES) / len(BRANCHES)
    return {'normalized_mse': score, 'branches': metrics}


def train(args, device, run):
    source, paths = common.load_brain_context()
    targets = load_targets('train', source['split']['source_image_ids'], args.bigg_dir, args.smoke)
    train_rows, val_rows = source['split']['train_indices'], source['split']['val_indices']
    if args.smoke:
        train_rows = train_rows[train_rows < len(targets['ids'])][:8]
        val_rows = val_rows[val_rows < len(targets['ids'])][:4]
    if len(train_rows) < 2 or len(val_rows) < 2 or np.intersect1d(train_rows, val_rows).size:
        raise ValueError('Need disjoint nonempty train/validation image subsets')
    rows = {'train': train_rows, 'val': val_rows}
    protocol = protocol_for(args, source, targets, train_rows, val_rows)
    ckpts = run / 'checkpoints'
    last, best, stats_path, protocol_path = (ckpts / 'last.pt', ckpts / 'best.pt',
                                           run / 'train_target_stats.npz', run / 'protocol.json')
    with mapping_lock(run):
        ckpts.mkdir(parents=True, exist_ok=True)
        if args.resume:
            if not all(p.is_file() for p in (last, stats_path, protocol_path)):
                raise FileNotFoundError('Resume requires last.pt, target statistics and protocol')
            old = json.loads(protocol_path.read_text(encoding='utf-8'))
            stats_hash = old.pop('train_target_stats_sha256')
            if old != protocol or file_sha256(stats_path) != stats_hash:
                raise ValueError('Training source, settings, code, or target statistics changed')
            means, scales = scales_and_means(stats_path)
        else:
            if any(p.exists() for p in (last, best, stats_path, protocol_path)):
                raise FileExistsError('Run exists; use --resume or choose a new --run-name')
            print('[prepare] computing train-only means and variances for {} images'.format(len(train_rows)), flush=True)
            means, scales = training_statistics(targets, train_rows)
            temp = stats_path.with_name(stats_path.name + '.tmp')
            with temp.open('wb') as handle:
                np.savez(handle, **{'mean_' + k: means[k] for k in BRANCHES},
                         **{'var_' + k: np.asarray(scales[k], dtype=np.float64) for k in BRANCHES})
            os.replace(str(temp), str(stats_path))
            atomic_json(dict(protocol, train_target_stats_sha256=file_sha256(stats_path)), protocol_path)
        print('[prepare] branch variances {}'.format(json.dumps(scales, sort_keys=True)), flush=True)
        common.seed_all(args.seed)
        brain = frozen_brain(source, device)
        del source['weights']
        head = ThreeConditionTextHead({k: torch.from_numpy(means[k]) for k in BRANCHES},
                                      **config_head(args)).to(device)
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
            if saved['protocol'] != protocol or saved['train_target_stats_sha256'] != file_sha256(stats_path):
                raise ValueError('Resumed three-head checkpoint identity mismatch')
            head.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            state, rng = saved['state'], saved['rng']
            if state['best_epoch'] and not best.is_file():
                raise FileNotFoundError('Selected best.pt is missing')
            del saved

        def model_payload():
            return {'model': head.state_dict(), 'protocol': protocol, 'head_config': config_head(args),
                    'train_target_stats_sha256': file_sha256(stats_path)}

        def save_last():
            common.save_torch(dict(model_payload(), state=state, optimizer=optimizer.state_dict(),
                                   scaler=scaler.state_dict(), rng=capture_rng_state()), last)

        if not args.resume:
            save_last()
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        cache, lookup = None, None
        if args.cache_means == 'ram':
            cache_rows = np.concatenate((train_rows, val_rows))
            cache, lookup = cache_brain_means(brain, source, paths, cache_rows, args.encode_batch_size, device)
            if cache is None:
                save_last()
                return 128 + STOP_SIGNAL
            del brain
            brain = None
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        datasets = make_datasets(source, paths, targets, rows, cache, lookup)
        for epoch in range(state['epoch'], config_training(args)['epochs']):
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
                mean = head_input(batch, brain, device)
                if tuple(mean.shape[1:]) != (256, 1664) or not torch.isfinite(mean).all():
                    raise FloatingPointError('Invalid frozen BrainVAE mean')
                targets_batch = {name: batch[name].float().to(device, non_blocking=True) for name in BRANCHES}
                optimizer.zero_grad(set_to_none=True)
                with common.autocast_for(device, args.precision):
                    prediction = head(mean)
                loss, raw = output_loss(prediction, targets_batch, scales)
                if not torch.isfinite(loss) or not all(torch.isfinite(raw[k]) for k in BRANCHES):
                    raise FloatingPointError('Nonfinite three-condition training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0,
                                               error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                state['amp_skipped_updates'] += int(scaler.get_scale() < previous_scale)
                if brain is not None and any(p.grad is not None for p in brain.parameters()):
                    raise RuntimeError('Frozen BrainVAE unexpectedly received gradients')
                n = len(mean)
                state.update(next_batch=step + 1, global_step=state['global_step'] + 1,
                             sum_loss=state['sum_loss'] + float(loss.detach()) * n,
                             samples=state['samples'] + n)
                if state['global_step'] % args.checkpoint_every == 0 or STOP_SIGNAL:
                    save_last()
                    print('[train] epoch {} batch {}/{} normalized_mse {:.6f}'.format(
                        epoch + 1, step + 1, limit, float(loss.detach())), flush=True)
                if STOP_SIGNAL:
                    return 128 + STOP_SIGNAL
            if state['amp_skipped_updates'] == state['global_step']:
                raise FloatingPointError('All optimization updates were skipped')
            validation = validate(brain, head, datasets['val'], scales, args, device)
            if validation is None:
                save_last()
                print('[paused] partial validation discarded; it will rerun after resume', flush=True)
                return 128 + STOP_SIGNAL
            if validation['normalized_mse'] < state['best_score']:
                state.update(best_score=validation['normalized_mse'], best_epoch=epoch + 1, stale=0)
                common.save_torch(model_payload(), best)
            else:
                state['stale'] += 1
            result = {'epoch': epoch + 1, 'train_normalized_mse': state['sum_loss'] / state['samples'],
                      'validation': validation, 'best_epoch': state['best_epoch'],
                      'amp_skipped_updates': state['amp_skipped_updates']}
            state['history'].append(result)
            state.update(epoch=epoch + 1, next_batch=0, sum_loss=0., samples=0,
                         finished=(epoch + 1 >= config_training(args)['epochs'] or
                                   (args.patience > 0 and state['stale'] >= args.patience)))
            save_last()
            temp = run / 'history.jsonl.tmp'
            temp.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in state['history']),
                            encoding='utf-8')
            os.replace(str(temp), str(run / 'history.jsonl'))
            print('[epoch] {}'.format(json.dumps(result, ensure_ascii=False)), flush=True)
        if not state['finished'] or not best.is_file():
            raise RuntimeError('Training did not produce a completed selected checkpoint')
        atomic_json({'status': 'complete', 'best_epoch': state['best_epoch'],
                     'best_validation_normalized_mse': state['best_score'],
                     'best_sha256': file_sha256(best), 'test_used_for_selection': False,
                     'debug_training': bool(args.smoke)}, run / 'training_complete.json')
        print('[complete] {}'.format(run), flush=True)
    return 0


def selected_head(run, device):
    best, stats_path = run / 'checkpoints/best.pt', run / 'train_target_stats.npz'
    complete = json.loads((run / 'training_complete.json').read_text(encoding='utf-8'))
    if complete.get('status') != 'complete' or file_sha256(best) != complete.get('best_sha256'):
        raise ValueError('Missing or changed selected three-condition head')
    saved = torch.load(best, map_location='cpu', weights_only=False)
    protocol = saved['protocol']
    for key, path in [('head', ROOT / 'src/models/sdxl_three_text_head.py'),
                      ('common_head', ROOT / 'src/models/sdxl_text_head.py')]:
        if protocol['code_sha256'][key] != file_sha256(path):
            raise ValueError('Trained head code changed: {}'.format(path))
    if (bool(protocol['debug_training']) != bool(complete['debug_training'])
            or file_sha256(stats_path) != saved['train_target_stats_sha256']
            or json.loads((run / 'protocol.json').read_text(encoding='utf-8')) !=
            dict(protocol, train_target_stats_sha256=saved['train_target_stats_sha256'])):
        raise ValueError('Head protocol, debug flag or train statistics changed')
    means, scales = scales_and_means(stats_path)
    head = ThreeConditionTextHead({k: torch.from_numpy(means[k]) for k in BRANCHES}, **saved['head_config'])
    head.load_state_dict(saved['model'], strict=True)
    return head.to(device).eval().requires_grad_(False), protocol, complete, scales


class PredictionArrays:
    """A shared progress point for all three test-prediction arrays."""

    def __init__(self, directory, ids, identity):
        self.directory, self.ids, self.identity = directory, ids, identity
        self.finals = {name: directory / 'predicted_{}.npy'.format(name) for name in BRANCHES}
        self.partials = {name: path.with_name(path.name + '.partial') for name, path in self.finals.items()}
        self.progress = directory / 'prediction.progress.json'
        self.complete = directory / 'prediction_complete.json'
        self.ids_path = directory / 'image_ids.npy'
        self.arrays, self.next_index, self.completed = {}, 0, False
        if self.complete.is_file():
            metadata = json.loads(self.complete.read_text(encoding='utf-8'))
            if (metadata.get('identity_sha256') != identity
                    or file_sha256(self.ids_path) != metadata['image_ids_sha256']
                    or not np.array_equal(np.load(self.ids_path, allow_pickle=False), ids)):
                raise ValueError('Completed prediction identity or IDs changed')
            for name in BRANCHES:
                value = np.load(self.finals[name], mmap_mode='r', allow_pickle=False)
                if (value.shape != (len(ids), *SHAPES[name]) or value.dtype != np.float32
                        or file_sha256(self.finals[name]) != metadata['feature_sha256'][name]):
                    raise ValueError('Completed {} prediction changed'.format(name))
            self.completed, self.next_index = True, len(ids)
            return
        exists = any(path.exists() for path in list(self.finals.values()) + list(self.partials.values()))
        if exists or self.progress.exists() or self.ids_path.exists():
            if not self.progress.is_file() or not self.ids_path.is_file():
                raise ValueError('Partial prediction lacks progress or image IDs')
            state = json.loads(self.progress.read_text(encoding='utf-8'))
            if (state.get('identity_sha256') != identity or state.get('count') != len(ids)
                    or not np.array_equal(np.load(self.ids_path, allow_pickle=False), ids)):
                raise ValueError('Partial prediction belongs to different weights, data or settings')
            self.next_index = int(state['next_index'])
            if not 0 <= self.next_index <= len(ids):
                raise ValueError('Invalid prediction progress')
            for name in BRANCHES:
                if self.finals[name].exists() and (self.partials[name].exists() or self.next_index != len(ids)):
                    raise ValueError('Ambiguous final/partial {} prediction'.format(name))
                working = self.finals[name] if self.finals[name].exists() else self.partials[name]
                if not working.is_file():
                    raise ValueError('Missing paired prediction: {}'.format(working))
                value = np.load(working, mmap_mode='r+', allow_pickle=False)
                if value.shape != (len(ids), *SHAPES[name]) or value.dtype != np.float32:
                    raise ValueError('Partial {} prediction shape/dtype mismatch'.format(name))
                self.arrays[name] = value
        else:
            import shutil
            required = sum(len(ids) * int(np.prod(SHAPES[k])) * 4 for k in BRANCHES)
            free = shutil.disk_usage(str(directory)).free
            if free < required + 1024**3:
                raise RuntimeError('Insufficient disk space for three test predictions')
            extract_common.save_array_once(self.ids_path, ids)
            for name in BRANCHES:
                self.arrays[name] = np.lib.format.open_memmap(
                    str(self.partials[name]), mode='w+', dtype=np.float32, shape=(len(ids), *SHAPES[name]))
            self.checkpoint(0)

    def checkpoint(self, index):
        for value in self.arrays.values():
            value.flush()
        self.next_index = int(index)
        atomic_json({'identity_sha256': self.identity, 'count': len(self.ids),
                     'next_index': self.next_index}, self.progress)

    def finish(self, metadata):
        if self.completed:
            return
        if self.next_index != len(self.ids):
            raise ValueError('Cannot finalize incomplete three-condition predictions')
        for value in self.arrays.values():
            value.flush()
        self.arrays.clear()
        for name in BRANCHES:
            if self.partials[name].exists():
                os.replace(str(self.partials[name]), str(self.finals[name]))
        atomic_json(dict(metadata, status='complete', identity_sha256=self.identity,
                         feature_sha256={k: file_sha256(self.finals[k]) for k in BRANCHES},
                         image_ids_sha256=file_sha256(self.ids_path)), self.complete)
        self.progress.unlink()
        self.completed = True


@torch.no_grad()
def predict(args, device, run):
    head, protocol, complete, _ = selected_head(run, device)
    source, paths = common.load_brain_context()
    if source['identity'] != protocol['brain_source']:
        raise ValueError('Frozen BrainVAE differs from the trained source')
    rows = np.arange(len(source['split']['test_image_ids']))
    if args.smoke:
        rows = rows[:2]
    ids = source['split']['test_image_ids'][rows].astype(np.int64, copy=False)
    test_fmri_source = file_signature(paths.nsd_root / 'subj01/nsd_test_fmri_scale_sub1.npy')
    prediction_identity = {
        'head_sha256': complete['best_sha256'],
        'brainvae_sha256': source['identity']['brainvae_checkpoint_sha256'],
        'test_fmri_source': test_fmri_source,
        'image_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'head_precision': args.precision, 'brainvae_encoding_precision': 'fp32',
        'batch_size': args.batch_size, 'debug_prediction': bool(args.smoke),
        'prediction_code_sha256': file_sha256(Path(__file__)),
    }
    dataset = source_nsd_dataset(source, paths.nsd_root, 1, 'test', rows,
                                 average_repeats=True, load_targets=False,
                                 expected_ids=source['split']['test_image_ids'])
    brain = frozen_brain(source, device)
    del source['weights']
    out = run / 'nsd_test'
    out.mkdir(exist_ok=True)
    output = PredictionArrays(out, ids, extract_common.digest_json(prediction_identity))
    if output.completed:
        print('[verified] {}'.format(out), flush=True)
        return 0
    start = output.next_index
    loader = DataLoader(torch.utils.data.Subset(dataset, range(start, len(dataset))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == 'cuda')
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    pos = start
    for step, batch in enumerate(loader, 1):
        mean = brain.encode(batch['fmri'].float().to(device, non_blocking=True)).mode().float()
        with common.autocast_for(device, args.precision):
            prediction = head(mean)
        for name in BRANCHES:
            values = prediction[name].float().cpu().numpy()
            if values.shape != (len(mean), *SHAPES[name]) or not np.isfinite(values).all():
                raise FloatingPointError('Invalid {} predicted tokens'.format(name))
            output.arrays[name][pos:pos + len(values)] = values
        batch_ids = np.asarray(batch['image_id'], dtype=np.int64)
        if not np.array_equal(batch_ids, ids[pos:pos + len(mean)]):
            raise ValueError('Prediction batch image IDs differ from expected test order')
        pos += len(mean)
        if step % 8 == 0 or pos == len(dataset) or STOP_SIGNAL:
            output.checkpoint(pos)
            print('[predict] {}/{}'.format(pos, len(dataset)), flush=True)
        if STOP_SIGNAL:
            return 128 + STOP_SIGNAL
    if output.next_index != len(dataset):
        output.checkpoint(pos)
    output.finish({'subject': 1, 'split': 'nsd_test', 'images': pos,
                   'shapes': {k: [pos, *SHAPES[k]] for k in BRANCHES}, 'dtype': 'float32',
                   'source_brainvae_sha256': source['identity']['brainvae_checkpoint_sha256'],
                   'head_sha256': complete['best_sha256'], 'test_fmri_source': test_fmri_source,
                   'head_precision': args.precision, 'brainvae_encoding_precision': 'fp32',
                   'test_text_used_as_input': False, 'debug_prediction': bool(args.smoke)})
    print('[complete] {}'.format(out), flush=True)
    return 0


def evaluate(args, device, run):
    head, protocol, complete, scales = selected_head(run, device)
    source, _ = common.load_brain_context()
    if source['identity'] != protocol['brain_source']:
        raise ValueError('Frozen BrainVAE differs from the trained source')
    targets = load_targets('test', source['split']['test_image_ids'], args.bigg_dir, args.smoke)
    out = run / 'nsd_test'
    info = json.loads((out / 'prediction_complete.json').read_text(encoding='utf-8'))
    ids_path = out / 'image_ids.npy'
    ids = np.load(ids_path, allow_pickle=False)
    if (info.get('status') != 'complete' or info.get('test_text_used_as_input') is not False
            or bool(info.get('debug_prediction')) != bool(args.smoke)
            or info['head_sha256'] != complete['best_sha256']
            or file_sha256(ids_path) != info['image_ids_sha256']
            or not np.array_equal(ids, targets['ids'][:len(ids)])):
        raise ValueError('Test prediction provenance or IDs differ')
    predicted = {}
    for name in BRANCHES:
        p = out / 'predicted_{}.npy'.format(name)
        if file_sha256(p) != info['feature_sha256'][name]:
            raise ValueError('Changed {} prediction'.format(name))
        predicted[name] = np.load(p, mmap_mode='r', allow_pickle=False)
        if predicted[name].shape != (len(ids), *SHAPES[name]):
            raise ValueError('Invalid {} prediction shape'.format(name))
    destination = out / 'text_feature_metrics.json'
    if destination.exists():
        previous = json.loads(destination.read_text(encoding='utf-8'))
        if (previous.get('protocol') != 'nsd_test_three_text_first_caption_feature_evaluation_v1'
                or previous.get('head_sha256') != complete['best_sha256']
                or previous.get('prediction_sha256') != info['feature_sha256']
                or previous.get('target_clip_l_sha256') != targets['clip_l']['features_sha256']
                or previous.get('target_bigg_tokens_sha256') != targets['bigg_meta']['tokens_sha256']
                or previous.get('target_bigg_pooled_sha256') != targets['bigg_meta']['pooled_sha256']
                or previous.get('test_used_for_selection') is not False
                or bool(previous.get('debug_evaluation')) != bool(args.smoke)):
            raise ValueError('Existing test text-feature evaluation has different sources')
        print('[verified] {}'.format(destination), flush=True)
        return 0
    totals = {name: common.empty_totals() for name in BRANCHES}
    with torch.no_grad():
        for start in range(0, len(ids), args.batch_size):
            end = min(start + args.batch_size, len(ids))
            labels = target_rows(targets, np.arange(start, end))
            for name in BRANCHES:
                p = torch.from_numpy(np.array(predicted[name][start:end], dtype=np.float32, copy=True)).to(device)
                t = torch.from_numpy(np.array(labels[name], dtype=np.float32, copy=True)).to(device)
                common.metrics_update(totals[name], p, t, getattr(head, 'mean_' + name))
    metrics = {name: common.metric_values(totals[name]) for name in BRANCHES}
    result = {'protocol': 'nsd_test_three_text_first_caption_feature_evaluation_v1',
              'metrics': metrics,
              'normalized_mse': sum(metrics[k]['token_mse'] / scales[k] for k in BRANCHES) / len(BRANCHES),
              'head_sha256': complete['best_sha256'], 'prediction_sha256': info['feature_sha256'],
              'target_clip_l_sha256': targets['clip_l']['features_sha256'],
              'target_bigg_tokens_sha256': targets['bigg_meta']['tokens_sha256'],
              'target_bigg_pooled_sha256': targets['bigg_meta']['pooled_sha256'],
              'test_used_for_selection': False, 'debug_evaluation': bool(args.smoke)}
    atomic_json(result, destination)
    print('[evaluate] {}'.format(json.dumps(result, ensure_ascii=False)), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('train', 'predict', 'evaluate'), default='train')
    parser.add_argument('--run-name', default='three_text_first_caption_v1')
    parser.add_argument('--bigg-dir', type=Path, default=BIGG_DIR)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp16', 'bf16', 'fp32'), default='fp16')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--encode-batch-size', type=int, default=4)
    parser.add_argument('--cache-means', choices=('ram', 'none'), default='ram')
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
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    validate_tag(args.run_name)
    if min(args.batch_size, args.encode_batch_size, args.epochs, args.checkpoint_every, args.width,
           args.heads, args.depth, args.mlp_ratio, args.threads) < 1 or args.patience < 0:
        parser.error('Invalid integer argument')
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.dropout < 1):
        parser.error('Invalid learning rate, weight decay or dropout')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on the GPU host')
    if device.type == 'cpu' and args.precision != 'fp32':
        raise ValueError('CPU requires --precision fp32')
    if device.type == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support BF16')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = (DEBUG_ROOT if args.smoke else FORMAL_ROOT) / args.run_name
    if args.action == 'train':
        return train(args, device, run)
    if args.resume:
        parser.error('--resume applies only to training')
    if args.action == 'predict':
        return predict(args, device, run)
    return evaluate(args, device, run)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
