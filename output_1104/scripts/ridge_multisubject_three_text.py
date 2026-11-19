#!/usr/bin/env python3
"""GPU multi-output dual ridge from frozen BrainVAE means to SDXL text.

Uses each subject's selected BrainVAE and one common ridge penalty for
all three outputs. The kernel is computed over every flattened μ coordinate.
Only internal validation selects alpha; test caption features are loaded
after test fMRI predictions have been saved.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'output_1104/scripts'))

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_sdxl_clipl_head as common
import train_sdxl_three_text as three
from src.data.scheme1 import file_sha256, validate_tag
from src.utils.clip_mapping import atomic_json, frozen_brain, mapping_lock, source_nsd_dataset

BRANCHES, SHAPES = three.BRANCHES, three.SHAPES
INPUT_SHAPE = (256, 1664)
INPUT_DIM = int(np.prod(INPUT_SHAPE))
FORMAL_ROOT = ROOT / 'output_1104/nsd_subj01/ridge_three_text'
DEBUG_ROOT = ROOT / 'output_1104/debug/ridge_three_text'
RUN_STATE_FILE = None
INTERRUPTED_SIGNAL = 0


def load_subject_context(subject):
    selection_path = ROOT / f'outputs_new/w010_multisubject_v1/subj{subject:02d}/selection.json'
    selected = json.loads(selection_path.read_text())['selected']
    config_path, checkpoint = Path(selected['config']), Path(selected['checkpoint'])
    if file_sha256(config_path) != selected['config_sha256'] or file_sha256(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Selected w010 config or checkpoint digest mismatch')
    mapping, brain_cfg, paths = common.load_head_source_config(config_path, ROOT)
    source = common.load_brain_source(mapping, brain_cfg, ROOT, subject)
    protocol = json.loads((checkpoint.parent.parent / 'protocol.json').read_text())
    if (source['identity']['subject'] != subject or source['identity'].get('source_debug_training')
            or protocol['subject'] != subject
            or protocol['source_identity']['brainvae_checkpoint_sha256'] != source['identity']['brainvae_checkpoint_sha256']):
        raise ValueError('Selected w010 and BrainVAE subject or source do not match')
    common.check_source_files(paths.nsd_root, subject, source['manifest'],
                              ('train_fmri', 'train_ids', 'test_fmri', 'test_ids'))
    return source, paths


def load_subject_targets(directory, split, expected_ids, subject):
    info = json.loads((directory / 'extraction_complete.json').read_text())
    if (info.get('status') != 'complete' or info.get('subject') != subject or info.get('debug')
            or info.get('caption_policy') != 'first_nonempty_coco_caption'):
        raise ValueError('Expected complete formal subject-specific caption extraction')
    selected = json.loads((ROOT / f'outputs_new/w010_multisubject_v1/subj{subject:02d}/selection.json').read_text())['selected']
    if info.get('selected_w010_checkpoint_sha256') != selected['checkpoint_sha256']:
        raise ValueError('Text extraction belongs to a different selected checkpoint')
    names = [split + '_image_ids.npy', split + '_captions.jsonl'] + [split + '_' + name + '.npy' for name in BRANCHES]
    for name in names:
        path = directory / name
        if not path.is_file() or file_sha256(path) != info['files'][name]['sha256']:
            raise ValueError('Target file missing or hash mismatch: ' + str(path))
    ids = np.load(directory / names[0], allow_pickle=False)
    if not np.array_equal(ids, expected_ids):
        raise ValueError('Caption targets and subject image IDs differ')
    values = {}
    for name in BRANCHES:
        value = np.load(directory / (split + '_' + name + '.npy'), mmap_mode='r', allow_pickle=False)
        if value.shape != (len(ids), *SHAPES[name]) or value.dtype != np.float16:
            raise ValueError('Target shape/dtype mismatch: ' + name)
        values[name] = value
    return {'ids': ids, 'clip_l': {'tokens': values['clip_l'], 'offsets': np.arange(len(ids)),
                                  'features_sha256': info['files'][split+'_clip_l.npy']['sha256']},
            'bigg_tokens': values['bigg_tokens'], 'bigg_pooled': values['bigg_pooled'],
            'bigg_meta': {'tokens_sha256': info['files'][split+'_bigg_tokens.npy']['sha256'],
                          'pooled_sha256': info['files'][split+'_bigg_pooled.npy']['sha256']}}


@torch.no_grad()
def cache_subject_means(brain, source, paths, rows, batch_size, device, subject):
    needed = len(rows) * INPUT_DIM * 4
    if three.available_memory() < needed + 6 * 1024**3:
        raise RuntimeError('Insufficient RAM for subject BrainVAE mean cache and 6 GiB reserve')
    cache = np.empty((len(rows), *INPUT_SHAPE), dtype=np.float32)
    lookup = np.full(len(source['split']['source_image_ids']), -1, dtype=np.int64)
    lookup[rows] = np.arange(len(rows), dtype=np.int64)
    dataset = source_nsd_dataset(source, paths.nsd_root, subject, 'train', rows,
                                 average_repeats=True, load_targets=False,
                                 expected_ids=source['split']['source_image_ids'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    pos = 0
    for batch in loader:
        value = brain.encode(batch['fmri'].float().to(device, non_blocking=True)).mode().float()
        if tuple(value.shape[1:]) != INPUT_SHAPE or not torch.isfinite(value).all():
            raise FloatingPointError('Invalid BrainVAE mean')
        cache[pos:pos+len(value)] = value.cpu().numpy()
        pos += len(value)
        if pos % (batch_size * 50) == 0 or pos == len(rows):
            print(f'[cache] {pos}/{len(rows)}', flush=True)
    if pos != len(rows):
        raise RuntimeError('Incomplete BrainVAE cache')
    return cache, lookup


def arr_digest(value):
    return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()


def parse_grid(text):
    values = [float(x) for x in text.split(',')]
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Every relative alpha must be finite and positive')
    if values != sorted(set(values)):
        raise ValueError('Relative alphas must be unique and increasing')
    return values


def solve_projections(gram, cross, relative_alphas):
    """Return cross @ (gram + alpha*mean(diag(gram))*I)^-1 for each alpha."""
    scale = float(gram.diagonal().mean().item())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Centered training kernel has invalid mean diagonal')
    projections, accepted, diagnostics = [], [], []
    for relative in relative_alphas:
        started = time.monotonic()
        system = gram.clone()
        system.diagonal().add_(relative * scale)
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        status = int(info.item())
        if status != 0:
            diagnostics.append({'relative_alpha': relative, 'status': 'cholesky_failed', 'info': status})
            print('[alpha_skip] relative={} Cholesky info={}'.format(relative, status), flush=True)
            del factor, system
            continue
        solution = torch.cholesky_solve(cross.T.contiguous(), factor)
        # Inspect several columns without making a second large matrix product.
        check = min(8, solution.shape[1])
        residual = system @ solution[:, :check] - cross.T[:, :check]
        relerr = float((torch.linalg.vector_norm(residual) /
                        torch.linalg.vector_norm(cross.T[:, :check]).clamp_min(1e-12)).item())
        if not math.isfinite(relerr) or relerr > 0.001:
            diagnostics.append({'relative_alpha': relative, 'status': 'solve_residual_too_large',
                                'solve_residual': relerr})
            print('[alpha_skip] relative={} solve_residual={:.3g}'.format(relative, relerr), flush=True)
            del factor, system, solution, residual
            continue
        projections.append(solution.T.contiguous())
        accepted.append(relative)
        diagnostics.append({'relative_alpha': relative, 'status': 'ok', 'solve_residual': relerr})
        print('[alpha] relative={} absolute={:.8g} solve_residual={:.3g} seconds={:.1f}'.format(
            relative, relative * scale, relerr, time.monotonic() - started), flush=True)
        del factor, system, solution, residual
    if not projections:
        raise RuntimeError('No numerically reliable ridge regularization candidate remains')
    return torch.stack(projections), scale, accepted, diagnostics


def feature_matrix(targets, name, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if name == 'clip_l':
        source = targets['clip_l']['tokens']
        indices = targets['clip_l']['offsets'][rows]
    else:
        source, indices = targets[name], rows
    values = np.asarray(source[indices])
    if values.dtype != np.float16 or values.shape != (len(rows), *SHAPES[name]):
        raise ValueError('Unexpected {} target shape/dtype'.format(name))
    return values.reshape(len(rows), -1)


def selection_scores(projections, train_y, val_y, mean, scale, chunk, device):
    """One target branch: same training-only mean and loss as the neural head."""
    totals = torch.zeros(projections.shape[0], device=device, dtype=torch.float64)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    for start in range(0, train_y.shape[1], chunk):
        end = min(start + chunk, train_y.shape[1])
        y = torch.from_numpy(np.ascontiguousarray(train_y[:, start:end])).to(device, dtype=torch.float32)
        reference = torch.from_numpy(np.ascontiguousarray(val_y[:, start:end])).to(device, dtype=torch.float32)
        offset = torch.from_numpy(mean[start:end].copy()).to(device)
        if not torch.isfinite(y).all() or not torch.isfinite(reference).all():
            raise FloatingPointError('Nonfinite ridge targets')
        y.sub_(offset)
        reference.sub_(offset)
        error = torch.matmul(projections, y) - reference.unsqueeze(0)
        totals += error.double().square().sum(dim=(1, 2))
        if start == 0 or end == train_y.shape[1] or (start // chunk) % 8 == 0:
            print('[sweep] dimensions {}/{}'.format(end, train_y.shape[1]), flush=True)
    raw = (totals / (len(val_y) * train_y.shape[1])).cpu().numpy()
    return raw, raw / float(scale)


def copy_rows_to_gpu(cache, start, count, device, transfer_batch):
    result = torch.empty((count, INPUT_DIM), device=device, dtype=torch.float32)
    for offset in range(0, count, transfer_batch):
        end = min(offset + transfer_batch, count)
        batch = np.asarray(cache[start + offset:start + end]).reshape(end - offset, INPUT_DIM)
        result[offset:end].copy_(torch.from_numpy(batch).to(device))
    return result


def check_kernel_reference(x, gram, xv=None, cross=None):
    """Compare rows spanning the matrix against a float64 Gram reference.

    This also checks rows beyond the 2**31-element offset of a full run.
    """
    positions = np.unique(np.linspace(0, len(x) - 1, min(17, len(x)), dtype=np.int64))
    indices = torch.from_numpy(positions).to(x.device)
    sampled = x.index_select(0, indices).double()
    reference = sampled @ sampled.T
    actual = gram.index_select(0, indices).index_select(1, indices).double()
    maximum = float((actual - reference).abs().max().item())
    scale = max(float(reference.diagonal().abs().mean().item()), 1e-12)
    relative = maximum / scale
    if not math.isfinite(relative) or relative > 0.0001:
        raise FloatingPointError('Float32 Gram differs from sampled float64 reference: {}'.format(relative))
    result = {'sample_rows': positions.tolist(), 'max_absolute_error': maximum,
              'relative_to_mean_sample_diagonal': relative}
    if xv is not None:
        if cross is None:
            raise ValueError('Validation cross-kernel is required')
        val_positions = np.unique(np.linspace(0, len(xv) - 1, min(17, len(xv)), dtype=np.int64))
        val_indices = torch.from_numpy(val_positions).to(x.device)
        val_sample = xv.index_select(0, val_indices).double()
        reference_cross = val_sample @ sampled.T
        actual_cross = cross.index_select(0, val_indices).index_select(1, indices).double()
        cross_max = float((actual_cross - reference_cross).abs().max().item())
        cross_rel = cross_max / scale
        if not math.isfinite(cross_rel) or cross_rel > 0.0001:
            raise FloatingPointError('Validation kernel differs from float64 reference: {}'.format(cross_rel))
        result.update(validation_sample_rows=val_positions.tolist(),
                      validation_cross_max_absolute_error=cross_max,
                      validation_cross_relative_to_mean_sample_diagonal=cross_rel)
    return result


def block_kernels(x, xv, chunk, use_double=False):
    """High-precision accumulation avoids error across the long input dimension."""
    gram = torch.zeros((len(x), len(x)), device=x.device, dtype=torch.float64)
    cross = torch.zeros((len(xv), len(x)), device=x.device, dtype=torch.float64)
    dimensions = x.shape[1]
    started = time.monotonic()
    for start in range(0, dimensions, chunk):
        end = min(start + chunk, dimensions)
        block = x[:, start:end].contiguous()
        val_block = xv[:, start:end].contiguous()
        if use_double:
            block, val_block = block.double(), val_block.double()
        gram.add_(block @ block.T)
        cross.add_(val_block @ block.T)
        if start == 0 or end == dimensions or (start // chunk) % 8 == 0:
            if x.device.type == 'cuda':
                torch.cuda.synchronize(x.device)
            print('[kernel_blocks] {} {}/{} {:.1f}s'.format(
                'fp64' if use_double else 'fp32_fp64sum', end, dimensions,
                time.monotonic() - started), flush=True)
    gram = ((gram + gram.T) * 0.5).float()
    return gram, cross.float()


def block_cross(query, train, chunk, use_double=False, show_progress=True):
    result = torch.zeros((len(query), len(train)), device=train.device, dtype=torch.float64)
    for start in range(0, train.shape[1], chunk):
        end = min(start + chunk, train.shape[1])
        train_block, query_block = train[:, start:end].contiguous(), query[:, start:end].contiguous()
        if use_double:
            train_block, query_block = train_block.double(), query_block.double()
        result.add_(query_block @ train_block.T)
        if show_progress and (start == 0 or end == train.shape[1] or (start // chunk) % 8 == 0):
            print('[cross_blocks] {}/{}'.format(end, train.shape[1]), flush=True)
    return result.float()


def check_cross_reference(train, query, cross):
    train_rows = np.unique(np.linspace(0, len(train) - 1, min(17, len(train)), dtype=np.int64))
    query_rows = np.unique(np.linspace(0, len(query) - 1, min(17, len(query)), dtype=np.int64))
    ti, qi = torch.from_numpy(train_rows).to(train.device), torch.from_numpy(query_rows).to(train.device)
    train_sample, query_sample = train.index_select(0, ti).double(), query.index_select(0, qi).double()
    reference = query_sample @ train_sample.T
    actual = cross.index_select(0, qi).index_select(1, ti).double()
    maximum = float((actual - reference).abs().max().item())
    scale = max(float(train_sample.square().sum(1).mean().item()), 1e-12)
    relative = maximum / scale
    if not math.isfinite(relative) or relative > 0.0001:
        raise FloatingPointError('Cross-kernel differs from float64 reference: {}'.format(relative))
    return {'max_absolute_error': maximum, 'relative_to_sample_train_diagonal': relative,
            'train_sample_rows': train_rows.tolist(), 'query_sample_rows': query_rows.tolist()}


@torch.no_grad()
def test_kernel(brain, source, paths, rows, train_gpu, input_mean, batch_size, device,
                kernel_chunk=16384, use_double=False, subject=1):
    dataset = source_nsd_dataset(source, paths.nsd_root, subject, 'test', rows,
                                 average_repeats=True, load_targets=False,
                                 expected_ids=source['split']['test_image_ids'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True)
    query = torch.empty((len(rows), INPUT_DIM), device=device, dtype=torch.float32)
    pos, started = 0, time.monotonic()
    for batch in loader:
        mean = brain.encode(batch['fmri'].float().to(device, non_blocking=True)).mode().float()
        if tuple(mean.shape[1:]) != INPUT_SHAPE or not torch.isfinite(mean).all():
            raise FloatingPointError('Invalid test BrainVAE posterior mean')
        expected = source['split']['test_image_ids'][rows[pos:pos + len(mean)]]
        if not np.array_equal(np.asarray(batch['image_id']), expected):
            raise ValueError('Test fMRI image ID order changed')
        x = mean.reshape(len(mean), INPUT_DIM).sub_(input_mean).mul_(INPUT_DIM ** -0.5)
        query[pos:pos + len(mean)].copy_(x)
        pos += len(mean)
        if pos % (batch_size * 50) == 0 or pos == len(rows):
            print('[test_mu] {}/{} {:.1f} images/s'.format(
                pos, len(rows), pos / max(time.monotonic() - started, 1e-9)), flush=True)
    if pos != len(rows):
        raise ValueError('Incomplete test fMRI kernel')
    kernel = block_cross(query, train_gpu, kernel_chunk, use_double=use_double)
    check = check_cross_reference(train_gpu, query, kernel)
    print('[test_kernel] float64_sample_check {}'.format(json.dumps(check, sort_keys=True)), flush=True)
    return kernel


def create_partial(path, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    if path.exists() or partial.exists():
        raise FileExistsError('Output exists: {}'.format(path))
    return np.lib.format.open_memmap(str(partial), mode='w+', dtype=np.float32, shape=shape), partial


def finalize_array(value, partial, path):
    value.flush()
    os.replace(str(partial), str(path))


def save_branch(name, targets, rows, val_rows, means, factor, cross_val, cross_test,
                chunk, model_dir, val_dir, test_dir, device):
    ytrain = feature_matrix(targets, name, rows)
    dims = ytrain.shape[1]
    mean = means[name].reshape(-1)
    weights_path = model_dir / ('dual_' + name + '.npy')
    val_path = val_dir / ('predicted_' + name + '.npy')
    test_path = test_dir / ('predicted_' + name + '.npy')
    weights, wp = create_partial(weights_path, (len(rows), *SHAPES[name]))
    val, vp = create_partial(val_path, (len(val_rows), *SHAPES[name]))
    test, tp = create_partial(test_path, (len(cross_test), *SHAPES[name]))
    wf, vf, tf = weights.reshape(len(rows), dims), val.reshape(len(val_rows), dims), test.reshape(len(cross_test), dims)
    for start in range(0, dims, chunk):
        end = min(start + chunk, dims)
        block = torch.from_numpy(np.ascontiguousarray(ytrain[:, start:end])).to(device, dtype=torch.float32)
        offset = torch.from_numpy(np.ascontiguousarray(mean[start:end])).to(device)
        block.sub_(offset)
        coefficients = torch.cholesky_solve(block, factor)
        pval = cross_val @ coefficients + offset
        ptest = cross_test @ coefficients + offset
        if not all(torch.isfinite(x).all() for x in (coefficients, pval, ptest)):
            raise FloatingPointError('Nonfinite {} coefficients or predictions'.format(name))
        wf[:, start:end] = coefficients.cpu().numpy()
        vf[:, start:end] = pval.cpu().numpy()
        tf[:, start:end] = ptest.cpu().numpy()
        if start == 0 or end == dims or (start // chunk) % 8 == 0:
            print('[fit] {} dimensions {}/{}'.format(name, end, dims), flush=True)
    for array, partial, path in ((weights, wp, weights_path), (val, vp, val_path), (test, tp, test_path)):
        finalize_array(array, partial, path)
    return {'dual_sha256': file_sha256(weights_path), 'validation_sha256': file_sha256(val_path),
            'test_sha256': file_sha256(test_path)}


def evaluate_predictions(directory, targets, rows, means, scales, batch_size, device):
    predictions = {name: np.load(directory / ('predicted_' + name + '.npy'), mmap_mode='r') for name in BRANCHES}
    totals = {name: common.empty_totals() for name in BRANCHES}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            reference = three.target_rows(targets, rows[start:end])
            for name in BRANCHES:
                p = torch.from_numpy(np.array(predictions[name][start:end], copy=True)).to(device)
                t = torch.from_numpy(reference[name]).to(device)
                base = torch.from_numpy(means[name]).to(device)
                common.metrics_update(totals[name], p, t, base)
    metrics = {name: common.metric_values(totals[name]) for name in BRANCHES}
    aggregate = sum(metrics[name]['token_mse'] / scales[name] for name in BRANCHES) / len(BRANCHES)
    return {'normalized_mse': aggregate, 'branches': metrics}


def main():
    global RUN_STATE_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, choices=range(2, 9), required=True)
    parser.add_argument('--target-dir', type=Path)
    parser.add_argument('--run-name', default='ridge_mu_three_text_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--alpha-grid', default='0.0001,0.001,0.01,0.1,1,10,100')
    parser.add_argument('--target-chunk', type=int, default=4096)
    parser.add_argument('--encode-batch-size', type=int, default=4)
    parser.add_argument('--transfer-batch', type=int, default=32)
    parser.add_argument('--kernel-chunk', type=int, default=16384)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-train', type=int, default=32)
    parser.add_argument('--smoke-validation', type=int, default=8)
    args = parser.parse_args()
    validate_tag(args.run_name)
    alphas = parse_grid(args.alpha_grid)
    if min(args.target_chunk, args.encode_batch_size, args.transfer_batch, args.kernel_chunk, args.threads,
           args.smoke_train, args.smoke_validation) < 1:
        raise ValueError('Batch and chunk sizes must be positive')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This full-dimensional ridge experiment requires CUDA')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    subject_root = ROOT / 'output_1104/multisubject_nsd_imagery_v1' / f'subj{args.subject:02d}'
    target_dir = args.target_dir or subject_root / 'coco_text_resumable'
    run = (ROOT / 'output_1104/multisubject_nsd_imagery_v1/debug' / f'subj{args.subject:02d}' / 'ridge'
           if args.smoke else subject_root / 'ridge') / args.run_name
    with mapping_lock(run):
        if any(p.name != '.mapping.lock' for p in run.iterdir()):
            raise FileExistsError('Run already contains files; choose a new --run-name: {}'.format(run))
        RUN_STATE_FILE = run / 'state.json'
        shutil.copyfile(Path(__file__), run / 'runner_snapshot.py')
        model_dir, val_dir, test_dir = run / 'model', run / 'validation', run / 'nsd_test'
        for path in (model_dir, val_dir, test_dir):
            path.mkdir(parents=True)
        source, paths = load_subject_context(args.subject)
        split = source['split']
        rows, val_rows = split['train_indices'], split['val_indices']
        test_rows = np.arange(len(split['test_image_ids']), dtype=np.int64)
        if args.smoke:
            rows, val_rows, test_rows = rows[:args.smoke_train], val_rows[:args.smoke_validation], test_rows[:8]
        if np.intersect1d(rows, val_rows).size or len(rows) < 2 or len(val_rows) < 2:
            raise ValueError('Train and validation splits must be disjoint')
        need_gpu = len(rows) * INPUT_DIM * 4 + 5 * 1024**3
        gpu_free, _ = torch.cuda.mem_get_info(device)
        if gpu_free < need_gpu:
            raise RuntimeError('Need {:.1f} GiB free GPU memory; available {:.1f} GiB'.format(
                need_gpu / 1024**3, gpu_free / 1024**3))
        need_disk = sum(len(rows) * int(np.prod(SHAPES[k])) * 4 for k in BRANCHES)
        need_disk += sum((len(val_rows) + len(test_rows)) * int(np.prod(SHAPES[k])) * 4 for k in BRANCHES)
        if shutil.disk_usage(run).free < need_disk + 2 * 1024**3:
            raise RuntimeError('Insufficient disk for ridge weights and predictions')
        atomic_json({'status': 'running', 'stage': 'load_train_targets'}, run / 'state.json')
        targets = load_subject_targets(target_dir, 'train', split['source_image_ids'], args.subject)
        config = {'protocol': 'multisubject_mu_three_text_multioutput_dual_ridge_v1',
                  'subject': args.subject, 'brain_source': source['identity'],
                  'input': 'frozen_brainvae_mu_flattened_without_w010',
                  'input_centering': 'per-coordinate internal training mean only',
                  'input_scaling': 'divide centered features by sqrt(425984)',
                  'caption_policy': 'first_nonempty_coco_per_image',
                  'target_clip_l_sha256': targets['clip_l']['features_sha256'],
                  'target_bigg_tokens_sha256': targets['bigg_meta']['tokens_sha256'],
                  'target_bigg_pooled_sha256': targets['bigg_meta']['pooled_sha256'],
                  'train_rows_sha256': arr_digest(rows), 'validation_rows_sha256': arr_digest(val_rows),
                  'train_images': len(rows), 'validation_images': len(val_rows), 'test_images': len(test_rows),
                  'alpha_grid_relative_to_mean_train_kernel_diagonal': alphas,
                  'alpha_selection': 'minimum mean of validation branch MSE divided by training-only branch variance',
                  'one_common_alpha': True, 'test_used_for_selection': False,
                  'upstream_validation_note': 'BrainVAE was previously selected using this internal split',
                  'linear_algebra': 'cuda fp32 kernel blocks, fp64 accumulation, TF32 disabled; fp64 fallback and residual checks',
                  'target_chunk': args.target_chunk, 'kernel_chunk': args.kernel_chunk,
                  'encode_batch_size': args.encode_batch_size,
                  'debug': args.smoke, 'torch_version': str(torch.__version__),
                  'gpu_name': torch.cuda.get_device_name(device), 'code_sha256': file_sha256(Path(__file__))}
        atomic_json(config, run / 'protocol.json')
        print('[prepare] training-only target means and variances', flush=True)
        means, scales = three.training_statistics(targets, rows)
        np.savez(model_dir / 'target_stats.npz', **{'mean_' + k: means[k] for k in BRANCHES},
                 **{'var_' + k: np.asarray(scales[k], dtype=np.float64) for k in BRANCHES})
        print('[prepare] variances {}'.format(json.dumps(scales, sort_keys=True)), flush=True)
        atomic_json({'status': 'running', 'stage': 'brain_mu_train_validation'}, run / 'state.json')
        brain = frozen_brain(source, device)
        del source['weights']
        cache_rows = np.concatenate((rows, val_rows))
        cache, lookup = cache_subject_means(brain, source, paths, cache_rows, args.encode_batch_size, device, args.subject)
        if cache is None or not np.array_equal(lookup[cache_rows], np.arange(len(cache_rows))):
            raise RuntimeError('Failed or misaligned BrainVAE mean cache')
        print('[kernel] transferring {} training μ vectors to GPU'.format(len(rows)), flush=True)
        x = copy_rows_to_gpu(cache, 0, len(rows), device, args.transfer_batch)
        mean_x = x.mean(dim=0)
        np.save(model_dir / 'input_mean.npy', mean_x.cpu().numpy(), allow_pickle=False)
        x.sub_(mean_x).mul_(INPUT_DIM ** -0.5)
        xv = copy_rows_to_gpu(cache, len(rows), len(val_rows), device, args.transfer_batch)
        xv.sub_(mean_x).mul_(INPUT_DIM ** -0.5)
        del cache, lookup
        gc.collect()
        atomic_json({'status': 'running', 'stage': 'kernel_and_validation'}, run / 'state.json')
        started = time.monotonic()
        gram, cross_val = block_kernels(x, xv, args.kernel_chunk)
        torch.cuda.synchronize(device)
        if not torch.isfinite(gram).all() or not torch.isfinite(cross_val).all():
            raise FloatingPointError('Nonfinite training or validation kernel')
        print('[kernel] train={} validation={} elapsed={:.1f}s'.format(
            list(gram.shape), list(cross_val.shape), time.monotonic() - started), flush=True)
        print('[kernel] mean_diag={:.8g} min_diag={:.8g} max_diag={:.8g}'.format(
            float(gram.diagonal().mean()), float(gram.diagonal().min()), float(gram.diagonal().max())), flush=True)
        try:
            kernel_check = check_kernel_reference(x, gram, xv, cross_val)
            kernel_check['method'] = 'fp32_blocks_fp64_accumulation'
        except FloatingPointError as error:
            print('[kernel] {}; recomputing with full FP64 block products'.format(error), flush=True)
            del gram, cross_val
            torch.cuda.empty_cache()
            gram, cross_val = block_kernels(x, xv, args.kernel_chunk, use_double=True)
            kernel_check = check_kernel_reference(x, gram, xv, cross_val)
            kernel_check['method'] = 'fp64_blocks_fp64_accumulation'
        del xv
        print('[kernel] float64_sample_check {}'.format(json.dumps(kernel_check, sort_keys=True)), flush=True)
        projections, kernel_scale, accepted_alphas, solve_diagnostics = solve_projections(gram, cross_val, alphas)
        branch_scores, normalized_scores = {}, {}
        for name in BRANCHES:
            print('[sweep] branch {}'.format(name), flush=True)
            ytrain, yval = feature_matrix(targets, name, rows), feature_matrix(targets, name, val_rows)
            raw, normalized = selection_scores(projections, ytrain, yval, means[name], scales[name],
                                                args.target_chunk, device)
            branch_scores[name], normalized_scores[name] = raw.tolist(), normalized.tolist()
            del ytrain, yval
            gc.collect()
        candidates = []
        for i, alpha in enumerate(accepted_alphas):
            candidates.append({'relative_alpha': alpha, 'absolute_alpha': alpha * kernel_scale,
                               'validation_normalized_mse': sum(normalized_scores[k][i] for k in BRANCHES) / len(BRANCHES),
                               'branch_mse': {k: branch_scores[k][i] for k in BRANCHES}})
        best = min(candidates, key=lambda item: item['validation_normalized_mse'])
        selection = {'selection_split': 'internal_validation', 'test_used_for_selection': False,
                     'train_only_target_variances': scales, 'kernel_mean_diagonal': kernel_scale,
                     'best': best, 'candidates': candidates, 'numerical_diagnostics': solve_diagnostics,
                     'kernel_float64_sample_check': kernel_check}
        atomic_json(selection, run / 'validation_selection.json')
        print('[selected] {}'.format(json.dumps(best, sort_keys=True)), flush=True)
        del projections
        system = gram.clone()
        system.diagonal().add_(best['absolute_alpha'])
        factor = torch.linalg.cholesky(system)
        del system, gram
        atomic_json({'status': 'running', 'stage': 'test_fmri_and_fit_coefficients'}, run / 'state.json')
        # No test text labels have been opened before this point.
        cross_test = test_kernel(brain, source, paths, test_rows, x, mean_x, args.encode_batch_size, device,
                                 args.kernel_chunk, use_double=(kernel_check['method'] == 'fp64_blocks_fp64_accumulation'),
                                 subject=args.subject)
        del brain, x, mean_x
        torch.cuda.empty_cache()
        np.save(model_dir / 'train_rows.npy', rows, allow_pickle=False)
        np.save(model_dir / 'train_image_ids.npy', split['source_image_ids'][rows], allow_pickle=False)
        np.save(val_dir / 'image_ids.npy', split['source_image_ids'][val_rows], allow_pickle=False)
        np.save(test_dir / 'image_ids.npy', split['test_image_ids'][test_rows], allow_pickle=False)
        hashes = {}
        for name in BRANCHES:
            hashes[name] = save_branch(name, targets, rows, val_rows, means, factor, cross_val, cross_test,
                                       args.target_chunk, model_dir, val_dir, test_dir, device)
        atomic_json({'status': 'complete', 'images': len(test_rows), 'test_text_used_as_input': False,
                     'head': 'ridge', 'predictions_sha256': {k: hashes[k]['test_sha256'] for k in BRANCHES}},
                    test_dir / 'prediction_complete.json')
        atomic_json({'status': 'complete', 'representation': 'dual_ridge_coefficients',
                     'requires': 're-encode train_image_ids with the recorded frozen BrainVAE and source fMRI',
                     'input_mean_sha256': file_sha256(model_dir / 'input_mean.npy'),
                     'target_stats_sha256': file_sha256(model_dir / 'target_stats.npz'),
                     'train_rows_sha256': file_sha256(model_dir / 'train_rows.npy'),
                     'train_ids_sha256': file_sha256(model_dir / 'train_image_ids.npy'),
                     'dual_sha256': {k: hashes[k]['dual_sha256'] for k in BRANCHES},
                     'alpha': best, 'brain_source': config['brain_source'], 'kernel_chunk': args.kernel_chunk,
                     'kernel_method': kernel_check['method'], 'debug': args.smoke},
                    model_dir / 'model_complete.json')
        del factor, cross_val, cross_test
        torch.cuda.empty_cache()
        atomic_json({'status': 'running', 'stage': 'feature_evaluation'}, run / 'state.json')
        val = evaluate_predictions(val_dir, targets, val_rows, means, scales, 32, device)
        atomic_json(val, val_dir / 'text_feature_metrics.json')
        if not args.smoke:
            test_targets = load_subject_targets(target_dir, 'test', split['test_image_ids'], args.subject)
            test = evaluate_predictions(test_dir, test_targets, test_rows, means, scales, 32, device)
            test.update(test_used_for_selection=False, test_text_used_as_prediction_input=False)
            atomic_json(test, test_dir / 'text_feature_metrics.json')
        atomic_json({'status': 'complete', 'stage': 'complete', 'selected_relative_alpha': best['relative_alpha'],
                     'test_evaluated': not args.smoke}, run / 'state.json')
        print('[complete] {}'.format(run), flush=True)
    return 0


if __name__ == '__main__':
    def signal_stop(signum, _frame):
        global INTERRUPTED_SIGNAL
        INTERRUPTED_SIGNAL = signum
        raise KeyboardInterrupt('received signal {}'.format(signum))

    signal.signal(signal.SIGTERM, signal_stop)
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError, KeyboardInterrupt) as error:
        if RUN_STATE_FILE is not None and RUN_STATE_FILE.parent.is_dir():
            previous = {}
            if RUN_STATE_FILE.is_file():
                previous = json.loads(RUN_STATE_FILE.read_text(encoding='utf-8'))
            atomic_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         'stage': previous.get('stage', 'startup'), 'error': str(error)}, RUN_STATE_FILE)
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(128 + INTERRUPTED_SIGNAL if INTERRUPTED_SIGNAL else 1)
