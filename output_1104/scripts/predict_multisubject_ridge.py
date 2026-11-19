#!/usr/bin/env python3
"""Apply a fitted dual ridge head to new BrainVAE μ values or NSD test fMRI.

The compact model stores dual coefficients but intentionally does not duplicate
the very large training μ matrix. It re-encodes the recorded training images
with the recorded frozen BrainVAE, then uses the same centered linear kernel.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'output_1104/scripts'))

import numpy as np
import torch

import ridge_multisubject_three_text as ridge
import train_sdxl_clipl_head as common
import train_sdxl_three_text as three
from src.data.scheme1 import file_sha256, validate_tag
from src.utils.clip_mapping import atomic_json, frozen_brain, mapping_lock
from multisubject_resume import archive_incomplete


def checked_model(directory, source, allow_debug):
    info = json.loads((directory / 'model_complete.json').read_text(encoding='utf-8'))
    if (info.get('status') != 'complete' or info.get('representation') != 'dual_ridge_coefficients'
            or info.get('brain_source') != source['identity'] or (info.get('debug') and not allow_debug)):
        raise ValueError('Ridge model is incomplete, debug-only, or from a different BrainVAE')
    for name, key in [('input_mean.npy', 'input_mean_sha256'), ('target_stats.npz', 'target_stats_sha256'),
                      ('train_rows.npy', 'train_rows_sha256'), ('train_image_ids.npy', 'train_ids_sha256')]:
        if file_sha256(directory / name) != info[key]:
            raise ValueError('Ridge model source changed: {}'.format(name))
    rows = np.load(directory / 'train_rows.npy', allow_pickle=False)
    ids = np.load(directory / 'train_image_ids.npy', allow_pickle=False)
    if (rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(source['split']['source_image_ids']))
            or len(np.unique(rows)) != len(rows)
            or not np.array_equal(ids, source['split']['source_image_ids'][rows])):
        raise ValueError('Ridge training image IDs or rows no longer match BrainVAE source')
    for name in ridge.BRANCHES:
        path = directory / ('dual_' + name + '.npy')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if (values.dtype != np.float32 or values.shape != (len(rows), *ridge.SHAPES[name])
                or file_sha256(path) != info['dual_sha256'][name]):
            raise ValueError('Ridge {} coefficients changed or have the wrong shape'.format(name))
    means, _ = three.scales_and_means(directory / 'target_stats.npz')
    mu_mean = np.load(directory / 'input_mean.npy', allow_pickle=False)
    if mu_mean.shape != (ridge.INPUT_DIM,) or not np.isfinite(mu_mean).all():
        raise ValueError('Invalid training μ mean')
    return info, rows, means, mu_mean


def query_kernel(mu, train_gpu, input_mean, batch_size, device, kernel_chunk=16384, use_double=False):
    if mu.ndim != 3 or tuple(mu.shape[1:]) != ridge.INPUT_SHAPE or not len(mu):
        raise ValueError('Input must be a nonempty [N,256,1664] μ array')
    result = torch.empty((len(mu), len(train_gpu)), device=device, dtype=torch.float32)
    for start in range(0, len(mu), batch_size):
        end = min(start + batch_size, len(mu))
        block = torch.from_numpy(np.array(mu[start:end], dtype=np.float32, copy=True)).to(device)
        if not torch.isfinite(block).all():
            raise FloatingPointError('Input μ has nonfinite values')
        block = block.reshape(end - start, ridge.INPUT_DIM).sub_(input_mean).mul_(ridge.INPUT_DIM ** -0.5)
        value = ridge.block_cross(block, train_gpu, kernel_chunk, use_double=use_double, show_progress=False)
        ridge.check_cross_reference(train_gpu, block, value)
        result[start:end] = value
    return result


def apply_coefficients(kernel, directory, means, output, chunk, device):
    hashes = {}
    for name in ridge.BRANCHES:
        dual = np.load(directory / ('dual_' + name + '.npy'), mmap_mode='r', allow_pickle=False)
        dims = int(np.prod(ridge.SHAPES[name]))
        dual = dual.reshape(len(dual), dims)
        output_path = output / ('predicted_' + name + '.npy')
        prediction, partial = ridge.create_partial(output_path, (len(kernel), *ridge.SHAPES[name]))
        flat = prediction.reshape(len(kernel), dims)
        average = means[name].reshape(-1)
        for start in range(0, dims, chunk):
            end = min(start + chunk, dims)
            w = torch.from_numpy(np.array(dual[:, start:end], copy=True, order='C')).to(device)
            offset = torch.from_numpy(np.ascontiguousarray(average[start:end])).to(device)
            value = kernel @ w + offset
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite {} prediction'.format(name))
            flat[:, start:end] = value.cpu().numpy()
        ridge.finalize_array(prediction, partial, output_path)
        hashes[name] = file_sha256(output_path)
        print('[predict] {} {}'.format(name, list(prediction.shape)), flush=True)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, choices=range(2, 9), required=True)
    parser.add_argument('--model-dir', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--nsd-test', action='store_true', help='predict the recorded NSD test fMRI')
    group.add_argument('--mu-file', type=Path, help='.npy with [N,256,1664] BrainVAE means')
    parser.add_argument('--name', default='ridge_inference_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--encode-batch-size', type=int, default=4)
    parser.add_argument('--target-chunk', type=int, default=4096)
    parser.add_argument('--query-batch', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0, help='use only a prefix for debugging')
    parser.add_argument('--allow-debug-model', action='store_true')
    parser.add_argument('--resume', action='store_true', help='archive incomplete known outputs and retry')
    args = parser.parse_args()
    validate_tag(args.name)
    if min(args.encode_batch_size, args.target_chunk, args.query_batch) < 1 or args.limit < 0:
        raise ValueError('Batch/chunk sizes must be positive')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this ridge predictor')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = (ROOT / 'output_1104/multisubject_nsd_imagery_v1/debug' / f'subj{args.subject:02d}' / 'ridge'
            if args.allow_debug_model else ROOT / 'output_1104/multisubject_nsd_imagery_v1' / f'subj{args.subject:02d}' / 'ridge')
    if args.model_dir is None:
        args.model_dir = root / 'ridge_mu_three_text_v1/model'
    output = root / 'predictions' / args.name
    with mapping_lock(output):
        if (output/'prediction_complete.json').exists():
            raise FileExistsError('Completed prediction is immutable: {}'.format(output))
        existing = [path for path in output.iterdir() if path.name not in ('.mapping.lock','incomplete_attempts')]
        if existing:
            if not args.resume:raise FileExistsError('Prediction directory is not empty: {}'.format(output))
            names=['image_ids.npy']
            names += ['predicted_'+k+'.npy'+suffix for k in ridge.BRANCHES for suffix in ('','.partial')]
            archived=archive_incomplete(output,names)
            print('[resume] preserved incomplete prediction files in {}'.format(archived),flush=True)
        source, paths = ridge.load_subject_context(args.subject)
        info, rows, means, mu_mean = checked_model(args.model_dir, source, args.allow_debug_model)
        need = len(rows) * ridge.INPUT_DIM * 4 + 5 * 1024**3
        free, _ = torch.cuda.mem_get_info(device)
        if free < need:
            raise RuntimeError('Ridge inference requires {:.1f} GiB free GPU memory'.format(need / 1024**3))
        brain = frozen_brain(source, device)
        del source['weights']
        cache, lookup = ridge.cache_subject_means(brain, source, paths, rows, args.encode_batch_size, device, args.subject)
        if cache is None or not np.array_equal(lookup[rows], np.arange(len(rows))):
            raise RuntimeError('Unable to reconstruct the training μ basis')
        train = ridge.copy_rows_to_gpu(cache, 0, len(rows), device, 32)
        del cache, lookup
        center = torch.from_numpy(mu_mean).to(device)
        mean_diff = float((train.mean(dim=0) - center).abs().max().item())
        if not math.isfinite(mean_diff) or mean_diff > 1e-3:
            raise ValueError('Recomputed training μ mean differs: max_abs={}'.format(mean_diff))
        train.sub_(center).mul_(ridge.INPUT_DIM ** -0.5)
        kernel_chunk = int(info.get('kernel_chunk', 16384))
        use_double = info.get('kernel_method') == 'fp64_blocks_fp64_accumulation'
        if args.nsd_test:
            rows_test = np.arange(len(source['split']['test_image_ids']), dtype=np.int64)
            if args.limit:
                rows_test = rows_test[:args.limit]
            ids = source['split']['test_image_ids'][rows_test]
            kernel = ridge.test_kernel(brain, source, paths, rows_test, train, center,
                                       args.encode_batch_size, device, kernel_chunk, use_double=use_double,
                                       subject=args.subject)
            np.save(output / 'image_ids.npy', ids, allow_pickle=False)
            input_info = {'kind': 'nsd_test_fmri', 'images': len(ids),
                          'image_ids_sha256': file_sha256(output / 'image_ids.npy')}
        else:
            manifest_path = args.mu_file.parent / 'imagery_mu_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if (manifest.get('status') != 'complete' or manifest.get('subject') != args.subject
                    or manifest.get('mu_sha256') != file_sha256(args.mu_file)
                    or manifest.get('brainvae_sha256') != source['identity']['brainvae_checkpoint_sha256']
                    or manifest.get('data_fingerprint') != source['identity']['data_fingerprint']
                    or manifest.get('ridge_model_metadata_sha256') != file_sha256(args.model_dir / 'model_complete.json')
                    or manifest.get('test_images_or_text_used_as_input') is not False):
                raise ValueError('Imagery μ provenance does not match this subject and Ridge model')
            mu = np.load(args.mu_file, mmap_mode='r', allow_pickle=False)
            if args.limit:
                mu = mu[:args.limit]
            kernel = query_kernel(mu, train, center, args.query_batch, device, kernel_chunk, use_double=use_double)
            input_info = {'kind': 'user_supplied_brainvae_mu', 'images': len(mu),
                          'path': str(args.mu_file.resolve()), 'sha256': file_sha256(args.mu_file),
                          'manifest_sha256': file_sha256(manifest_path)}
        del brain, train, center
        torch.cuda.empty_cache()
        hashes = apply_coefficients(kernel, args.model_dir, means, output, args.target_chunk, device)
        atomic_json({'status': 'complete', 'input': input_info, 'model_dir': str(args.model_dir.resolve()),
                     'model_metadata_sha256': file_sha256(args.model_dir / 'model_complete.json'),
                     'prediction_sha256': hashes, 'test_text_used_as_input': False,
                     'regenerated_train_mu_mean_max_abs_difference': mean_diff}, output / 'prediction_complete.json')
        print('[complete] {}'.format(output), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
