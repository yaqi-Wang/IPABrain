#!/usr/bin/env python3
"""Extract SDXL CLIP-L hidden_states[11] for NSD images and COCO captions.

Default: subject 01, all available captions, train and test, GPU FP16.
The result is per-caption [N,77,768]; image_ids and caption_offsets make the
mapping to each fMRI image explicit. No caption averaging or token projection.

Examples, from imagery_probabilistic_alignment/:
  python output_1104/scripts/extract_sdxl_clip_l11.py --dry-run
  python output_1104/scripts/extract_sdxl_clip_l11.py --benchmark --max-images 32
  python -u output_1104/scripts/extract_sdxl_clip_l11.py --batch-size 256
  python -u output_1104/scripts/extract_sdxl_clip_l11.py --caption-policy first
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast


OUTPUT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = OUTPUT_ROOT.parent
NSD_ROOT = PROJECT_ROOT / 'data/raw_nsd'
CLIP_DIR = PROJECT_ROOT / 'huggingface/clip-vit-large'
TOKEN_SHAPE = (77, 768)
TARGET = 'sdxl_clip_l_hidden_states_11_pre_final_layernorm_unprojected'
STOP_SIGNAL = 0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(str(temp), str(path))


def save_array_once(path, values):
    path = Path(path)
    values = np.asarray(values)
    if path.exists():
        old = np.load(path, mmap_mode='r', allow_pickle=False)
        if old.shape != values.shape or old.dtype != values.dtype or not np.array_equal(old, values):
            raise ValueError('Existing alignment array differs: {}'.format(path))
        return
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(str(temp), str(path))


@contextlib.contextmanager
def extraction_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.extract.lock').open('a', encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another extraction is using {}'.format(directory)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sources(nsd_root):
    ann = nsd_root / 'nsddata_stimuli/stimuli/nsd/annotations'
    return {
        'mapping': nsd_root / 'nsddata/experiments/nsd/nsd_stim_info_merged.csv',
        'train2017': ann / 'captions_train2017.json',
        'val2017': ann / 'captions_val2017.json',
    }


def read_ids(subject, splits, max_images):
    folder = PROJECT_ROOT / 'data/processed/nsd' / 'subj{:02d}'.format(subject)
    ids, paths = {}, {}
    for split in splits:
        path = folder / 'nsd_{}_image_id_sub{}.npy'.format(split, subject)
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1 or array.dtype.kind not in 'iu' or not len(array):
            raise ValueError('Invalid image IDs in {}'.format(path))
        array = array.astype(np.int64, copy=False)
        if np.any(array < 0) or len(np.unique(array)) != len(array):
            raise ValueError('Negative or repeated image IDs in {}'.format(path))
        paths[split] = path
        ids[split] = array[:max_images] if max_images else array
    if len(ids) == 2 and np.intersect1d(ids['train'], ids['test']).size:
        raise ValueError('Train/test image IDs overlap')
    return ids, paths


def read_captions(source_paths, requested_ids, policy):
    requested = set(map(int, requested_ids))
    mapping = {}
    with source_paths['mapping'].open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            image_id = int(row['nsdId'])
            if image_id in requested:
                if image_id in mapping:
                    raise ValueError('Duplicate NSD mapping for {}'.format(image_id))
                mapping[image_id] = (row['cocoSplit'], int(row['cocoId']))
    if set(mapping) != requested:
        raise ValueError('Missing NSD-to-COCO mapping: {}'.format(sorted(requested - set(mapping))[:10]))
    grouped = {}
    for split in ('train2017', 'val2017'):
        wanted = {coco for actual, coco in mapping.values() if actual == split}
        by_image = collections.defaultdict(list)
        if wanted:
            with source_paths[split].open(encoding='utf-8') as handle:
                annotations = json.load(handle)['annotations']
            for record in annotations:
                coco_id = int(record['image_id'])
                text = str(record['caption']).strip()
                if coco_id in wanted and text:
                    by_image[coco_id].append({'annotation_id': int(record['id']), 'text': text})
        for image_id, (actual, coco_id) in mapping.items():
            if actual != split:
                continue
            values = by_image[coco_id]
            if not values:
                raise ValueError('No nonempty COCO caption for NSD image {}'.format(image_id))
            if policy == 'first':
                values = values[:1]
            grouped[image_id] = [dict(item, image_id=image_id, coco_id=coco_id, coco_split=split,
                                      caption_index=i) for i, item in enumerate(values)]
    if set(grouped) != requested:
        raise ValueError('Missing image captions or unsupported COCO split')
    return grouped


def build_records(image_ids, grouped):
    records, offsets = [], [0]
    for image_id in image_ids:
        records.extend(grouped[int(image_id)])
        offsets.append(len(records))
    return records, np.asarray(offsets, dtype=np.int64)


def caption_manifest_sha256(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    return digest.hexdigest()


def write_alignment(directory, split, image_ids, records, offsets):
    save_array_once(directory / '{}_image_ids.npy'.format(split), image_ids)
    save_array_once(directory / '{}_caption_offsets.npy'.format(split), offsets)
    save_array_once(directory / '{}_caption_image_ids.npy'.format(split),
                    np.asarray([r['image_id'] for r in records], dtype=np.int64))
    save_array_once(directory / '{}_caption_annotation_ids.npy'.format(split),
                    np.asarray([r['annotation_id'] for r in records], dtype=np.int64))
    path = directory / '{}_captions.jsonl'.format(split)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    if path.exists():
        if sha256_file(path) != sha256_file(temp):
            temp.unlink()
            raise ValueError('Existing caption manifest differs: {}'.format(path))
        temp.unlink()
    else:
        os.replace(str(temp), str(path))


def tokenize_all(texts, tokenizer, fast_tokenizer, batch_size=2048):
    """Use fast tokens only after checking every ID against the SDXL slow tokenizer."""
    chunks, fallback_rows = [], 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        kwargs = dict(truncation=True, max_length=77, padding='max_length', return_tensors='np')
        fast = fast_tokenizer(batch, **kwargs)['input_ids']
        slow = tokenizer(batch, **kwargs)['input_ids']
        if fast.shape != (len(batch), 77) or slow.shape != fast.shape:
            raise ValueError('Unexpected CLIP tokenizer output shape')
        different = np.any(fast != slow, axis=1)
        if np.any(different):
            fallback_rows += int(different.sum())
            fast[different] = slow[different]
        chunks.append(fast.astype(np.int64, copy=False))
    if not chunks:
        raise ValueError('No captions to tokenize')
    return np.concatenate(chunks), fallback_rows


class FastClipL11(torch.nn.Module):
    """The first 11 CLIP text blocks, using fused QKV and causal PyTorch SDPA."""

    def __init__(self, model):
        super().__init__()
        text = model.text_model
        if text.config.hidden_size != 768 or len(text.encoder.layers) != 12:
            raise ValueError('Expected the 12-layer, 768-wide CLIP-L text encoder')
        self.embeddings = text.embeddings
        self.layers = torch.nn.ModuleList(list(text.encoder.layers[:11]))
        self.qkv_weight = torch.nn.ParameterList()
        self.qkv_bias = torch.nn.ParameterList()
        for layer in self.layers:
            attn = layer.self_attn
            if attn.embed_dim != 768 or attn.num_heads != 12 or attn.head_dim != 64:
                raise ValueError('Unexpected CLIP-L attention dimensions')
            self.qkv_weight.append(torch.nn.Parameter(torch.cat(
                (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0), requires_grad=False))
            self.qkv_bias.append(torch.nn.Parameter(torch.cat(
                (attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias), dim=0), requires_grad=False))
        self.eval().requires_grad_(False)

    def forward(self, input_ids):
        hidden = self.embeddings(input_ids=input_ids)
        batch, length, width = hidden.shape
        if length != 77 or width != 768:
            raise ValueError('Expected [batch,77,768] embeddings')
        for layer, weight, bias in zip(self.layers, self.qkv_weight, self.qkv_bias):
            residual = hidden
            normed = layer.layer_norm1(hidden)
            q, k, v = F.linear(normed, weight, bias).chunk(3, dim=-1)
            q = q.reshape(batch, length, 12, 64).transpose(1, 2)
            k = k.reshape(batch, length, 12, 64).transpose(1, 2)
            v = v.reshape(batch, length, 12, 64).transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            attended = attended.transpose(1, 2).reshape(batch, length, width)
            hidden = residual + layer.self_attn.out_proj(attended)
            hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
        return hidden


def load_model(model_dir, device, precision):
    dtype = {'float32': torch.float32, 'float16': torch.float16,
             'bfloat16': torch.bfloat16}[precision]
    model = CLIPTextModel.from_pretrained(str(model_dir), local_files_only=True,
                                         use_safetensors=True, torch_dtype=dtype,
                                         attn_implementation='sdpa')
    model = model.to(device=device).eval().requires_grad_(False)
    return model, FastClipL11(model).to(device=device).eval()


def verify_fast(model, fast, ids, precision):
    """Compare optimized path with the same checkpoint's HF hidden_states[11]."""
    ids = ids[:min(4, len(ids))]
    with torch.inference_mode():
        reference = model(input_ids=ids, output_hidden_states=True, return_dict=True).hidden_states[11].float()
        actual = fast(ids).float()
    if tuple(actual.shape) != (len(ids), *TOKEN_SHAPE) or not torch.isfinite(actual).all():
        raise FloatingPointError('Invalid optimized CLIP-L output')
    diff = actual - reference
    max_abs = float(diff.abs().max().item())
    relative_rmse = float((diff.square().mean().sqrt() /
                           reference.square().mean().sqrt().clamp_min(1e-12)).item())
    limits = {'float32': (0.001, 0.0001), 'float16': (0.1, 0.005),
              'bfloat16': (0.5, 0.025)}
    max_limit, rmse_limit = limits[precision]
    if max_abs > max_limit or relative_rmse > rmse_limit:
        raise FloatingPointError('Optimized CLIP-L differs from HF hidden_states[11]: '
                                 'max_abs={}, relative_rmse={}'.format(max_abs, relative_rmse))
    result = {'samples': len(ids), 'max_abs': max_abs, 'relative_rmse': relative_rmse,
              'max_abs_limit': max_limit, 'relative_rmse_limit': rmse_limit,
              'reference': 'CLIPTextModel(input_ids=...,output_hidden_states=True).hidden_states[11]',
              'tokenizer_attention_mask_passed': False}
    print('[verification] {}'.format(json.dumps(result, sort_keys=True)), flush=True)
    return result


def make_identity(args, ids_paths, source_paths, device, counts, tokenization):
    files = {'model_weights': args.clip_dir / 'model.safetensors',
             'model_config': args.clip_dir / 'config.json',
             'tokenizer_vocab': args.clip_dir / 'vocab.json',
             'tokenizer_merges': args.clip_dir / 'merges.txt',
             'tokenizer_config': args.clip_dir / 'tokenizer_config.json',
             'tokenizer_json': args.clip_dir / 'tokenizer.json',
             'special_tokens': args.clip_dir / 'special_tokens_map.json',
             'extractor': Path(__file__).resolve()}
    files.update({'caption_' + name: path for name, path in source_paths.items()})
    files.update({split + '_ids': path for split, path in ids_paths.items()})
    info = {}
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        print('[hash] {}'.format(key), flush=True)
        info[key] = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
    return {
        'schema_version': 1, 'subject': args.subject, 'splits': list(ids_paths),
        'split_counts': counts, 'representation': TARGET,
        'model_version': 'local OpenAI CLIP ViT-L/14 text tower',
        'caption_policy': args.caption_policy,
        'max_images_per_split': args.max_images,
        'caption_order': 'original nonempty COCO annotations, preserving source order per image',
        'tokenizer': 'CLIPTokenizerFast compared against CLIPTokenizer for every caption; slow IDs used on mismatch',
        'tokenization_by_split': tokenization,
        'max_length': 77, 'hidden_state_index': 11,
        'tokenizer_attention_mask_passed': False,
        'normalization': 'none; no final layer norm, text projection, pooled normalization, or caption averaging',
        'calculation_precision': args.precision, 'storage_dtype': args.output_dtype,
        'batch_size': args.batch_size, 'attention_implementation': 'fused_qkv_causal_pytorch_sdpa',
        'torch_version': str(torch.__version__), 'transformers_version': str(transformers.__version__),
        'numpy_version': str(np.__version__), 'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'allow_tf32': False, 'sources': info,
        'test_usage': 'held-out ground truth targets for evaluation/oracle only; never use as predicted text',
        'checkpoint_note': 'CLIP-L is loaded from the named local model; equivalence to a future SDXL checkpoint requires verifying that checkpoint\'s text-encoder weights.',
    }


class FeatureArray:
    def __init__(self, path, shape, dtype, identity):
        self.path = Path(path)
        self.partial = self.path.with_name(self.path.name + '.partial')
        self.progress = self.path.with_name(self.path.name + '.progress.json')
        self.meta = self.path.with_suffix('.meta.json')
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.identity = identity
        self.array = None
        self.next_index = 0
        self.completed = False
        if self.meta.exists():
            info = json.loads(self.meta.read_text(encoding='utf-8'))
            if info.get('identity_sha256') != identity or not self.path.is_file():
                raise ValueError('Existing completed feature identity mismatch: {}'.format(self.path))
            array = np.load(self.path, mmap_mode='r', allow_pickle=False)
            if array.shape != self.shape or array.dtype != self.dtype or sha256_file(self.path) != info['features_sha256']:
                raise ValueError('Completed feature shape, dtype, or hash mismatch: {}'.format(self.path))
            self.completed = True
            self.next_index = self.shape[0]
            return
        if self.path.exists() or self.partial.exists() or self.progress.exists():
            if not self.progress.is_file():
                raise ValueError('Partial features lack a progress file: {}'.format(self.path))
            state = json.loads(self.progress.read_text(encoding='utf-8'))
            if (state.get('identity_sha256') != identity or state.get('shape') != list(self.shape)
                    or state.get('dtype') != str(self.dtype)):
                raise ValueError('Partial feature identity mismatch: {}'.format(self.path))
            self.next_index = int(state['next_index'])
            if not 0 <= self.next_index <= self.shape[0]:
                raise ValueError('Invalid progress in {}'.format(self.progress))
            if self.path.exists() and (self.partial.exists() or self.next_index != self.shape[0]):
                raise ValueError('Ambiguous final/partial feature files: {}'.format(self.path))
            working = self.path if self.path.exists() else self.partial
            if not working.is_file():
                raise ValueError('Progress exists without its feature array: {}'.format(working))
            self.array = np.load(working, mmap_mode='r+', allow_pickle=False)
            if self.array.shape != self.shape or self.array.dtype != self.dtype:
                raise ValueError('Partial feature shape or dtype mismatch: {}'.format(working))
        else:
            self.array = np.lib.format.open_memmap(str(self.partial), mode='w+',
                                                   dtype=self.dtype, shape=self.shape)
            self.checkpoint(0)

    def checkpoint(self, index):
        self.array.flush()
        self.next_index = int(index)
        atomic_json(self.progress, {'identity_sha256': self.identity, 'shape': list(self.shape),
                                    'dtype': str(self.dtype), 'next_index': self.next_index})

    def finish(self, metadata):
        if self.completed:
            return
        if self.next_index != self.shape[0]:
            raise ValueError('Cannot finish an incomplete feature array')
        self.array.flush()
        del self.array
        self.array = None
        if self.partial.exists():
            os.replace(str(self.partial), str(self.path))
        info = dict(metadata, identity_sha256=self.identity, shape=list(self.shape), dtype=str(self.dtype),
                    features_path=str(self.path), features_sha256=sha256_file(self.path))
        atomic_json(self.meta, info)
        self.progress.unlink()
        self.completed = True


def ensure_disk_space(directory, lengths, dtype):
    needed = 0
    itemsize = np.dtype(dtype).itemsize
    for split, count in lengths.items():
        path = directory / '{}_clip_l11_tokens.npy'.format(split)
        if path.is_file() and path.with_suffix('.meta.json').is_file():
            continue
        expected = count * 77 * 768 * itemsize + 4096
        partial = path.with_name(path.name + '.partial')
        working = path if path.is_file() else partial
        allocated = working.stat().st_blocks * 512 if working.is_file() else 0
        needed += max(0, expected - allocated)
    free = shutil.disk_usage(str(directory)).free
    reserve = 1024 ** 3
    if free < needed + reserve:
        raise RuntimeError('Insufficient disk space: need {:.2f} GiB plus 1 GiB reserve; '
                           'available {:.2f} GiB'.format(needed / 1024**3, free / 1024**3))
    print('[disk] remaining allocation {:.2f} GiB; free {:.2f} GiB'.format(
        needed / 1024**3, free / 1024**3), flush=True)


def extract_tokens(fast, token_ids, output, args, device):
    if output.completed:
        print('[verified] {}'.format(output.path), flush=True)
        return True
    start_index = output.next_index
    position, committed, batches = start_index, start_index, 0
    started = time.monotonic()
    cpu_ids = torch.from_numpy(token_ids)
    if device.type == 'cuda':
        cpu_ids = cpu_ids.pin_memory()
    pending = collections.deque()
    out_torch_dtype = torch.float16 if args.output_dtype == 'float16' else torch.float32

    def write_batch(start, end, values):
        if values.shape != (end - start, *TOKEN_SHAPE) or not np.isfinite(values).all():
            raise FloatingPointError('Invalid or nonfinite CLIP-L token features')
        output.array[start:end] = values
        return end

    def wait_all():
        nonlocal committed
        while pending:
            end, future = pending.popleft()
            if future.result() != end:
                raise RuntimeError('Feature writer returned an incorrect offset')
            committed = end

    with ThreadPoolExecutor(max_workers=1) as writer:
        with torch.inference_mode():
            while position < len(token_ids):
                end = min(position + args.batch_size, len(token_ids))
                batch = cpu_ids[position:end].to(device=device, non_blocking=(device.type == 'cuda'))
                result = fast(batch).to(dtype=out_torch_dtype).cpu().numpy()
                pending.append((end, writer.submit(write_batch, position, end, result)))
                position = end
                batches += 1
                if len(pending) >= args.writer_prefetch:
                    done, future = pending.popleft()
                    if future.result() != done:
                        raise RuntimeError('Feature writer returned an incorrect offset')
                    committed = done
                should_stop = bool(STOP_SIGNAL) or (args.stop_after_batches and batches >= args.stop_after_batches)
                if batches % args.flush_batches == 0 or position == len(token_ids) or should_stop:
                    wait_all()
                    output.checkpoint(committed)
                    rate = (committed - start_index) / max(time.monotonic() - started, 1e-9)
                    print('[extract] {} {}/{} {:.1f} captions/s'.format(
                        output.path.name, committed, len(token_ids), rate), flush=True)
                if should_stop:
                    return False
        wait_all()
    if output.next_index != len(token_ids):
        output.checkpoint(committed)
    return True


def benchmark(fast, ids, args, device):
    if device.type != 'cuda':
        raise ValueError('--benchmark requires CUDA')
    values = []
    out_dtype = torch.float16 if args.output_dtype == 'float16' else torch.float32
    print('[benchmark] repeated real caption tokens; includes GPU inference and transfer to CPU, not disk writes', flush=True)
    with torch.inference_mode():
        for batch_size in args.benchmark_batches:
            if batch_size < 1:
                raise ValueError('Benchmark batch sizes must be positive')
            batch = repeated = None
            try:
                repeated = ids.repeat(((batch_size + len(ids) - 1) // len(ids), 1))[:batch_size]
                batch = repeated.to(device=device, non_blocking=False)
                for _ in range(args.benchmark_warmup):
                    fast(batch).to(dtype=out_dtype).cpu()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for _ in range(args.benchmark_steps):
                    fast(batch).to(dtype=out_dtype).cpu()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                row = {'batch_size': batch_size,
                       'captions_per_second': batch_size * args.benchmark_steps / elapsed,
                       'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3}
                values.append(row)
                print('[benchmark] {}'.format(json.dumps(row, sort_keys=True)), flush=True)
            except torch.cuda.OutOfMemoryError:
                print('[benchmark] batch {}: CUDA out of memory, skipped'.format(batch_size), flush=True)
                del batch, repeated
                torch.cuda.empty_cache()
                break
    if not values:
        raise RuntimeError('No benchmark batch size completed')
    winner = max(values, key=lambda row: row['captions_per_second'])
    print('[benchmark] recommended --batch-size {} ({:.1f} captions/s; excluding disk writes)'.format(
        winner['batch_size'], winner['captions_per_second']), flush=True)
    return values


def on_signal(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum
    print('[signal] received {}; saving the current completed batch before stopping'.format(signum), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subject', type=int, default=1, choices=range(1, 9))
    parser.add_argument('--split', choices=('train', 'test', 'both'), default='both')
    parser.add_argument('--caption-policy', choices=('all', 'first'), default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('float16', 'float32', 'bfloat16'), default='float16')
    parser.add_argument('--output-dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--batch-size', type=int, default=256, help='captions per GPU batch')
    parser.add_argument('--flush-batches', type=int, default=8)
    parser.add_argument('--writer-prefetch', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=0, help='debug prefix per split; separate output directory')
    parser.add_argument('--nsd-root', type=Path, default=NSD_ROOT)
    parser.add_argument('--clip-dir', type=Path, default=CLIP_DIR)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true', help='check IDs, captions and storage without GPU or writes')
    parser.add_argument('--benchmark', action='store_true', help='benchmark GPU batch sizes without writing outputs')
    parser.add_argument('--benchmark-batches', nargs='+', type=int, default=[64, 128, 256, 512, 768])
    parser.add_argument('--benchmark-warmup', type=int, default=2)
    parser.add_argument('--benchmark-steps', type=int, default=5)
    parser.add_argument('--stop-after-batches', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.flush_batches, args.writer_prefetch, args.threads,
           args.benchmark_warmup, args.benchmark_steps) < 1 or min(args.max_images, args.stop_after_batches) < 0:
        parser.error('Batch sizes, thread counts and benchmark steps must be positive')
    if args.dry_run and args.benchmark:
        parser.error('--dry-run and --benchmark are mutually exclusive')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    splits = ('train', 'test') if args.split == 'both' else (args.split,)
    ids, ids_paths = read_ids(args.subject, splits, args.max_images)
    source_paths = sources(args.nsd_root)
    grouped = read_captions(source_paths, np.concatenate(list(ids.values())), args.caption_policy)
    records, offsets = {}, {}
    for split in splits:
        records[split], offsets[split] = build_records(ids[split], grouped)
        counts = np.diff(offsets[split])
        print('[alignment] {}: {} images, {} captions, per-image min/max {}/{}'.format(
            split, len(ids[split]), len(records[split]), int(counts.min()), int(counts.max())), flush=True)
    lengths = {split: len(records[split]) for split in splits}
    expected = sum(lengths.values()) * 77 * 768 * np.dtype(args.output_dtype).itemsize
    print('[estimate] token arrays {:.2f} GiB; per-caption shape [N,77,768] {}'.format(
        expected / 1024**3, args.output_dtype), flush=True)
    if args.dry_run:
        print('[dry-run] no model inference and no output files written', flush=True)
        return 0
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on the GPU host with the StableDiffusionReconstruction environment')
    if device.type == 'cpu' and args.precision != 'float32':
        raise ValueError('CPU validation requires --precision float32')
    if device.type == 'cuda' and args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16')

    tokenizer = CLIPTokenizer.from_pretrained(str(args.clip_dir), local_files_only=True)
    fast_tokenizer = CLIPTokenizerFast.from_pretrained(str(args.clip_dir), local_files_only=True)
    tokens, tokenization = {}, {}
    for split in splits:
        started = time.monotonic()
        tokens[split], fallback = tokenize_all([r['text'] for r in records[split]], tokenizer, fast_tokenizer)
        tokenization[split] = {
            'caption_manifest_sha256': caption_manifest_sha256(records[split]),
            'input_ids_sha256': hashlib.sha256(tokens[split].tobytes(order='C')).hexdigest(),
            'input_ids_dtype': str(tokens[split].dtype),
            'input_ids_byte_order': sys.byteorder,
            'slow_tokenizer_fallback_rows': fallback,
        }
        print('[tokenizer] {}: {} captions in {:.2f}s; slow-tokenizer fallback rows {}'.format(
            split, len(tokens[split]), time.monotonic() - started, fallback), flush=True)
    if args.benchmark:
        model, fast = load_model(args.clip_dir, device, args.precision)
        sample_ids = torch.from_numpy(tokens[splits[0]][:4]).to(device)
        verify_fast(model, fast, sample_ids, args.precision)
        benchmark(fast, torch.from_numpy(tokens[splits[0]]), args, device)
        return 0

    name = 'sdxl_clip_l11_{}_{}_v1'.format(args.caption_policy, args.split)
    directory = args.output_dir or (OUTPUT_ROOT / 'debug' / '{}_subj{:02d}_first{}'.format(
        name, args.subject, args.max_images) if args.max_images else
        OUTPUT_ROOT / 'nsd_subj{:02d}'.format(args.subject) / name)
    with extraction_lock(directory):
        counts = {split: {'images': len(ids[split]), 'captions': lengths[split]} for split in splits}
        identity_data = make_identity(args, ids_paths, source_paths, device, counts, tokenization)
        identity = digest_json(identity_data)
        identity_path = directory / 'source_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity_data:
                raise ValueError('Existing source identity differs; use original parameters or a new output directory')
        else:
            existing = [p.name for p in directory.iterdir() if p.name != '.extract.lock']
            if existing:
                raise ValueError('Output directory is not empty and has no source identity: {}'.format(directory))
            atomic_json(identity_path, identity_data)
        ensure_disk_space(directory, lengths, args.output_dtype)
        for split in splits:
            write_alignment(directory, split, ids[split], records[split], offsets[split])
        # Load the GPU model only after CPU tokenization, source hashing, disk
        # checks, and alignment writes, so it does not sit idle during setup.
        model, fast = load_model(args.clip_dir, device, args.precision)
        sample_ids = torch.from_numpy(tokens[splits[0]][:4]).to(device)
        verification = verify_fast(model, fast, sample_ids, args.precision)
        verification_path = directory / 'verification.json'
        if not verification_path.exists():
            atomic_json(verification_path, verification)
        empty_tokens, _ = tokenize_all([''], tokenizer, fast_tokenizer)
        with torch.inference_mode():
            empty = fast(torch.from_numpy(empty_tokens).to(device)).to(
                dtype=torch.float16 if args.output_dtype == 'float16' else torch.float32).cpu().numpy()
        if not np.isfinite(empty).all():
            raise FloatingPointError('Invalid empty-prompt CLIP-L tokens')
        save_array_once(directory / 'empty_prompt_clip_l11.npy', empty)
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        for split in splits:
            output = FeatureArray(directory / '{}_clip_l11_tokens.npy'.format(split),
                                  (lengths[split], *TOKEN_SHAPE), args.output_dtype, identity)
            done = extract_tokens(fast, tokens[split], output, args, device)
            if not done:
                print('[paused] saved progress in {}'.format(directory), flush=True)
                return 128 + STOP_SIGNAL if STOP_SIGNAL else 0
            output.finish({'split': split, 'subject': args.subject, 'representation': TARGET,
                           'caption_policy': args.caption_policy,
                           'image_ids_path': str(directory / '{}_image_ids.npy'.format(split)),
                           'caption_offsets_path': str(directory / '{}_caption_offsets.npy'.format(split)),
                           'captions_path': str(directory / '{}_captions.jsonl'.format(split)),
                           'source_identity_path': str(identity_path)})
        print('[complete] {}'.format(directory), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
