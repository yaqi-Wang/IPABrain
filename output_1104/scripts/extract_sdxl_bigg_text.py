#!/usr/bin/env python3
"""Extract SDXL-style OpenCLIP-bigG text conditions for the same first captions.

Only the text tower is loaded from the local OpenCLIP checkpoint. For each
image the output is its 31-block penultimate token tensor [77,1280] and its
32-block, final-LN, projected EOS pooled vector [1280]. The already-verified
CLIP-L caption manifest defines the exact image/caption pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import open_clip
from open_clip.transformer import TextTransformer

import extract_sdxl_clip_l11 as base

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / 'output_1104'
CLIPL_DIR = OUTPUT_ROOT / 'nsd_subj01/sdxl_clip_l11_all_both_v1'
CHECKPOINT = ROOT / 'huggingface/open_clip_pytorch_model.bin'
STOP_SIGNAL = 0
TOKEN_REPR = 'openclip_bigg_32block_penultimate_before_final_ln'
POOLED_REPR = 'openclip_bigg_final_ln_eos_projected_not_normalized'


def on_signal(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum
    print('[signal] {}: will checkpoint the completed batch'.format(signum), flush=True)


def load_alignment(split, limit):
    """Select offset[image] and verify it against image and COCO annotation IDs."""
    source_path = CLIPL_DIR / 'source_identity.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    clip_meta = json.loads((CLIPL_DIR / '{}_clip_l11_tokens.meta.json'.format(split)).read_text(encoding='utf-8'))
    if (source.get('subject') != 1 or source.get('max_images_per_split') != 0
            or source.get('caption_policy') != 'all'
            or clip_meta.get('identity_sha256') != base.digest_json(source)):
        raise ValueError('The source must be the completed full subj01 CLIP-L caption manifest')
    ids = np.load(CLIPL_DIR / '{}_image_ids.npy'.format(split), allow_pickle=False)
    offsets = np.load(CLIPL_DIR / '{}_caption_offsets.npy'.format(split), allow_pickle=False)
    caption_ids = np.load(CLIPL_DIR / '{}_caption_image_ids.npy'.format(split), allow_pickle=False)
    annotation_ids = np.load(CLIPL_DIR / '{}_caption_annotation_ids.npy'.format(split), allow_pickle=False)
    if (ids.ndim != 1 or offsets.shape != (len(ids) + 1,) or offsets[0] != 0
            or np.any(np.diff(offsets) < 1) or offsets[-1] != len(caption_ids)
            or len(annotation_ids) != len(caption_ids)
            or not np.array_equal(caption_ids, np.repeat(ids, np.diff(offsets)))):
        raise ValueError('Invalid CLIP-L caption offsets or image alignment')
    manifest = CLIPL_DIR / '{}_captions.jsonl'.format(split)
    if base.sha256_file(manifest) != source['tokenization_by_split'][split]['caption_manifest_sha256']:
        raise ValueError('CLIP-L caption manifest SHA-256 mismatch: {}'.format(manifest))
    chosen_count = min(limit, len(ids)) if limit else len(ids)
    selected = offsets[:chosen_count]
    chosen, selected_pos = [], 0
    with manifest.open(encoding='utf-8') as handle:
        for i, line in enumerate(handle):
            if selected_pos >= chosen_count:
                break
            if i != int(selected[selected_pos]):
                continue
            record = json.loads(line)
            if (int(record['image_id']) != int(ids[selected_pos])
                    or int(record['annotation_id']) != int(annotation_ids[i])
                    or int(record['caption_index']) != 0 or not str(record['text']).strip()):
                raise ValueError('Caption manifest and selected IDs disagree at image {}'.format(selected_pos))
            chosen.append(record)
            selected_pos += 1
    if len(chosen) != chosen_count:
        raise ValueError('Caption manifest is shorter than the recorded offsets')
    return {'ids': ids[:chosen_count].astype(np.int64, copy=False), 'records': chosen,
            'annotation_ids': annotation_ids[selected].astype(np.int64, copy=False),
            'source_sha256': base.sha256_file(source_path),
            'manifest_sha256': base.sha256_file(manifest),
            'source_tokens_sha256': clip_meta['features_sha256'],
            'original_image_count': len(ids)}


def load_model(checkpoint, device, precision):
    dtype = {'float16': torch.float16, 'float32': torch.float32,
             'bfloat16': torch.bfloat16}[precision]
    with torch.device('meta'):
        model = TextTransformer(context_length=77, vocab_size=49408, width=1280,
                                heads=20, layers=32, output_dim=1280)
    state = torch.load(str(checkpoint), map_location='cpu', weights_only=True, mmap=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    required = model.state_dict()
    missing = sorted(set(required) - set(state))
    mismatch = [name for name in required if name in state and required[name].shape != state[name].shape]
    if missing or mismatch:
        raise ValueError('OpenCLIP bigG checkpoint missing/incompatible keys: {} {}'.format(missing[:5], mismatch[:5]))
    model.load_state_dict({key: state[key] for key in required}, strict=True, assign=True)
    # attn_mask is a non-persistent buffer, so it must be rebuilt after meta initialization.
    model.attn_mask = model.build_causal_mask()
    del state, required
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def encode(model, tokens):
    hidden = model.token_embedding(tokens) + model.positional_embedding
    batch_first = getattr(model.transformer.resblocks[0].attn, 'batch_first', False)
    if not batch_first:
        hidden = hidden.permute(1, 0, 2)
    penultimate = None
    for index, block in enumerate(model.transformer.resblocks):
        if index == len(model.transformer.resblocks) - 1:
            penultimate = hidden if batch_first else hidden.permute(1, 0, 2)
        hidden = block(hidden, attn_mask=model.attn_mask)
    last = hidden if batch_first else hidden.permute(1, 0, 2)
    eot = tokens.argmax(dim=-1)
    selected = last[torch.arange(len(tokens), device=tokens.device), eot]
    pooled = model.ln_final(selected) @ model.text_projection
    if penultimate is None:
        raise ValueError('Missing penultimate block output')
    return penultimate, pooled


def verify_against_openclip(model, ids, precision):
    """Independently hook the last block and use OpenCLIP's own pooled forward."""
    captured = []
    def hook(_module, inputs):
        captured.append(inputs[0].detach())
    handle = model.transformer.resblocks[-1].register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            reference_pooled = model(ids)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError('Failed to capture OpenCLIP penultimate state')
    batch_first = getattr(model.transformer.resblocks[0].attn, 'batch_first', False)
    reference_tokens = captured[0] if batch_first else captured[0].permute(1, 0, 2)
    with torch.inference_mode():
        tokens, pooled = encode(model, ids)
    stats = {}
    for name, actual, reference in [('tokens', tokens, reference_tokens), ('pooled', pooled, reference_pooled)]:
        delta = actual.float() - reference.float()
        max_abs = float(delta.abs().max().item())
        rel = float((delta.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-12)).item())
        limit = {'float32': (0.002, 0.0002), 'float16': (0.1, 0.005),
                 'bfloat16': (0.5, 0.025)}[precision]
        if not torch.isfinite(actual).all() or max_abs > limit[0] or rel > limit[1]:
            raise FloatingPointError('OpenCLIP {} reference mismatch: {} {}'.format(name, max_abs, rel))
        stats[name] = {'max_abs': max_abs, 'relative_rmse': rel}
    print('[verification] {}'.format(json.dumps(stats, sort_keys=True)), flush=True)
    return stats


class PairedArrays:
    def __init__(self, directory, split, count, identity):
        self.directory, self.split, self.count, self.identity = directory, split, count, identity
        self.paths = {kind: directory / '{}_bigg_{}.npy'.format(split, kind) for kind in ('tokens', 'pooled')}
        self.partial = {kind: path.with_name(path.name + '.partial') for kind, path in self.paths.items()}
        self.progress = directory / '{}_bigg.progress.json'.format(split)
        self.meta = directory / '{}_bigg.meta.json'.format(split)
        self.shapes = {'tokens': (count, 77, 1280), 'pooled': (count, 1280)}
        self.arrays = {}
        self.next_index = 0
        self.completed = False
        if self.meta.is_file():
            info = json.loads(self.meta.read_text(encoding='utf-8'))
            if info.get('identity_sha256') != identity:
                raise ValueError('Completed bigG target identity mismatch')
            for kind, path in self.paths.items():
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if (array.shape != self.shapes[kind] or array.dtype != np.float16
                        or base.sha256_file(path) != info[kind + '_sha256']):
                    raise ValueError('Completed bigG target changed: {}'.format(path))
            self.completed = True
            self.next_index = count
            return
        exists = any(p.exists() for p in list(self.paths.values()) + list(self.partial.values()))
        if exists or self.progress.exists():
            if not self.progress.is_file():
                raise ValueError('Partial bigG arrays have no progress record')
            state = json.loads(self.progress.read_text(encoding='utf-8'))
            if state.get('identity_sha256') != identity or state.get('count') != count:
                raise ValueError('Partial bigG arrays belong to a different run')
            self.next_index = int(state['next_index'])
            if not 0 <= self.next_index <= count:
                raise ValueError('Invalid bigG progress')
            for kind in self.paths:
                if self.paths[kind].exists() and (self.partial[kind].exists() or self.next_index != count):
                    raise ValueError('Ambiguous final/partial bigG output')
                working = self.paths[kind] if self.paths[kind].exists() else self.partial[kind]
                if not working.is_file():
                    raise ValueError('A paired bigG array is missing: {}'.format(working))
                value = np.load(working, mmap_mode='r+', allow_pickle=False)
                if value.shape != self.shapes[kind] or value.dtype != np.float16:
                    raise ValueError('Partial bigG shape or dtype mismatch: {}'.format(working))
                self.arrays[kind] = value
        else:
            for kind in self.paths:
                self.arrays[kind] = np.lib.format.open_memmap(
                    str(self.partial[kind]), mode='w+', dtype=np.float16, shape=self.shapes[kind])
            self.checkpoint(0)

    def checkpoint(self, index):
        for array in self.arrays.values():
            array.flush()
        self.next_index = int(index)
        base.atomic_json(self.progress, {'identity_sha256': self.identity, 'count': self.count,
                                         'next_index': self.next_index})

    def finish(self, metadata):
        if self.completed:
            return
        if self.next_index != self.count:
            raise ValueError('Cannot finalize unfinished paired bigG targets')
        for array in self.arrays.values():
            array.flush()
        self.arrays.clear()
        for kind in self.paths:
            if self.partial[kind].exists():
                os.replace(str(self.partial[kind]), str(self.paths[kind]))
        result = dict(metadata, identity_sha256=self.identity, images=self.count,
                      token_shape=list(self.shapes['tokens']), pooled_shape=list(self.shapes['pooled']),
                      dtype='float16', tokens_sha256=base.sha256_file(self.paths['tokens']),
                      pooled_sha256=base.sha256_file(self.paths['pooled']))
        base.atomic_json(self.meta, result)
        self.progress.unlink()
        self.completed = True


def ensure_space(directory, aligned):
    needed = 0
    for split, data in aligned.items():
        count = len(data['ids'])
        if (directory / '{}_bigg.meta.json'.format(split)).is_file():
            continue
        for kind, bytes_per in [('tokens', 77 * 1280 * 2), ('pooled', 1280 * 2)]:
            final = directory / '{}_bigg_{}.npy'.format(split, kind)
            partial = final.with_name(final.name + '.partial')
            working = final if final.is_file() else partial
            allocated = working.stat().st_blocks * 512 if working.is_file() else 0
            needed += max(0, count * bytes_per + 4096 - allocated)
    free = shutil.disk_usage(str(directory)).free
    if free < needed + 1024**3:
        raise RuntimeError('Need {:.2f} GiB plus 1 GiB reserve, only {:.2f} GiB free'.format(
            needed / 1024**3, free / 1024**3))
    print('[disk] remaining {:.2f} GiB; free {:.2f} GiB'.format(needed / 1024**3, free / 1024**3), flush=True)


def extract(model, token_ids, output, args, device):
    if output.completed:
        print('[verified] {} bigG targets'.format(output.split), flush=True)
        return True
    start, pos, batches = output.next_index, output.next_index, 0
    started = time.monotonic()
    ids_cpu = token_ids.pin_memory() if device.type == 'cuda' else token_ids
    with torch.inference_mode():
        while pos < len(ids_cpu):
            end = min(pos + args.batch_size, len(ids_cpu))
            ids = ids_cpu[pos:end].to(device, non_blocking=(device.type == 'cuda'))
            token, pool = encode(model, ids)
            token, pool = token.to(dtype=torch.float16).cpu().numpy(), pool.to(dtype=torch.float16).cpu().numpy()
            if (token.shape != (end - pos, 77, 1280) or pool.shape != (end - pos, 1280)
                    or not np.isfinite(token).all() or not np.isfinite(pool).all()):
                raise FloatingPointError('Invalid or nonfinite bigG conditions')
            output.arrays['tokens'][pos:end] = token
            output.arrays['pooled'][pos:end] = pool
            pos, batches = end, batches + 1
            stopping = bool(STOP_SIGNAL) or (args.stop_after_batches and batches >= args.stop_after_batches)
            if batches % args.flush_batches == 0 or pos == len(ids_cpu) or stopping:
                output.checkpoint(pos)
                print('[extract] {} {}/{} {:.1f} images/s'.format(
                    output.split, pos, len(ids_cpu), (pos - start) / max(time.monotonic() - started, 1e-9)), flush=True)
            if stopping:
                return False
    if output.next_index != len(ids_cpu):
        output.checkpoint(pos)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('float16', 'float32', 'bfloat16'), default='float16')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--flush-batches', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=0, help='debug prefix per split, written separately')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after-batches', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.flush_batches, args.threads) < 1 or min(args.max_images, args.stop_after_batches) < 0:
        parser.error('Invalid batch size, threads or image limit')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    aligned = {split: load_alignment(split, args.max_images) for split in ('train', 'test')}
    if np.intersect1d(aligned['train']['ids'], aligned['test']['ids']).size:
        raise ValueError('Train and test IDs overlap')
    expected = sum(len(x['ids']) for x in aligned.values()) * (77 * 1280 + 1280) * 2
    for split, values in aligned.items():
        print('[alignment] {}: {} images, one CLIP-L-aligned caption per image'.format(split, len(values['ids'])), flush=True)
    print('[estimate] paired FP16 arrays {:.2f} GiB'.format(expected / 1024**3), flush=True)
    if args.dry_run:
        print('[dry-run] no model inference or output writes', flush=True)
        return 0
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on the GPU host')
    if device.type == 'cpu' and args.precision != 'float32':
        raise ValueError('CPU requires --precision float32')
    if device.type == 'cuda' and args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('GPU does not support bfloat16')
    text_ids = {split: open_clip.tokenize([r['text'] for r in values['records']], context_length=77)
                for split, values in aligned.items()}
    directory = args.output_dir or (OUTPUT_ROOT / 'debug' / 'sdxl_bigg_first_first{}'.format(args.max_images)
                                    if args.max_images else OUTPUT_ROOT / 'nsd_subj01/sdxl_bigg_text_first_both_v1')
    with base.extraction_lock(directory):
        from open_clip.tokenizer import default_bpe
        files = {'checkpoint': args.checkpoint, 'tokenizer_bpe': Path(default_bpe()),
                 'extractor': Path(__file__).resolve(), 'common_extractor': Path(base.__file__).resolve()}
        file_info = {}
        for name, path in files.items():
            print('[hash] {}'.format(name), flush=True)
            file_info[name] = {'path': str(path.resolve()), 'sha256': base.sha256_file(path)}
        identity_data = {
            'schema_version': 1, 'subject': 1, 'caption_policy': 'first_nonempty_coco_per_image',
            'max_images_per_split': args.max_images, 'tokens_representation': TOKEN_REPR,
            'pooled_representation': POOLED_REPR, 'tokenizer': 'open_clip.tokenize context_length=77',
            'output_dtype': 'float16', 'precision': args.precision, 'batch_size': args.batch_size,
            'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'torch_version': str(torch.__version__), 'open_clip_version': str(open_clip.__version__),
            'files': file_info, 'clip_l_source_identity_sha256': aligned['train']['source_sha256'],
            'caption_manifest_sha256': {k: v['manifest_sha256'] for k, v in aligned.items()},
            'caption_token_ids_sha256': {k: hashlib.sha256(text_ids[k].numpy().tobytes()).hexdigest() for k in aligned},
            'images': {k: len(v['ids']) for k, v in aligned.items()},
            'checkpoint_note': 'Local OpenCLIP bigG weights; equality to a future SDXL base checkpoint must be verified separately.',
            'test_usage': 'held-out targets only, never model inputs or selection',
        }
        identity = base.digest_json(identity_data)
        identity_path = directory / 'source_identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity_data:
                raise ValueError('BigG source or settings changed; use original parameters or a new output directory')
        else:
            if [x for x in directory.iterdir() if x.name != '.extract.lock']:
                raise ValueError('Output directory is not empty and has no source identity')
            base.atomic_json(identity_path, identity_data)
        ensure_space(directory, aligned)
        for split, values in aligned.items():
            base.save_array_once(directory / '{}_image_ids.npy'.format(split), values['ids'])
            base.save_array_once(directory / '{}_caption_annotation_ids.npy'.format(split), values['annotation_ids'])
        model = load_model(args.checkpoint, device, args.precision)
        verification = verify_against_openclip(model, text_ids['train'][:2].to(device), args.precision)
        if not (directory / 'verification.json').exists():
            base.atomic_json(directory / 'verification.json', verification)
        with torch.inference_mode():
            empty_tokens, empty_pool = encode(model, open_clip.tokenize([''], context_length=77).to(device))
        base.save_array_once(directory / 'empty_prompt_bigg_tokens.npy', empty_tokens.half().cpu().numpy())
        base.save_array_once(directory / 'empty_prompt_bigg_pooled.npy', empty_pool.half().cpu().numpy())
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        for split, values in aligned.items():
            result = PairedArrays(directory, split, len(values['ids']), identity)
            if not extract(model, text_ids[split], result, args, device):
                print('[paused] paired progress saved at {}'.format(directory), flush=True)
                return 128 + STOP_SIGNAL if STOP_SIGNAL else 0
            result.finish({'split': split, 'subject': 1, 'token_representation': TOKEN_REPR,
                           'pooled_representation': POOLED_REPR, 'caption_policy': 'first',
                           'clip_l_source_identity_sha256': values['source_sha256'],
                           'source_caption_manifest_sha256': values['manifest_sha256'],
                           'image_ids_sha256': base.sha256_file(directory / '{}_image_ids.npy'.format(split)),
                           'caption_annotation_ids_sha256': base.sha256_file(
                               directory / '{}_caption_annotation_ids.npy'.format(split))})
        print('[complete] {}'.format(directory), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
