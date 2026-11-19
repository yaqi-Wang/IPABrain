#!/usr/bin/env python3
"""Encode subj01 original-stimulus Qwen/GIT captions into SDXL text branches.

The captions originate from the held-out *original test stimulus images*, so
these embeddings are oracle/reference conditions, not fMRI predictions. See
the accompanying README for the native 77-token truncation and output format.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import open_clip
from open_clip.tokenizer import SimpleTokenizer, default_bpe

import extract_sdxl_clip_l11 as common
import extract_sdxl_bigg_text as bigg


ROOT = Path(__file__).resolve().parents[2]
CAPTION_ROOT = ROOT / 'output_1104/imagery_caption_v1'
DEFAULT_OUTPUT = CAPTION_ROOT / 'sdxl_text/stimulus_qwen_git5_truncate77_v1'
QWEN = CAPTION_ROOT / 'runs/stimulus_sample00/qwen3_5_2b/captions.json'
GIT_BEAM5 = CAPTION_ROOT / 'experiments/git_beam5/stimulus_sample00/git_large_coco/captions.json'
GIT_GREEDY = CAPTION_ROOT / 'runs/stimulus_sample00/git_large_coco/captions.json'
CLIP_DIR = common.CLIP_DIR
BIGG_CHECKPOINT = bigg.CHECKPOINT
L_IDENTITY = ROOT / 'output_1104/nsd_subj01/sdxl_clip_l11_all_both_v1/source_identity.json'
G_IDENTITY = ROOT / 'output_1104/nsd_subj01/sdxl_bigg_text_first_both_v1/source_identity.json'
CONTEXT = 77


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def caption_pairs(git_source, limit, qwen_path=QWEN):
    paths = {'qwen': qwen_path, 'git_' + git_source: GIT_BEAM5 if git_source == 'beam5' else GIT_GREEDY}
    contents = {name: read_json(path) for name, path in paths.items()}
    qwen, git = contents['qwen'], contents['git_' + git_source]
    for name, data in contents.items():
        if data['metadata']['source'] != 'stimulus' or data['metadata']['subject'] != 1:
            raise ValueError('{} is not subj01 original-stimulus captions'.format(name))
        if len(data['captions']) != 12:
            raise ValueError('{} does not contain 12 captions'.format(name))
    if git['metadata']['generation']['num_beams'] != (5 if git_source == 'beam5' else 1):
        raise ValueError('GIT beam count does not match requested caption source')
    seen = set()
    group_counts = Counter()
    selected = []
    for a, b in zip(qwen['captions'], git['captions']):
        for key in ('index', 'group', 'stimulus_id', 'image_sha256', 'image_path'):
            if a[key] != b[key]:
                raise ValueError('Qwen/GIT inputs mismatch at {}: {}'.format(a['index'], key))
        if a['stimulus_id'] in seen or a['group'] not in ('natural', 'geometry'):
            raise ValueError('Duplicate stimulus or unsupported category')
        seen.add(a['stimulus_id'])
        group_counts[a['group']] += 1
        if not Path(a['image_path']).is_file() or common.sha256_file(a['image_path']) != a['image_sha256']:
            raise ValueError('Original stimulus image changed or is missing: {}'.format(a['image_path']))
        if not str(a['caption']).strip() or not str(b['caption']).strip():
            raise ValueError('Empty caption for {}'.format(a['stimulus_id']))
        if not limit or group_counts[a['group']] <= limit:
            selected.append((a, b))
    if group_counts != {'natural': 6, 'geometry': 6}:
        raise ValueError('Expected six natural and six geometric stimuli')
    records = []
    for group_name, position in [('qwen', 0), ('git_' + git_source, 1)]:
        for index, pair in enumerate(selected):
            row = pair[position]
            records.append({'row_index': index, 'caption_source': group_name,
                            'stimulus_id': row['stimulus_id'], 'stimulus_group': row['group'],
                            'image_path': row['image_path'], 'image_sha256': row['image_sha256'],
                            'caption': str(row['caption']).strip()})
    return records, paths


def prepare_tokens(records, allow_truncation):
    tokenizer_l = CLIPTokenizer.from_pretrained(str(CLIP_DIR), local_files_only=True)
    tokenizer_g = SimpleTokenizer()
    texts = [x['caption'] for x in records]
    # Exactly the tokenization used by the existing CLIP-L/OpenCLIP text branches.
    ids_l = tokenizer_l(texts, padding='max_length', truncation=True, max_length=CONTEXT,
                        return_tensors='pt')['input_ids']
    ids_g = open_clip.tokenize(texts, context_length=CONTEXT)
    if tuple(ids_l.shape) != (len(records), CONTEXT) or tuple(ids_g.shape) != (len(records), CONTEXT):
        raise ValueError('Unexpected tokenized input shape')
    for index, record in enumerate(records):
        text = record['caption']
        l_count = len(tokenizer_l.encode(text, add_special_tokens=False, verbose=False)) + 2
        g_count = len(tokenizer_g.encode(text)) + 2
        g_ids = ids_g[index].tolist()
        eot = g_ids.index(tokenizer_g.eot_token_id)
        record.update({'clip_l_original_token_count_including_bos_eos': l_count,
                       'bigg_original_token_count_including_bos_eos': g_count,
                       'clip_l_truncated': l_count > CONTEXT, 'bigg_truncated': g_count > CONTEXT,
                       'clip_l_effective_text': tokenizer_l.decode(
                           ids_l[index].tolist(), skip_special_tokens=True,
                           clean_up_tokenization_spaces=False),
                       'bigg_effective_text': tokenizer_g.decode(g_ids[1:eot]).strip()})
    truncated = [r for r in records if r['clip_l_truncated'] or r['bigg_truncated']]
    print('[tokenize] captions={} truncated={} context={} (includes BOS/EOS)'.format(
        len(records), len(truncated), CONTEXT), flush=True)
    for name in sorted({r['caption_source'] for r in records}):
        selected = [r for r in records if r['caption_source'] == name]
        print('[tokenize] {}: count={}, CLIP-L truncated={}, bigG truncated={}'.format(
            name, len(selected), sum(r['clip_l_truncated'] for r in selected),
            sum(r['bigg_truncated'] for r in selected)), flush=True)
    if truncated and not allow_truncation:
        raise ValueError('Captions exceed 77 tokens; explicitly pass --allow-truncation to use the native SDXL prefix')
    return ids_l, ids_g


def weight_provenance():
    expected_l = read_json(L_IDENTITY)['sources']['model_weights']['sha256']
    expected_g = read_json(G_IDENTITY)['files']['checkpoint']['sha256']
    result = {}
    for name, path, expected in [('clip_l', CLIP_DIR / 'model.safetensors', expected_l),
                                 ('bigg', BIGG_CHECKPOINT, expected_g)]:
        print('[weights] SHA-256 {} ({:.2f} GiB) ...'.format(name, path.stat().st_size / 1024**3), flush=True)
        actual = common.sha256_file(path)
        if actual != expected:
            raise ValueError('{} weights differ from existing NSD three-branch targets'.format(name))
        result[name] = {'path': str(path), 'sha256': actual,
                        'expected_from_nsd_targets': expected}
    return result


def batches(ids, batch_size):
    for start in range(0, len(ids), batch_size):
        yield start, min(start + batch_size, len(ids))


def compute_clip_l(ids, device, precision, batch_size):
    dtype = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}[precision]
    print('[model] loading frozen OpenAI CLIP-L text tower', flush=True)
    model = CLIPTextModel.from_pretrained(str(CLIP_DIR), local_files_only=True, use_safetensors=True,
                                         torch_dtype=dtype, attn_implementation='sdpa')
    model = model.to(device).eval().requires_grad_(False)
    # Existing extraction optimized implementation independently checks hidden_states[11].
    verifier = common.FastClipL11(model).to(device).eval().requires_grad_(False)
    stats = common.verify_fast(model, verifier, ids[:min(4, len(ids))].to(device), precision)
    del verifier
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    parts = []
    with torch.inference_mode():
        for start, end in batches(ids, batch_size):
            input_ids = ids[start:end].to(device)
            output = model(input_ids=input_ids, output_hidden_states=True, return_dict=True).hidden_states[11]
            if tuple(output.shape) != (end - start, 77, 768) or not torch.isfinite(output).all():
                raise FloatingPointError('Invalid CLIP-L output')
            parts.append(output.float().cpu().numpy())
            print('[clip_l] {}/{}'.format(end, len(ids)), flush=True)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(parts, axis=0), stats


def compute_bigg(ids, device, precision, batch_size):
    print('[model] loading frozen OpenCLIP-bigG text tower only', flush=True)
    model = bigg.load_model(BIGG_CHECKPOINT, device, precision)
    stats = bigg.verify_against_openclip(model, ids[:min(2, len(ids))].to(device), precision)
    token_parts, pooled_parts = [], []
    with torch.inference_mode():
        for start, end in batches(ids, batch_size):
            tokens, pooled = bigg.encode(model, ids[start:end].to(device))
            if (tuple(tokens.shape) != (end - start, 77, 1280) or tuple(pooled.shape) != (end - start, 1280)
                    or not torch.isfinite(tokens).all() or not torch.isfinite(pooled).all()):
                raise FloatingPointError('Invalid OpenCLIP-bigG output')
            token_parts.append(tokens.float().cpu().numpy())
            pooled_parts.append(pooled.float().cpu().numpy())
            print('[bigg] {}/{}'.format(end, len(ids)), flush=True)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(token_parts, axis=0), np.concatenate(pooled_parts, axis=0), stats


def save_array(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(str(temp), str(path))


def write_results(out, records, paths, ids_l, ids_g, clip_l, tokens_g, pooled_g, metadata, output_dtype):
    sources = list(dict.fromkeys(row['caption_source'] for row in records))
    output_files = {}
    for source in sources:
        indices = [i for i, row in enumerate(records) if row['caption_source'] == source]
        folder = out / source
        folder.mkdir(parents=True, exist_ok=False)
        arrays = {'clip_l.npy': clip_l[indices].astype(output_dtype),
                  'bigg_tokens.npy': tokens_g[indices].astype(output_dtype),
                  'bigg_pooled.npy': pooled_g[indices].astype(output_dtype),
                  'clip_l_input_ids.npy': ids_l[indices].numpy(),
                  'bigg_input_ids.npy': ids_g[indices].numpy()}
        for name, value in arrays.items():
            path = folder / name
            save_array(path, value)
            output_files[str(path.relative_to(out))] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                                                       'sha256': common.sha256_file(path)}
        manifest = folder / 'captions.jsonl'
        with manifest.open('w', encoding='utf-8') as handle:
            for index in indices:
                handle.write(json.dumps(records[index], ensure_ascii=False, sort_keys=True) + '\n')
        output_files[str(manifest.relative_to(out))] = {'records': len(indices),
                                                        'sha256': common.sha256_file(manifest)}
    metadata['output_files'] = output_files
    metadata['source_captions'] = {name: {'path': str(path), 'sha256': common.sha256_file(path)}
                                   for name, path in paths.items()}
    metadata['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    common.atomic_json(out / 'extraction_complete.json', metadata)
    print('[complete] {}'.format(out / 'extraction_complete.json'), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('float16', 'bfloat16', 'float32'), default='float16')
    parser.add_argument('--output-dtype', choices=('float16', 'float32'), default='float32')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--git-source', choices=('beam5', 'greedy'), default='beam5')
    parser.add_argument('--qwen-caption-file', type=Path, default=QWEN,
                        help='Qwen-compatible caption JSON; use a new output directory for revised captions')
    parser.add_argument('--allow-truncation', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit-per-group', type=int, choices=range(1, 7),
                        help='smoke test: first N natural and N geometric images; use another output dir')
    parser.add_argument('--dry-run', action='store_true', help='validate input and tokenization without GPU')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.limit_per_group and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error('--limit-per-group requires a separate --output-dir')
    if args.git_source != 'beam5' and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error('nondefault GIT captions require a separate --output-dir')
    if args.qwen_caption_file.resolve() != QWEN.resolve() and args.output_dir.resolve() == DEFAULT_OUTPUT.resolve():
        parser.error('revised captions require a separate --output-dir')
    if args.device == 'cpu' and args.precision != 'float32':
        parser.error('CPU inference requires --precision float32')
    return args


def main():
    args = parse_args()
    records, paths = caption_pairs(args.git_source, args.limit_per_group, args.qwen_caption_file.resolve())
    ids_l, ids_g = prepare_tokens(records, args.allow_truncation)
    print('[input] original stimulus images; oracle/reference conditions, not fMRI predictions', flush=True)
    if args.dry_run:
        print('[dry-run] no model loaded and no embeddings written', flush=True)
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for {}'.format(device))
    if device.type == 'cuda' and args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bfloat16')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output_dir.resolve()
    started = time.monotonic()
    with common.extraction_lock(out):
        prior = [p.name for p in out.iterdir() if p.name != '.extract.lock']
        if prior:
            raise FileExistsError('Output directory is not empty ({}); choose a new --output-dir'.format(prior[:5]))
        weights = weight_provenance()
        clip_l, l_check = compute_clip_l(ids_l, device, args.precision, args.batch_size)
        g_tokens, g_pooled, g_check = compute_bigg(ids_g, device, args.precision, args.batch_size)
        metadata = {
            'experiment': 'imagery_subj01_original_stimulus_caption_text_features',
            'input_role': 'oracle/reference: captions generated from original test stimuli, not fMRI',
            'subject': 1, 'git_source': args.git_source, 'count_per_caption_source': len(records) // 2,
            'image_scope': 'six natural and six geometry' if not args.limit_per_group else 'smoke test',
            'tokenization': 'native CLIP-L and OpenCLIP, padded/truncated to 77 including BOS/EOS',
            'truncation_allowed': args.allow_truncation, 'precision': args.precision,
            'output_dtype': args.output_dtype, 'batch_size': args.batch_size,
            'representations': {'clip_l': 'hidden_states[11] before final layernorm, no projection',
                                'bigg_tokens': bigg.TOKEN_REPR, 'bigg_pooled': bigg.POOLED_REPR},
            'weights': weights, 'validation': {'clip_l_fast_vs_hf': l_check, 'bigg_vs_native': g_check},
            'script_sha256': common.sha256_file(Path(__file__)),
            'existing_extractors_sha256': {'clip_l': common.sha256_file(Path(common.__file__)),
                                           'bigg': common.sha256_file(Path(bigg.__file__))},
            'tokenizer_files_sha256': {'clip_l_vocab': common.sha256_file(CLIP_DIR / 'vocab.json'),
                                       'clip_l_merges': common.sha256_file(CLIP_DIR / 'merges.txt'),
                                       'bigg_bpe': common.sha256_file(Path(default_bpe()))},
            'versions': {'python': sys.version.split()[0], 'torch': str(torch.__version__),
                         'transformers': transformers.__version__, 'open_clip': open_clip.__version__,
                         'numpy': np.__version__},
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'elapsed_seconds_before_save': round(time.monotonic() - started, 2),
        }
        write_results(out, records, paths, ids_l, ids_g, clip_l, g_tokens, g_pooled,
                      metadata, args.output_dtype)


if __name__ == '__main__':
    main()
