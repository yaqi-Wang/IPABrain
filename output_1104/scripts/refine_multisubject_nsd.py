#!/usr/bin/env python3
"""Refine subject-specific w010 NSD images with Ridge text and Zavy SDXL.

Sampling follows MindEye2's enhanced_recon_inference notebook: 38-step legacy
schedule, timepoint 13, Euler EDM, CFG 5 and the notebook's negative prompt.
Positive text conditions are saved fMRI predictions, never true captions.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / 'src/sdxl/generative_models'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'output_1104/scripts'))
sys.path.insert(0, str(GEN))
sys.path.insert(0, str(GEN.parent))

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

from extract_sdxl_clip_l11 import CLIP_DIR, digest_json, sha256_file
from src.data.scheme1 import validate_tag
from src.utils.clip_mapping import mapping_lock
from multisubject_resume import archive_file

BASE = ROOT / ('outputs_new/w010_multisubject_v1/subj01/runs/w010_reused/evaluation/'
               'w010_selected_nsd_test_all_s42/nsd_test')
RIDGE = ROOT / 'output_1104/nsd_subj01/ridge_three_text/ridge_mu_three_text_v3'
ZAVY = ROOT / 'huggingface/zavychromaxl_v30.safetensors'
ZAVY_SHA = 'f1270c1f025398346c59dab6d9ee234ec50b8f650b01f19376f9b647e6411b75'
FORMAL = ROOT / 'outputs_new/mindeye2_img2img/subj01/runs'
DEBUG = ROOT / 'outputs_new/mindeye2_img2img/debug'
BRANCHES = {'clip_l': (77, 768), 'bigg_tokens': (77, 1280), 'bigg_pooled': (1280,)}
NEGATIVE = ('painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, '
            'deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, '
            'skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing '
            'lips, ugly face, distorted face, extra legs, anime')
STOP_SIGNAL = 0


def on_signal(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum
    print('[signal] {} received; stopping after the current image'.format(signum), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(str(temp), str(path))


def pinned_image_path(item):
    base = (BASE / 'sample00').resolve()
    path = (base / item['path']).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        raise ValueError('Initial image path escapes the w010 sample directory')
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def load_sources(limit, debug_generation=None):
    if sha256_file(ZAVY) != ZAVY_SHA:
        raise ValueError('Zavy checkpoint SHA-256 does not match the official release')
    base_meta = read_json(BASE / 'metadata.json')
    base_manifest = read_json(BASE / 'sample00/manifest.json')
    ridge_protocol = read_json(RIDGE / 'protocol.json')
    ridge_model = read_json(RIDGE / 'model/model_complete.json')
    pred_dir = RIDGE / 'nsd_test'
    pred_info = read_json(pred_dir / 'prediction_complete.json')
    ids = np.load(pred_dir / 'image_ids.npy', allow_pickle=False)
    records = base_manifest['items']
    total = len(ids)
    subject = base_meta['subject']
    if (subject not in range(1, 9) or ridge_protocol.get('subject') != subject or base_meta['dataset'] != 'nsd_test'
            or base_meta['aggregation'] != 'mean_fmri' or base_meta['n_items'] != total
            or base_meta['num_samples'] != 1 or base_meta['partial_test']
            or base_meta['debug_training'] or base_meta.get('rgb_mode') != 'standard'
            or any(base_manifest.get(key) != base_meta.get(key)
                   for key in ('checkpoint_sha256', 'data_fingerprint', 'subject', 'dataset', 'num_samples'))
            or total == 0 or len(records) != total or len(np.unique(ids)) != total
            or not np.array_equal(ids, np.asarray([x['image_id'] for x in records]))
            or ridge_model.get('status') != 'complete' or ridge_model.get('debug')
            or ridge_protocol.get('test_used_for_selection') is not False
            or pred_info.get('status') != 'complete' or pred_info.get('images') != total
            or pred_info.get('test_text_used_as_input') is not False
            or base_meta['brainvae_checkpoint_sha256'] != ridge_model['brain_source']['brainvae_checkpoint_sha256']
            or ridge_protocol['brain_source'] != ridge_model['brain_source']):
        raise ValueError('w010 and Ridge sources are incomplete or their subject/image/BrainVAE identities differ')
    arrays, target_hashes = {}, {}
    for name, shape in BRANCHES.items():
        path = pred_dir / ('predicted_' + name + '.npy')
        actual = sha256_file(path)
        if actual != pred_info['predictions_sha256'][name]:
            raise ValueError('Ridge {} prediction SHA-256 mismatch'.format(name))
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.shape != (total, *shape) or values.dtype != np.float32:
            raise ValueError('Ridge {} has an unexpected shape or dtype'.format(name))
        arrays[name], target_hashes[name] = values, actual
    count = min(limit, total) if limit else total
    selected = records[:count]
    paths = [pinned_image_path(item) for item in selected]
    def fingerprint(pair):
        index, path = pair
        with Image.open(path) as image:
            if image.size != (768, 768):
                raise ValueError('Expected 768x768 w010 input: {}'.format(path))
            image.verify()
        return {'image_id': int(selected[index]['image_id']), 'sha256': sha256_file(path)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        image_hashes = list(pool.map(fingerprint, enumerate(paths)))
    identity = {'subject': subject, 'dataset': 'nsd_test', 'count': count,
                'debug_generation': bool(limit) if debug_generation is None else bool(debug_generation),
                'zavy_sha256': ZAVY_SHA,
                'w010_metadata_sha256': sha256_file(BASE / 'metadata.json'),
                'w010_manifest_sha256': sha256_file(BASE / 'sample00/manifest.json'),
                'w010_checkpoint_sha256': base_meta['checkpoint_sha256'],
                'brainvae_checkpoint_sha256': base_meta['brainvae_checkpoint_sha256'],
                'ridge_protocol_sha256': sha256_file(RIDGE / 'protocol.json'),
                'ridge_model_metadata_sha256': sha256_file(RIDGE / 'model/model_complete.json'),
                'ridge_prediction_metadata_sha256': sha256_file(pred_dir / 'prediction_complete.json'),
                'ridge_image_ids_sha256': sha256_file(pred_dir / 'image_ids.npy'),
                'ridge_prediction_sha256': target_hashes,
                'w010_input_images_sha256': digest_json(image_hashes),
                'test_text_used_as_input': False, 'true_image_used_as_input': False}
    return base_meta, selected, paths, arrays, identity


def checkpoint_text_state(checkpoint, prefix, names):
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        available = set(source.keys())
        missing = [name for name in names if prefix + name not in available]
        if missing:
            raise ValueError('Zavy is missing text tensors: {}'.format(missing[:5]))
        return {name: source.get_tensor(prefix + name) for name in names}


@torch.no_grad()
def negative_condition(device):
    """Original notebook obtains negative text from Zavy's conditioner towers."""
    import open_clip
    from open_clip.transformer import TextTransformer
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from sgm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
    import extract_sdxl_bigg_text as bigg

    tokenizer = CLIPTokenizer.from_pretrained(str(CLIP_DIR), local_files_only=True)
    text_config = CLIPTextConfig.from_pretrained(str(CLIP_DIR), local_files_only=True)
    with torch.device('meta'):
        clip_l = CLIPTextModel(text_config)
    state = checkpoint_text_state(ZAVY, 'conditioner.embedders.0.transformer.', list(clip_l.state_dict()))
    clip_l.load_state_dict(state, strict=True, assign=True)
    del state
    # transformers position_ids is a nonpersistent buffer; the checkpoint's
    # legacy position_ids tensor is ignored just as in the MindEye2 notebook.
    clip_l.text_model.embeddings.position_ids = torch.arange(77, dtype=torch.long).expand((1, -1))
    clip_l = clip_l.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    tokens_l = tokenizer([NEGATIVE], padding='max_length', truncation=True, max_length=77,
                         return_tensors='pt').input_ids.to(device)
    with torch.autocast('cuda', dtype=torch.float16):
        l_tokens = clip_l(tokens_l, output_hidden_states=True).hidden_states[11].float().cpu()
    native_l = FrozenCLIPEmbedder.__new__(FrozenCLIPEmbedder)
    torch.nn.Module.__init__(native_l)
    native_l.transformer, native_l.tokenizer = clip_l, tokenizer
    native_l.device, native_l.max_length = str(device), 77
    native_l.layer, native_l.layer_idx, native_l.return_pooled = 'hidden', 11, False
    with torch.autocast('cuda', dtype=torch.float16):
        reference_l = native_l([NEGATIVE]).float().cpu()
    l_error = float((l_tokens - reference_l).abs().max())
    if l_error > 0.002:
        raise FloatingPointError('Negative CLIP-L differs from native SGM')
    del native_l, reference_l
    del clip_l, tokens_l
    torch.cuda.empty_cache()

    with torch.device('meta'):
        tower = TextTransformer(context_length=77, vocab_size=49408, width=1280,
                                heads=20, layers=32, output_dim=1280)
    state = checkpoint_text_state(ZAVY, 'conditioner.embedders.1.model.', list(tower.state_dict()))
    tower.load_state_dict(state, strict=True, assign=True)
    del state
    tower.attn_mask = tower.build_causal_mask()
    tower = tower.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    tokens_g = open_clip.tokenize([NEGATIVE], context_length=77).to(device)
    with torch.autocast('cuda', dtype=torch.float16):
        g_tokens, pooled = bigg.encode(tower, tokens_g)
    g_tokens, pooled = g_tokens.float().cpu(), pooled.float().cpu()
    native_g = FrozenOpenCLIPEmbedder2.__new__(FrozenOpenCLIPEmbedder2)
    torch.nn.Module.__init__(native_g)
    native_g.model, native_g.legacy, native_g.layer_idx = tower, False, 1
    with torch.autocast('cuda', dtype=torch.float16):
        reference_g = native_g.encode_with_transformer(tokens_g)
    g_error = float((g_tokens - reference_g['penultimate'].float().cpu()).abs().max())
    p_error = float((pooled - reference_g['pooled'].float().cpu()).abs().max())
    if max(g_error, p_error) > 0.002:
        raise FloatingPointError('Negative OpenCLIP differs from native SGM')
    del native_g, reference_g
    del tower, tokens_g
    torch.cuda.empty_cache()
    if (tuple(l_tokens.shape) != (1, 77, 768) or tuple(g_tokens.shape) != (1, 77, 1280)
            or tuple(pooled.shape) != (1, 1280)
            or not all(torch.isfinite(x).all() for x in (l_tokens, g_tokens, pooled))):
        raise FloatingPointError('Invalid negative prompt conditions')
    cross = torch.cat((l_tokens, g_tokens), dim=-1)
    audit = {'negative_prompt': NEGATIVE, 'source_zavy_sha256': ZAVY_SHA,
             'sgm_native_max_abs': {'clip_l': l_error, 'bigg_tokens': g_error, 'bigg_pooled': p_error},
             'crossattn_sha256': hashlib.sha256(cross.numpy().tobytes()).hexdigest(),
             'pooled_sha256': hashlib.sha256(pooled.numpy().tobytes()).hexdigest()}
    print('[text] native SGM negative-condition max_abs={}'.format(audit['sgm_native_max_abs']), flush=True)
    return cross, pooled, audit


def load_engine(device, steps, cfg_scale):
    from omegaconf import OmegaConf
    from sgm.models.diffusion import DiffusionEngine

    base = OmegaConf.to_container(OmegaConf.load(GEN / 'configs/inference/sd_xl_base.yaml'), resolve=True)
    unclip = OmegaConf.to_container(OmegaConf.load(GEN / 'configs/unclip6.yaml'), resolve=True)
    params = base['model']['params']
    network = params['network_config']
    network['params']['spatial_transformer_attn_type'] = 'softmax'  # native PyTorch SDPA
    network['params']['use_checkpoint'] = False
    first = params['first_stage_config']
    first['params']['ddconfig']['attn_type'] = 'vanilla'  # native PyTorch SDPA
    # Only size/crop embedders are required: positive/negative text is supplied directly.
    conditioner = params['conditioner_config']
    conditioner['params']['emb_models'] = conditioner['params']['emb_models'][2:]
    sampler = unclip['model']['params']['sampler_config']
    sampler['params']['num_steps'] = int(steps)
    sampler['params']['device'] = str(device)
    sampler['params']['guider_config']['params']['scale'] = float(cfg_scale)
    if float(params['scale_factor']) != 0.13025:
        raise ValueError('Unexpected SDXL VAE scale')
    print('[model] initializing SDXL UNet and VAE on CPU', flush=True)
    engine = DiffusionEngine(network_config=network, denoiser_config=params['denoiser_config'],
                             first_stage_config=first, conditioner_config=conditioner,
                             sampler_config=sampler, scale_factor=params['scale_factor'],
                             disable_first_stage_autocast=True).eval().requires_grad_(False)
    expected = engine.state_dict()
    with safe_open(str(ZAVY), framework='pt', device='cpu') as source:
        keys = set(source.keys())
        absent = [key for key in expected if key not in keys and key != 'denoiser.sigmas']
        if absent:
            raise ValueError('Zavy is missing required SDXL tensors: {}'.format(absent[:8]))
        state = {key: source.get_tensor(key) for key in expected if key in keys}
        missing, unexpected = engine.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {'denoiser.sigmas'}:
        raise ValueError('Unexpected SDXL load result: missing={} unexpected={}'.format(missing, unexpected))
    del state, expected
    # Frozen UNet weights are stored in FP16 in Zavy. GroupNorm32 receives FP32
    # input, so its affine parameters remain FP32; VAE stays FP32 for stability.
    engine.model.half()
    for layer in engine.model.modules():
        if isinstance(layer, torch.nn.GroupNorm):
            layer.float()
    engine.first_stage_model.float()
    gc.collect()
    engine.to(device)
    size = {'original_size_as_tuple': torch.full((1, 2), 768., device=device),
            'crop_coords_top_left': torch.zeros((1, 2), device=device),
            'target_size_as_tuple': torch.full((1, 2), 1024., device=device)}
    with torch.no_grad():
        suffix = engine.conditioner(size)['vector'].float()
    if tuple(suffix.shape) != (1, 1536):
        raise ValueError('Incorrect SDXL original/crop/target-size vector')
    print('[model] loaded Zavy; missing={} GPU allocated={:.2f} GiB'.format(
        missing, torch.cuda.memory_allocated(device) / 1024**3), flush=True)
    return engine, suffix


def conditions(arrays, index, suffix, negative_cross, negative_pool, device):
    values = {}
    for name, shape in BRANCHES.items():
        value = torch.from_numpy(np.array(arrays[name][index:index + 1], copy=True)).to(device)
        if tuple(value.shape) != (1, *shape) or not torch.isfinite(value).all():
            raise FloatingPointError('Invalid predicted {} condition at {}'.format(name, index))
        values[name] = value.float()
    cond = {'crossattn': torch.cat((values['clip_l'], values['bigg_tokens']), dim=-1),
            'vector': torch.cat((values['bigg_pooled'], suffix), dim=-1)}
    uncond = {'crossattn': negative_cross.to(device),
              'vector': torch.cat((negative_pool.to(device), suffix), dim=-1)}
    return cond, uncond


def img2img(engine, image, cond, uncond, seed, steps, timepoint):
    from sgm.util import append_dims
    device = image.device
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    sampler = engine.sampler
    sampler.num_steps = int(steps)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, cache_enabled=False), engine.ema_scope():
        latent = engine.encode_first_stage(image * 2.0 - 1.0)
        if tuple(latent.shape) != (1, 4, 96, 96) or not torch.isfinite(latent).all():
            raise FloatingPointError('Invalid initial SDXL VAE latent')
        full_sigmas = sampler.discretization(steps, device=str(device)).to(device)
        if timepoint < 2 or timepoint > len(full_sigmas):
            raise ValueError('Invalid MindEye2 img2img timepoint')
        noise = torch.randn_like(latent)
        z = (latent + noise * append_dims(full_sigmas[-timepoint], latent.ndim))
        z = z / torch.sqrt(1.0 + full_sigmas[0] ** 2.0)
        sigmas = full_sigmas[-timepoint:]
        sampler.num_steps = len(sigmas) - 1
        z, _, _, _, cond, uncond = sampler.prepare_sampling_loop(
            z, cond=cond, uc=uncond, num_steps=sampler.num_steps)
        def denoiser(value, sigma, conditioning):
            return engine.denoiser(engine.model, value, sigma, conditioning)
        for j in range(len(sigmas) - 1):
            z = sampler.sampler_step(sigmas[j].reshape(1), sigmas[j + 1].reshape(1),
                                     denoiser, z, cond, uncond, gamma=0.0)
        decoded = engine.decode_first_stage(z)
    rgb = ((decoded.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    if tuple(rgb.shape) != (1, 3, 768, 768) or not torch.isfinite(rgb).all():
        raise FloatingPointError('Nonfinite or invalid img2img output')
    return rgb


def read_progress(path, selected, sample_dir, base_seed):
    result = {}
    if not path.exists():
        return result
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError('Invalid progress line {}'.format(number)) from exc
        index = int(record['index'])
        if not 0 <= index < len(selected) or record['image_id'] != int(selected[index]['image_id']) \
                or record['seed'] != base_seed + index:
            raise ValueError('Image progress belongs to different inputs')
        result[index] = record
    complete = {}
    for index, record in result.items():
        path = sample_dir / selected[index]['output_path']
        if path.is_file() and path.stat().st_size and sha256_file(path) == record['image_sha256']:
            with Image.open(path) as image:
                image.verify()
            complete[index] = record
    return complete



def audit_reused_subj01(source,items,seed,steps,timepoint,cfg_scale):
    old_dir=ROOT/'output_1104/multisubject_nsd_imagery_v1/subj01/runs/w010_ridge_full_extend250_v1/nsd_test'
    old_meta=read_json(old_dir/'metadata.json');old_complete=read_json(old_dir/'generation_complete.json')
    old_manifest=read_json(old_dir/'sample00/manifest.json');old_log=old_dir/'sample00/progress.jsonl'
    old_identity=old_meta['source_identity']
    for key in ('w010_checkpoint_sha256','brainvae_checkpoint_sha256','ridge_model_metadata_sha256',
                'ridge_prediction_sha256','zavy_sha256','w010_metadata_sha256','w010_manifest_sha256'):
        if old_identity.get(key)!=source.get(key):raise ValueError('Existing subj01 sample identity mismatch: '+key)
    base_items=read_json(BASE/'sample00/manifest.json')['items']
    current=[{'image_id':int(item['image_id']),'sha256':sha256_file(pinned_image_path(item))} for item in base_items]
    if digest_json(current)!=old_identity['w010_input_images_sha256']:
        raise ValueError('Existing subj01 sample used different w010 input image bytes')
    if (old_complete.get('status')!='complete' or old_complete.get('manifest_sha256')!=sha256_file(old_dir/'sample00/manifest.json')
            or old_complete.get('progress_sha256')!=sha256_file(old_log) or old_identity['seed_base']!=seed
            or old_identity['steps']!=steps or old_identity['timepoint']!=timepoint or old_identity['cfg_scale']!=cfg_scale):
        raise ValueError('Existing subj01 sample protocol or completion mismatch')
    progress={int(x['index']):x for x in map(json.loads,old_log.read_text().splitlines())}
    for i,item in enumerate(items):
        old_item=old_manifest['items'][i];record=progress.get(i)
        if (old_item['image_id']!=item['image_id'] or old_item['seed']!=seed+i or not record
                or record['image_id']!=item['image_id'] or record['seed']!=seed+i
                or sha256_file(old_dir/'sample00'/old_item['path'])!=record['image_sha256']):
            raise ValueError('Existing subj01 sample image mismatch at '+str(i))
    return old_dir,progress

def main():
    global BASE, RIDGE, FORMAL, DEBUG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, choices=range(1, 9), required=True)
    parser.add_argument('--ridge-run', default='ridge_mu_three_text_v1')
    parser.add_argument('--run-name', default='w010_ridge_nsd250_3samples_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=38)
    parser.add_argument('--timepoint', type=int, default=13)
    parser.add_argument('--cfg-scale', type=float, default=5.0)
    parser.add_argument('--limit', type=int, default=0, help='debug prefix; written only under outputs_new/mindeye2_img2img/debug')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    validate_tag(args.run_name)
    validate_tag(args.ridge_run)
    if not 0 <= args.seed <= 2**63 - 201000 or args.steps < 2 or not 2 <= args.timepoint <= args.steps + 1 \
            or args.limit < 0 or args.limit > 250 \
            or not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        raise ValueError('Invalid sampling settings')
    selected = read_json(ROOT / f'outputs_new/w010_multisubject_v1/subj{args.subject:02d}/selection.json')['selected']
    checkpoint = Path(selected['checkpoint'])
    if sha256_file(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Selected w010 checkpoint digest mismatch')
    BASE = checkpoint.parent.parent / 'evaluation/w010_selected_nsd_test_all_s42/nsd_test'
    root = ROOT / f'output_1104/multisubject_nsd_imagery_v1/subj{args.subject:02d}'
    RIDGE = (ROOT / 'output_1104/nsd_subj01/ridge_three_text/ridge_mu_three_text_v3'
             if args.subject == 1 else root / 'ridge' / args.ridge_run)
    FORMAL = root / 'runs'
    DEBUG = ROOT / f'output_1104/multisubject_nsd_imagery_v1/debug/subj{args.subject:02d}/runs'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected_count = args.limit or 250
    base_meta, items, paths, arrays, source = load_sources(selected_count, debug_generation=bool(args.limit))
    if (base_meta['subject'] != args.subject or base_meta['checkpoint_sha256'] != selected['checkpoint_sha256']
            or any(int(item['subject_id']) != args.subject for item in items)):
        raise ValueError('Input manifest does not match the selected subject and checkpoint')
    print('[preflight] matched {} w010 images and three Ridge predictions; Zavy SHA verified'.format(len(items)), flush=True)
    reused_sample = audit_reused_subj01(source,items,args.seed,args.steps,args.timepoint,args.cfg_scale) if args.subject==1 else None
    if reused_sample:print('[preflight] existing subj01 first sample verified for reuse',flush=True)
    if args.dry_run:
        print('[dry-run] no output or GPU generation', flush=True)
        return 0
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; refusing silent CPU fallback')
    directory = (DEBUG if args.limit else FORMAL) / args.run_name / 'nsd_test'
    with mapping_lock(directory):
        required = len(items) * 3 * 2500000 + 1024**3
        if shutil.disk_usage(directory).free < required:
            raise RuntimeError('Insufficient output space for the requested images')
        from open_clip.tokenizer import default_bpe
        import open_clip
        import transformers
        import torchvision
        run_identity = dict(source, seed_base=args.seed, seed_policy='seed_base_plus_sample_times_100000_plus_item',
                            samples=3, steps=args.steps, timepoint=args.timepoint, actual_steps=args.timepoint - 1,
                            cfg_scale=args.cfg_scale, negative_prompt=NEGATIVE,
                            resolution=[768, 768], original_size=[768, 768], crop=[0, 0], target_size=[1024, 1024],
                            sampler='MindEye2 notebook EulerEDM + LegacyDDPMDiscretization',
                            vae_posterior='sampled', unet_precision='fp16_with_fp32_groupnorm',
                            vae_precision='fp32', attention='pytorch_sdpa',
                            zavy_negative_text_towers=True, positive_text='ridge_predicted_pretrained_towers',
                            torch_version=str(torch.__version__), gpu_name=torch.cuda.get_device_name(device),
                            open_clip_version=str(open_clip.__version__), transformers_version=str(transformers.__version__),
                            torchvision_version=str(torchvision.__version__),
                            clip_tokenizer_vocab_sha256=sha256_file(CLIP_DIR / 'vocab.json'),
                            clip_tokenizer_merges_sha256=sha256_file(CLIP_DIR / 'merges.txt'),
                            clip_tokenizer_config_sha256=sha256_file(CLIP_DIR / 'tokenizer_config.json'),
                            openclip_bpe_sha256=sha256_file(Path(default_bpe())),
                            runner_sha256=sha256_file(Path(__file__)),
                            base_config_sha256=sha256_file(GEN / 'configs/inference/sd_xl_base.yaml'),
                            sampler_config_sha256=sha256_file(GEN / 'configs/unclip6.yaml'),
                            sampler_implementation_sha256=sha256_file(GEN / 'sgm/modules/diffusionmodules/sampling.py'),
                            model_implementation_sha256=sha256_file(GEN / 'sgm/models/diffusion.py'))
        if args.subject == 1:
            old_dir, old_progress = reused_sample
            run_identity['reused_sample00']={'metadata_sha256':sha256_file(old_dir/'metadata.json'),
                'manifest_sha256':sha256_file(old_dir/'sample00/manifest.json'),
                'progress_sha256':sha256_file(old_dir/'sample00/progress.jsonl'),
                'generation_complete_sha256':sha256_file(old_dir/'generation_complete.json')}
        run_hash = digest_json(run_identity)
        metadata = {'protocol':'w010_ridge_nsd250_zavy_3samples_v1','subject':args.subject,'dataset':'nsd_test',
                    'aggregation':'mean_fmri','n_items':len(items),'num_samples':3,
                    'partial_test':len(items)!=base_meta['n_items'],'debug_training':False,
                    'debug_generation':bool(args.limit),'data_fingerprint':base_meta['data_fingerprint'],
                    'checkpoint':base_meta['checkpoint'],'checkpoint_sha256':base_meta['checkpoint_sha256'],
                    'brainvae_checkpoint_sha256':base_meta['brainvae_checkpoint_sha256'],
                    'raw_nsd_root':base_meta['raw_nsd_root'],'rgb_mode':'standard',
                    'generator_backend':'mindeye2_zavy_sdxl_img2img','text_condition':'ridge_predicted',
                    'text_oracle':False,'oracle':False,'target_usage':'evaluation_only',
                    'refiner_checkpoint':str(ZAVY),'refiner_checkpoint_sha256':ZAVY_SHA,
                    'source_identity_sha256':run_hash,'source_identity':run_identity}
        meta_path=directory/'metadata.json'
        if meta_path.exists() and read_json(meta_path)!=metadata:raise ValueError('Existing output has different identity')
        if not meta_path.exists():write_json(meta_path,metadata)
        pending=[]; selections={}
        for sample_index in range(3):
            sample=directory/f'sample{sample_index:02d}'
            (sample/'images').mkdir(parents=True,exist_ok=True)
            seed_base=args.seed+sample_index*100000
            selected=[]
            for index,item in enumerate(items):
                record={k:item[k] for k in ('subject_id','image_id','stimulus_id','stimulus_type','task_type','n_trials','item_index')}
                record.update(sample_index=sample_index,seed=seed_base+index,
                              output_path='images/item{:05d}_nsd{:05d}.png'.format(index,int(item['image_id'])))
                selected.append(record)
            selections[sample_index]=selected
            manifest_items=[{**{k:v for k,v in item.items() if k!='output_path'},'path':item['output_path']} for item in selected]
            manifest=dict(metadata,level='stimulus',sample_index=sample_index,n_images=len(items),items=manifest_items)
            path=sample/'manifest.json'
            if path.exists() and read_json(path)!=manifest:raise ValueError('Existing sample manifest differs')
            if not path.exists():write_json(path,manifest)
            done=read_progress(sample/'progress.jsonl',selected,sample,seed_base)
            if args.subject==1 and sample_index==0:
                old_manifest=read_json(old_dir/'sample00/manifest.json')
                with (sample/'progress.jsonl').open('a',encoding='utf-8') as progress:
                    for index in range(len(items)):
                        if index in done:continue
                        src=old_dir/'sample00'/old_manifest['items'][index]['path']
                        dest=sample/selected[index]['output_path']
                        if dest.exists():archive_file(dest)
                        temp=dest.with_name(dest.name+'.tmp')
                        shutil.copyfile(src,temp);os.replace(temp,dest)
                        if sha256_file(dest)!=old_progress[index]['image_sha256']:raise ValueError('Copied image checksum mismatch')
                        progress.write(json.dumps(old_progress[index])+'\n');progress.flush()
                done=read_progress(sample/'progress.jsonl',selected,sample,seed_base)
            pending.extend((sample_index,index) for index in range(len(items)) if index not in done)
            print('[resume] sample={} verified={} pending={}'.format(sample_index,len(done),len(items)-len(done)),flush=True)
        if pending:
            signal.signal(signal.SIGINT,on_signal);signal.signal(signal.SIGTERM,on_signal)
            negative_cross,negative_pool,negative_audit=negative_condition(device)
            negative_path=directory/'negative_condition_audit.json'
            if negative_path.exists() and read_json(negative_path)!=negative_audit:raise ValueError('Negative condition differs')
            if not negative_path.exists():write_json(negative_path,negative_audit)
            if STOP_SIGNAL:return 128+STOP_SIGNAL
            engine,suffix=load_engine(device,args.steps,args.cfg_scale)
            if STOP_SIGNAL:return 128+STOP_SIGNAL
            elapsed=time.monotonic()
            for number,(sample_index,index) in enumerate(pending,1):
                sample=directory/f'sample{sample_index:02d}'
                seed=args.seed+sample_index*100000+index
                with Image.open(paths[index]) as image:
                    image=ImageOps.exif_transpose(image).convert('RGB')
                    if image.size!=(768,768):raise ValueError('Input image dimensions changed')
                    tensor=pil_to_tensor(image).float().div_(255.0).unsqueeze(0).to(device)
                cond,uncond=conditions(arrays,index,suffix,negative_cross,negative_pool,device)
                rgb=img2img(engine,tensor,cond,uncond,seed,args.steps,args.timepoint)
                dest=sample/selections[sample_index][index]['output_path']
                if dest.exists():archive_file(dest)
                temp=dest.with_name(dest.stem+'.tmp.png')
                save_image(rgb.cpu(),str(temp));os.replace(str(temp),str(dest))
                record={'index':index,'image_id':int(items[index]['image_id']),'seed':seed,'image_sha256':sha256_file(dest)}
                with (sample/'progress.jsonl').open('a',encoding='utf-8') as progress:
                    progress.write(json.dumps(record)+'\n');progress.flush()
                print('[image] {}/{} sample={} id={} {:.2f}s/image'.format(number,len(pending),sample_index,record['image_id'],(time.monotonic()-elapsed)/number),flush=True)
                if STOP_SIGNAL:return 128+STOP_SIGNAL
            del engine;torch.cuda.empty_cache()
        receipts={}
        for sample_index in range(3):
            sample=directory/f'sample{sample_index:02d}'
            if len(read_progress(sample/'progress.jsonl',selections[sample_index],sample,args.seed+sample_index*100000))!=len(items):
                raise RuntimeError('Some output images failed verification')
            receipts[f'sample{sample_index:02d}']={'manifest_sha256':sha256_file(sample/'manifest.json'),
                                                     'progress_sha256':sha256_file(sample/'progress.jsonl')}
        write_json(directory/'generation_complete.json',{'status':'complete','images':len(items)*3,'targets':len(items),
                   'samples':3,'identity_sha256':run_hash,'sample_receipts':receipts,'text_oracle':False,'image_oracle':False})
        print('[complete] {}'.format(directory),flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print('[error] {}'.format(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
