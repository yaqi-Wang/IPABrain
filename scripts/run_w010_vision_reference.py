"""Subj01 w010 vision reference: 6 natural + 6 geometric stimuli, 3 seeds each.

This is a new B0 reference using the existing frozen perception model.  It is
not an evaluation of the imagery V adapter; its inputs are vision fMRI.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import add_project_root
ROOT = add_project_root()
from run_imagery_w010_pilot import (FrozenMapping, atomic_checkpoint, code_fingerprint,
                                    load_frozen, load_trials, require_identity, run_source, sha_config)
from src.data.scheme1 import file_sha256
from src.utils.staged_queue import StageQueue, atomic_json

EXP = ROOT / 'outputs_new/w010_subj01_vision_reference_v1'
DATASET = EXP / 'evaluation/vision'
PILOT = ROOT / 'outputs_new/imagery_w010_subj01_natural_v1'
METRICS = ('PixCorr', 'SSIM', 'Alex2_2way', 'Alex5_2way', 'Incep_2way',
           'CLIP_2way', 'Eff_corr_distance', 'SwAV_corr_distance')


def config_and_source():
    cfg = yaml.safe_load((ROOT / 'configs/imagery_w010_pilot.yaml').read_text())
    if (cfg['subject'], cfg['num_samples'], cfg['num_steps'], cfg['generation_seed'], cfg['rgb_mode']) != (1, 3, 38, 42, 'standard'):
        raise ValueError('Unexpected source pilot settings')
    source = run_source(cfg)
    identity = require_identity(cfg, PILOT, source)
    old_lock = json.loads((PILOT / 'protocol/code_lock.json').read_text())
    if old_lock['files'] != code_fingerprint():
        raise ValueError('Frozen source code no longer matches the completed imagery pilot')
    return cfg, source, identity


def vision_trials(data, split):
    task = data['task_type'].astype(str)
    kind = data['stimulus_type'].astype(str)
    ids = data['stimulus_id'].astype(str)
    natural = list(split['targets'])
    simple = sorted(set(ids[(task == 'vision') & (kind == 'simple')].tolist()))
    if len(natural) != 6 or len(simple) != 6 or len(set(natural + simple)) != 12:
        raise ValueError('Expected exactly six natural and six geometric vision targets')
    records, inputs = [], []
    for target in natural + simple:
        stimulus_type = 'complex' if target in natural else 'simple'
        rows = np.flatnonzero((task == 'vision') & (kind == stimulus_type) & (ids == target))
        if len(rows) != 8 or np.any(data['condition_index'][rows] != 0):
            raise ValueError(f'Invalid raw vision trials for {target}')
        actual_set = set(data['stim_set'][rows].astype(str))
        if actual_set != ({'B'} if stimulus_type == 'complex' else {'A'}):
            raise ValueError(f'Incorrect stimulus set for {target}')
        records.append({'subject_id': 1, 'stimulus_id': target, 'stimulus_type': stimulus_type,
                        'task_type': 'vision', 'n_trials': 8,
                        'source_trial_ids': [int(data['global_trial_idx'][r]) for r in rows],
                        'runs': [int(data['run'][r]) for r in rows]})
        inputs.append(np.asarray(data['fmri_scaled'][rows], dtype=np.float32).mean(axis=0))
    return records, np.stack(inputs).astype(np.float32)


def protocol(cfg, source, identity, records):
    value = {'protocol': 'frozen_w010_vision_reference_v1', 'subject': 1,
             'scope': 'vision fMRI B0; 6 natural and 6 geometric stimuli',
             'per_target_trials': 8, 'target_order': [r['stimulus_id'] for r in records],
             'records': records, 'w010_sha256': identity['w010_sha256'],
             'brainvae_sha256': identity['brainvae_sha256'], 'data_fingerprint': identity['data_fingerprint'],
             'imagery_trials_sha256': file_sha256(source['prepared'] / 'imagery_trials.npz'),
             'imagery_split_sha256': file_sha256(PILOT / 'protocol/split.json'),
             'source_code_sha256': file_sha256(Path(__file__)),
             'pilot_code_lock_sha256': file_sha256(PILOT / 'protocol/code_lock.json'),
             'pilot_config_sha256': sha_config(cfg),
             'generation': {'num_samples': 3, 'num_steps': 38, 'seed': 42, 'rgb_mode': 'standard'},
             'note': 'Vision B0 does not use a trained imagery adapter; six natural targets and their generation seeds match the imagery pilot.'}
    path = EXP / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing vision reference belongs to a different protocol')
    else:
        atomic_json(value, path)
    return value


def predict(cfg, source, identity, device):
    data = load_trials(source)
    split = json.loads((PILOT / 'protocol/split.json').read_text())
    records, inputs = vision_trials(data, split)
    spec = protocol(cfg, source, identity, records)
    DATASET.mkdir(parents=True, exist_ok=True)
    tokens = DATASET / 'tokens/chunk00000.pt'
    old_meta = DATASET / 'metadata.json'
    if old_meta.exists():
        metadata = json.loads(old_meta.read_text())
        if (not tokens.is_file() or file_sha256(tokens) != metadata['token_sha256'] or
                metadata['protocol_sha256'] != file_sha256(EXP / 'protocol.json')):
            raise ValueError('Existing vision prediction is incomplete or changed')
        print('[复用] vision tokens', flush=True)
        return
    if tokens.exists() or any(DATASET.glob('sample*/images/*.png')):
        raise ValueError('Incomplete existing outputs require manual provenance review')
    brain, mapper, _ = load_frozen(cfg, source, device, sdpa=False)
    mapping = FrozenMapping(brain, mapper)
    x = torch.from_numpy(inputs).to(device)
    with torch.inference_mode():
        output = torch.cat([mapping(chunk) for chunk in x.split(2)]).float().cpu()
    if tuple(output.shape) != (12, 256, 1664) or not bool(torch.isfinite(output).all()):
        raise FloatingPointError('Invalid vision tokens')
    tokens.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint({'mean': output, 'records': records, 'representation': 'mapped_raw_clip_tokens',
                       'mapping_checkpoint_sha256': identity['w010_sha256']}, tokens)
    metadata = {'protocol': spec['protocol'], 'subject': 1, 'dataset': 'vision', 'method': 'B0',
                'checkpoint_sha256': identity['w010_sha256'], 'w010_sha256': identity['w010_sha256'],
                'brainvae_sha256': identity['brainvae_sha256'], 'data_fingerprint': identity['data_fingerprint'],
                'aggregation': 'mean_fmri', 'num_samples': 3, 'num_steps': 38, 'seed': 42,
                'rgb_mode': 'standard', 'n_items': 12, 'partial_test': False, 'debug_training': False,
                'input': 'mean of all eight raw vision fMRI trials per stimulus',
                'target_usage': 'reference images are attached only after prediction for evaluation',
                'token_sha256': file_sha256(tokens), 'protocol_sha256': file_sha256(EXP / 'protocol.json')}
    atomic_json(metadata, old_meta)
    del mapping, brain, mapper, x, output
    gc.collect()
    torch.cuda.empty_cache()
    print('[预测完成] vision: 12 targets × 8 vision trials', flush=True)


def generation_ok():
    path = DATASET / 'metadata.json'
    if not path.is_file() or not (EXP / 'protocol.json').is_file(): return False
    meta = json.loads(path.read_text());spec = json.loads((EXP / 'protocol.json').read_text())
    tokens = DATASET / 'tokens/chunk00000.pt'
    if not tokens.is_file() or file_sha256(tokens) != meta['token_sha256']: return False
    for sample in range(3):
        directory = DATASET / f'sample{sample:02d}'
        manifest = directory / 'manifest.json'
        if not manifest.is_file(): return False
        value = json.loads(manifest.read_text());items = value['items']
        if len(items) != 12 or [r['stimulus_id'] for r in items] != spec['target_order']: return False
        for i, row in enumerate(items):
            if row['seed'] != 42 + sample * 100000 + i or row['n_trials'] != 8: return False
            if not (directory / row['path']).is_file() or not Path(row.get('target_image', '')).is_file(): return False
    return True


def generate(cfg, source, device):
    from torchvision.utils import save_image
    from src.utils.clip_mapping import load_mapping_config
    from decode_brainvae_scheme1 import generate_samples
    from decode_synthesised_fmri_unclip import load_sdxl_unclip, unclip_decode_batch
    spec = json.loads((EXP / 'protocol.json').read_text())
    metadata = json.loads((DATASET / 'metadata.json').read_text())
    if metadata['protocol_sha256'] != file_sha256(EXP / 'protocol.json'): raise ValueError('Protocol changed')
    if file_sha256(DATASET / 'tokens/chunk00000.pt') != metadata['token_sha256']: raise ValueError('Tokens changed')
    _, _, paths = load_mapping_config(source['config_path'], ROOT)
    target_root = paths.raw_nsd_root / 'nsddata/experiments/nsdimagery/rawtargetimages'
    targets = {r['stimulus_id']: target_root / ('setB' if r['stimulus_type'] == 'complex' else 'setA') /
               (r['stimulus_id'] + '.png') for r in spec['records']}
    if not all(path.is_file() for path in targets.values()): raise FileNotFoundError('Missing vision reference image')
    unclip = Path(source['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute(): unclip = ROOT / unclip
    engine, suffix, offset = load_sdxl_unclip(paths.generative_models_path, unclip, paths.openclip_version, device, 38)
    generate_samples([DATASET / 'tokens/chunk00000.pt'], DATASET, metadata, engine, suffix, offset,
                     partial(unclip_decode_batch, rgb_mode='standard'), save_image, 3, 42, targets, True)
    if not generation_ok(): raise RuntimeError('Generation is incomplete')
    atomic_json({'status': 'complete', 'n_targets': 12, 'n_samples': 3, 'n_images': 36}, EXP / 'generation_complete.json')
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(device):
    if not generation_ok(): raise ValueError('Complete generation is required before evaluation')
    command = [sys.executable, '-u', 'scripts/eval_scheme1_reconstruction.py', '--dataset-dir', str(DATASET),
               '--models', 'Alex2,Alex5,Incep,CLIP,Eff,SwAV', '--device', str(device), '--batch-size', '8']
    subprocess.run(command, cwd=ROOT, check=True)
    source = DATASET / 'image_metrics.json'
    values = json.loads(source.read_text())
    if values['n_targets'] != 12 or values['n_samples'] != 3: raise ValueError('Incorrect evaluation scope')
    for key in METRICS:
        if not math.isfinite(values['metrics'][key]) or not math.isfinite(values['metrics']['complex_' + key]):
            raise ValueError('Invalid evaluation metric ' + key)
    common = {'subject': 1, 'dataset': 'vision', 'method': 'w010_B0', 'samples_per_target': 3,
              'vision_trials_per_target': 8, 'source_metrics_sha256': file_sha256(source),
              'note': 'Perception vision input; no imagery adapter; not evaluation of method V.'}
    atomic_json({**common, 'targets': 6, 'stimulus_type': 'complex',
                 'metrics': {key: values['metrics']['complex_' + key] for key in METRICS}},
                EXP / 'metrics_vision_natural6.json')
    atomic_json({**common, 'targets': 12, 'stimulus_type': 'complex_and_simple',
                 'metrics': {key: values['metrics'][key] for key in METRICS}}, EXP / 'metrics_vision_all12.json')
    print('[评估完成] vision natural6 and all12', flush=True)


def verify(stage):
    if stage == 'predict':
        p = DATASET / 'metadata.json';t = DATASET / 'tokens/chunk00000.pt'
        return p.is_file() and t.is_file() and file_sha256(t) == json.loads(p.read_text())['token_sha256']
    if stage == 'generate': return (EXP / 'generation_complete.json').is_file() and generation_ok()
    if stage == 'evaluate':
        for name, count in (('metrics_vision_natural6.json', 6), ('metrics_vision_all12.json', 12)):
            p = EXP / name
            if not p.is_file(): return False
            value = json.loads(p.read_text())
            if value['targets'] != count or not all(math.isfinite(value['metrics'][k]) for k in METRICS): return False
        return True
    raise ValueError(stage)


def queue():
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / 'logs').mkdir(exist_ok=True)
    q = StageQueue(EXP, 120)
    active = {'process': None}
    def stop(signum, frame):
        proc = active['process']
        if proc is not None and proc.poll() is None: proc.send_signal(signum)
        q.state(status='interrupted', signal=signum)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def run(stage):
        log = EXP / 'logs' / f'{stage}.log'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', stage]
        q.state(stage=stage, log=str(log), command=cmd)
        with log.open('a') as f:
            f.write('\nCOMMAND ' + json.dumps(cmd) + '\n');f.flush()
            active['process'] = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            result = active['process'].wait();active['process'] = None
        if result:
            raise RuntimeError(f'{stage} failed ({result}): ' + '\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
    q.state(status='running', failures=[], error=None, signal=None)
    try:
        for stage in ('predict', 'generate', 'evaluate'):
            q.run(stage, lambda s=stage: run(s), lambda s=stage: verify(s), rest=stage != 'evaluate')
        q.state(status='complete', stage=None, cooldown_remaining_seconds=0, resume_at=None)
    except BaseException as exc:
        if not isinstance(exc, SystemExit): q.state(status='failed', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('run', 'predict', 'generate', 'evaluate'), default='run')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'run':
        queue();return
    cfg, source, identity = config_and_source()
    if args.stage in ('generate', 'evaluate'):
        records, _ = vision_trials(load_trials(source), json.loads((PILOT / 'protocol/split.json').read_text()))
        protocol(cfg, source, identity, records)
    device = torch.device('cuda')
    if args.stage == 'predict': predict(cfg, source, identity, device)
    elif args.stage == 'generate': generate(cfg, source, device)
    else: evaluate(device)


if __name__ == '__main__':
    main()
