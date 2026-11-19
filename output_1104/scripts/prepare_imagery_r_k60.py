#!/usr/bin/env python3
"""Create a new K=60 natural R experiment from the audited K=1000 teachers."""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
import yaml
import run_imagery_w010_pilot as pilot
from src.data.scheme1 import file_sha256
from src.utils.imagery_w010_pilot import pair_records, balanced_schedule
from src.utils.staged_queue import atomic_json

OLD = ROOT / 'outputs_new/imagery_w010_subj01_natural_v1'
OUT = ROOT / 'output_1104/imagery_k60_ridge_subj01_v1/r_training'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing protocol differs: ' + str(path))
    else:
        atomic_json(value, path)


def build():
    original = read(OLD / 'protocol/retrieval.json')
    split = read(OLD / 'protocol/split.json')
    groups = [copy.deepcopy(original['queries'][name][:60]) for name in split['targets']]
    if any(len(g) != 60 or len({r['image_id'] for r in g}) != 20 for g in groups):
        raise ValueError('Expected exactly 60 trials and 20 images per query')
    ids = sorted({r['image_id'] for group in groups for r in group})
    # The first deterministic global split with >=2 held-out images per query.
    # It is selected purely for coverage, without using test imagery or metrics.
    n_val = max(2, round(len(ids) * .1))
    for attempt in range(10000):
        rng = np.random.default_rng(42 + attempt)
        val = set(map(int, rng.permutation(ids)[:n_val]))
        if all(2 <= len({r['image_id'] for r in g} & val) <= 18 for g in groups):
            break
    else:
        raise ValueError('Could not find a globally grouped validation split')
    old_ids = sorted({r['teacher_index'] for group in groups for r in group})
    mapping = {old: i for i, old in enumerate(old_ids)}
    teachers = []
    for old in old_ids:
        row = copy.deepcopy(original['teachers'][old])
        if row['kind'] != 'nsd_trial' or row['teacher_index'] != old:
            raise ValueError('Teacher index or kind changed')
        row['original_teacher_index'] = old
        row['teacher_index'] = mapping[old]
        teachers.append(row)
    for group in groups:
        for row in group:
            row['original_teacher_index'] = row['teacher_index']
            row['teacher_index'] = mapping[row['teacher_index']]
            row['teacher_split'] = 'val' if row['image_id'] in val else 'train'
    train_ids = {r['image_id'] for group in groups for r in group if r['teacher_split'] == 'train'}
    if train_ids & val:
        raise AssertionError('Teacher image split overlap')
    schedules = {}
    for portion in ('train', 'val'):
        entries = []
        for label, name in enumerate(split['targets']):
            selected = [r['teacher_index'] for r in groups[label] if r['teacher_split'] == portion]
            entries.append(pair_records(split['rows'][name][portion], selected, 42 + label + (1000 if portion == 'val' else 0)))
        for schedule_attempt in range(512):
            try:
                batches = balanced_schedule(entries, 6, 42 + schedule_attempt + (2000 if portion == 'val' else 0))
                break
            except ValueError:
                continue
        else:
            raise ValueError('Unable to build collision-free balanced ' + portion + ' batches')
        schedules[portion] = {'protocol': 'k60_global_image_grouped_v1', 'portion': portion,
                              'schedule_attempt': schedule_attempt, 'batches': batches}
    retrieval = {'overview': {'protocol': 'natural_trial_top60_from_audited_top1000_v1',
                    'source_retrieval_sha256': file_sha256(OLD / 'protocol/retrieval.json'),
                    'k_unit': 'nsd_fmri_trial', 'k_per_query': 60, 'image_count_per_query': 20,
                    'teacher_image_grouped': True, 'split_seed': 42 + attempt,
                    'split_search_attempt': attempt, 'global_train_val_image_overlap': 0,
                    'target_query': 'original natural stimulus image; target-conditioned training',
                    'counts': [{'target': name, 'train': sum(r['teacher_split']=='train' for r in g),
                                'val': sum(r['teacher_split']=='val' for r in g)}
                               for name,g in zip(split['targets'],groups)]},
                 'queries': dict(zip(split['targets'], groups)), 'teachers': teachers}
    return split, retrieval, schedules, old_ids


def prepare():
    cfg = yaml.safe_load((ROOT / 'configs/imagery_w010_pilot.yaml').read_text())
    source = pilot.run_source(cfg)
    identity = pilot.require_identity(cfg, OLD, source)
    if read(OLD / 'protocol/code_lock.json')['files'] != pilot.code_fingerprint():
        raise ValueError('Previous pilot code lock changed')
    split, retrieval, schedules, old_ids = build()
    cfg = copy.deepcopy(cfg)
    cfg['knn_trials'] = 60
    cfg['experiment_root'] = str(OUT.relative_to(ROOT))
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / 'protocol'; protocol.mkdir(exist_ok=True)
    write_immutable(protocol / 'split.json', split)
    write_immutable(protocol / 'retrieval.json', retrieval)
    for portion, value in schedules.items():
        write_immutable(protocol / ('R_' + portion + '_pairs.json'), value)
    benchmark = read(OLD / 'protocol/gpu_benchmark.json')
    if benchmark['chosen_microbatch'] != 18:
        raise ValueError('Historical microbatch changed')
    write_immutable(protocol / 'gpu_benchmark.json', benchmark)
    write_immutable(protocol / 'config.json', cfg)
    write_immutable(protocol / 'code_lock.json', {'files': pilot.code_fingerprint(),
                    'new_setup_script_sha256': file_sha256(Path(__file__))})
    old_state, old_cache = pilot.validated_teacher_cache(OLD)
    folder = OUT / 'teacher_cache'; folder.mkdir(exist_ok=True)
    cache_path = folder / 'teacher_tokens.npy'; state_path = folder / 'state.json'
    spec = {'protocol': 'k60_subset_of_checked_w010_teacher_cache_v1', 'shape': [len(old_ids), 256, 1664],
            'storage_dtype': 'float16', 'source_cache_sha256': old_state['cache_sha256'],
            'retrieval_sha256': file_sha256(protocol / 'retrieval.json'), 'old_indices': old_ids}
    if state_path.exists():
        state = read(state_path)
        if state['spec'] != spec or not state['complete'] or file_sha256(cache_path) != state['cache_sha256']:
            raise ValueError('Existing K60 cache differs')
    else:
        tmp = cache_path.with_suffix('.tmp.npy')
        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=tuple(spec['shape']))
        for i, old in enumerate(old_ids):
            arr[i] = old_cache[old]
        arr.flush(); del arr
        tmp.replace(cache_path)
        atomic_json({'spec': spec, 'complete': True, 'next_index': len(old_ids),
                     'cache_sha256': file_sha256(cache_path)}, state_path)
    print(json.dumps(retrieval['overview'], ensure_ascii=False, indent=2), flush=True)
    return cfg, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'train'), required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg, source = prepare()
    if args.stage == 'train':
        torch.set_num_threads(4)
        pilot.train_adapter(cfg, OUT, source, torch.device(args.device), 'R')


if __name__ == '__main__':
    main()
