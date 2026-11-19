#!/usr/bin/env python3
"""Evaluate matched sample00 B0/R/V inputs and their Ridge-text refinements."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'output_1104/scripts'))

import refine_multisubject_imagery as refiner
import refine_w010_mindeye2_img2img as base
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import atomic_json, mapping_lock
from multisubject_resume import archive_file


RUN = ROOT / 'output_1104/imagery_ridge_refine_subj01_v1'
OUT = RUN / 'refinement'
BASELINE = RUN / 'baseline_sample00'
METHODS = ('B0', 'R', 'V', 'R_MSE')
KEYS = ('PixCorr', 'SSIM', 'Alex2_2way', 'Alex5_2way', 'Incep_2way', 'CLIP_2way',
        'Eff_corr_distance', 'SwAV_corr_distance')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ensure_value(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Existing evaluation setup has changed: {}'.format(path))
    if not path.exists():
        atomic_json(value, path)


def baseline_manifest(catalog, identity, method):
    folder = BASELINE / method / 'imagery'
    folder.mkdir(parents=True, exist_ok=True)
    ref_meta = read(OUT / method / 'imagery/metadata.json')
    metadata = {'protocol': 'imagery_existing_w010_sample00_matched_v1', 'subject': identity['subject'],
                'dataset': 'imagery', 'method': method, 'aggregation': 'mean_fmri',
                'n_items': 12, 'num_samples': 3, 'partial_test': False, 'debug_training': False,
                'checkpoint_sha256': identity['baseline_sources'][method]['checkpoint_sha256'],
                'data_fingerprint': identity['data_fingerprint'], 'target_usage': 'evaluation_only',
                'generator_backend': 'existing_w010_unclip', 'text_condition': 'none',
                'text_oracle': False, 'oracle': False,
                'matched_refinement_identity_sha256': ref_meta['source_identity_sha256']}
    ensure_value(folder / 'metadata.json', metadata)
    for sample_index in range(3):
        sample=folder/f'sample{sample_index:02d}';sample.mkdir(parents=True,exist_ok=True)
        items=[{**{k:r[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')},
                'item_index':r['index'],'sample_index':sample_index,'path':r['input_path']} for r in catalog[method][sample_index]]
        manifest=dict(metadata,level='stimulus_level',sample_index=sample_index,n_images=12,items=items)
        ensure_value(sample/'manifest.json',manifest)
    return folder


def evaluate(folder, label, target_hashes, image_hashes, device):
    metric = folder / 'image_metrics.json'
    receipt = folder / 'evaluation_complete.json'
    identity = {'metadata_sha256': file_sha256(folder / 'metadata.json'),
                'manifest_sha256': [file_sha256(folder/f'sample{i:02d}/manifest.json') for i in range(3)],
                'target_image_sha256': target_hashes,
                'evaluated_image_sha256': image_hashes,
                'evaluator_sha256': file_sha256(ROOT / 'scripts/eval_scheme1_reconstruction.py')}
    if receipt.exists():
        old = read(receipt)
        if old.get('identity') != identity or not metric.is_file() or old['metrics_sha256'] != file_sha256(metric):
            raise ValueError('Existing image evaluation has changed: {}'.format(label))
        return read(metric)
    if metric.exists():
        archived=archive_file(metric)
        print('[resume] preserved unverified metrics at {}'.format(archived),flush=True)
    command = [sys.executable, '-u', str(ROOT / 'scripts/eval_scheme1_reconstruction.py'),
               '--dataset-dir', str(folder), '--models', 'Alex2,Alex5,Incep,CLIP,Eff,SwAV',
               '--device', device, '--batch-size', '8']
    logs = RUN / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / ('eval_' + label + '.log')).open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND ' + json.dumps(command) + '\n')
        handle.flush()
        subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, check=True)
    result = read(metric)
    if result.get('n_targets') != 12 or result.get('n_samples') != 3 or len(result.get('per_seed',[]))!=3:
        raise ValueError('Unexpected evaluation scope for {}'.format(label))
    for prefix in ('complex_', 'simple_'):
        if not all(math.isfinite(result['metrics'][prefix + key]) and all(math.isfinite(row[prefix+key]) for row in result['per_seed']) for key in KEYS):
            raise ValueError('Nonfinite or missing metrics for {}'.format(label))
    atomic_json({'status': 'complete', 'identity': identity, 'metrics_sha256': file_sha256(metric)}, receipt)
    print('[evaluation] {} complete'.format(label), flush=True)
    return result


def main():
    global RUN, OUT, BASELINE, METHODS
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,choices=range(1,9),required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prediction-dir', type=Path)
    p.add_argument('--b0-only',action='store_true',help='evaluate B0 in its separate output directory')
    args = p.parse_args()
    root=ROOT/f'output_1104/multisubject_nsd_imagery_v1/subj{args.subject:02d}'
    RUN=root/'imagery_refine_k60_3samples_v1'
    OUT=RUN/'refinement'
    BASELINE=RUN/'baseline_sample00'
    if args.subject==1:
        refiner.MU=ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs'
        refiner.NATURAL=root/'imagery_initial_k60_3samples_v1/initial';refiner.GEOMETRY=None
        args.prediction_dir=args.prediction_dir or ROOT/'output_1104/nsd_subj01/ridge_three_text/predictions/imagery_subj01_12stim_v1'
    else:
        refiner.MU=root/'imagery_ridge_inputs_3samples_v1'
        refiner.NATURAL=root/'imagery_natural_k60_3samples_v1/runs'
        refiner.GEOMETRY=root/'imagery_geometry_transfer_k60_3samples_v1/runs'
        args.prediction_dir=args.prediction_dir or root/'ridge/predictions/imagery_12stim_3samples_v1'
    if args.b0_only:
        METHODS=('B0',);refiner.METHODS=METHODS
        RUN=root/'imagery_b0_refine_3samples_v1';OUT=RUN/'refinement';BASELINE=RUN/'baseline_sample00'
        refiner.NATURAL=root/'imagery_b0_initial_3samples_v1/initial';refiner.GEOMETRY=None
    catalog, _arrays, identity = refiner.checked_inputs(refiner.MU, args.prediction_dir.resolve())
    if identity['subject']!=args.subject:raise ValueError('Subject mismatch')
    overall = read(OUT / 'generation_complete.json')
    if overall.get('status') != 'complete' or overall.get('images') != 36*len(METHODS):
        raise ValueError('All requested refinements must be completed before evaluation')
    refined_hashes = {}
    for method in METHODS:
        folder = OUT / method / 'imagery'
        completed = read(folder / 'generation_complete.json')
        saved_identity = read(folder / 'metadata.json')['source_identity']
        for key in ('mu_sha256', 'ridge_prediction_sha256', 'baseline_images', 'baseline_sources'):
            if saved_identity.get(key) != identity[key]:
                raise ValueError('Current prediction or baseline images differ from refinement inputs: ' + key)
        if (completed.get('status') != 'complete' or completed.get('source_identity_sha256') != overall.get('source_identity_sha256')):
            raise ValueError('Refinement completion audit failed for ' + method)
        refined_hashes[method]=[]
        for sample_index in range(3):
            sample=folder/f'sample{sample_index:02d}'
            done=refiner.progress(sample/'progress.jsonl',catalog[method][sample_index],sample,42+sample_index*100000)
            receipt=completed['sample_receipts'][f'sample{sample_index:02d}']
            if len(done)!=12 or receipt['manifest_sha256']!=file_sha256(sample/'manifest.json') or receipt['progress_sha256']!=file_sha256(sample/'progress.jsonl'):
                raise ValueError('Sample completion audit failed')
            refined_hashes[method].append({catalog[method][sample_index][i]['stimulus_id']:done[i]['image_sha256'] for i in range(12)})
    # Only after generation, read target-derived reference metadata and image bytes for evaluation identity.
    target_hashes = {row['stimulus_id']:file_sha256(Path(row['target_image'])) for row in catalog['B0'][0]}
    target_paths = {row['stimulus_id']:row['target_image'] for row in catalog['B0'][0]}
    for method in METHODS:
        for sample in catalog[method]:
            for row in sample:
                if row['target_image'] != target_paths[row['stimulus_id']]:raise ValueError('Target path differs across methods')
    with mapping_lock(RUN / 'evaluation'):
        reports = []
        for method in METHODS:
            baseline = baseline_manifest(catalog, identity, method)
            refined = OUT / method / 'imagery'
            input_hashes = [{x['stimulus_id']:x['input_sha256'] for x in sample} for sample in catalog[method]]
            old = evaluate(baseline, 'baseline_' + method, target_hashes, input_hashes, args.device)
            new = evaluate(refined, 'refined_' + method, target_hashes, refined_hashes[method], args.device)
            for group, prefix in (('natural', 'complex_'), ('geometry', 'simple_')):
                baseline_metrics = {key: float(old['metrics'][prefix + key]) for key in KEYS}
                refined_metrics = {key: float(new['metrics'][prefix + key]) for key in KEYS}
                reports.append({'group': group, 'method': method,
                                'baseline': baseline_metrics, 'refined': refined_metrics,
                                'baseline_per_seed':[{key:float(seed[prefix+key]) for key in KEYS} for seed in old['per_seed']],
                                'refined_per_seed':[{key:float(seed[prefix+key]) for key in KEYS} for seed in new['per_seed']],
                                'delta_refined_minus_baseline': {k: refined_metrics[k] - baseline_metrics[k]
                                                                  for k in KEYS}})
        summary = {'status': 'complete', 'subject': args.subject, 'stimuli_per_group': 6, 'samples_per_stimulus': 3,
                   'comparison': 'matched three w010 samples vs NSD-Ridge text-conditioned refinements',
                   'text_condition': 'one prediction per stimulus from raw held-out imagery μ, shared across requested methods',
                   'text_oracle': False, 'target_usage': 'image evaluation only',
                   'caveat': 'Single subject, six stimuli per group, three diffusion samples; group 2-way metrics use 6 candidates. Seeds are not independent subjects.' + ('' if args.b0_only else ' Geometry adapters are natural-adapter transfer.'),
                   'rows': reports}
        atomic_json(summary, RUN / 'image_metrics_summary.json')
        csv_path = RUN / 'image_metrics_summary.csv'
        with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
            fieldnames = ['group', 'method', 'stage'] + list(KEYS)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in reports:
                for stage in ('baseline', 'refined', 'delta_refined_minus_baseline'):
                    writer.writerow(dict(group=row['group'], method=row['method'], stage=stage, **row[stage]))
                for sample in range(3):
                    for stage in ('baseline','refined'):
                        writer.writerow(dict(group=row['group'],method=row['method'],stage=f'{stage}_sample{sample:02d}',
                                             **row[stage+'_per_seed'][sample]))
        lines = [f'# subj{args.subject:02d} imagery：NSD Ridge 文本条件精修', '', summary['caveat'], '',
                 '| 类型 | 方法 | 阶段 | ' + ' | '.join(KEYS) + ' |',
                 '| --- | --- | --- | ' + ' | '.join(['---:'] * len(KEYS)) + ' |']
        for row in reports:
            for stage in ('baseline', 'refined', 'delta_refined_minus_baseline'):
                lines.append('| {} | {} | {} | {} |'.format(row['group'], row['method'], stage,
                              ' | '.join('{:.3f}'.format(row[stage][key]) for key in KEYS)))
            for sample in range(3):
                for stage in ('baseline','refined'):
                    lines.append('| {} | {} | {}_sample{:02d} | {} |'.format(row['group'],row['method'],stage,sample,
                                  ' | '.join('{:.3f}'.format(row[stage+'_per_seed'][sample][key]) for key in KEYS)))
        (RUN / 'image_metrics_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print('[complete] {}'.format(RUN / 'image_metrics_summary.md'), flush=True)


if __name__ == '__main__':
    main()
