"""Subject-specific frozen BrainVAE -> AdaToken -> w010 -> NSD reconstruction."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from _bootstrap import add_project_root
ROOT = add_project_root()
from src.data.scheme1 import (check_source_files, data_directory, digest_json, file_sha256, load_prepared,
                             load_scheme1_config, resolve_path)
from src.utils.clip_mapping import (load_brain_source, load_mapping_config,
                                   verified_completed_mean_cache)
from src.utils.scheme1_runs import training_directory
from src.utils.adatoken_semantic_selection import METRICS, POLICY, read_candidate
from src.utils.staged_queue import StageQueue, atomic_json

ACTIVE = None


def equal_relocated_source(first, second):
    """Location may change; encoder hash, geometry, data and normalization may not."""
    return ({k: v for k, v in first.items() if k != 'brainvae_checkpoint'} ==
            {k: v for k, v in second.items() if k != 'brainvae_checkpoint'})


def choose_w010(baseline, candidates):
    if not candidates:
        raise ValueError('No w010 validation candidates')
    for row in candidates:
        if row['identity'] != baseline['identity'] or row['image_ids'] != baseline['image_ids']:
            raise ValueError('Validation candidates have different subjects, encoders or image IDs')
    ordered = sorted(candidates, key=lambda row: row['label'])
    winner = max(ordered, key=lambda row: tuple(row['metrics'][name] for name in POLICY['ranking']))
    return {'protocol': 'subject_specific_w010_internal_generation_selection_v1',
            'selection_dataset': 'nsd_val', 'test_used_for_selection': False,
            'ranking': POLICY['ranking'], 'candidate_family': 'w010_epochs_1_to_4',
            'baseline_reference': baseline, 'candidates': ordered, 'selected': winner,
            'baseline_is_a_reference_not_a_w010_candidate': True}


def load_suite(args):
    cfg = yaml.safe_load(resolve_path(args.config, ROOT).read_text())
    for key in ('subjects', 'test_max_items', 'cooldown_scope', 'experiment_root'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if (not cfg['subjects'] or len(set(cfg['subjects'])) != len(cfg['subjects'])
            or not set(cfg['subjects']).issubset(set(range(1, 9)))):
        raise ValueError('Subjects must be unique IDs from 1 through 8')
    if cfg['cooldown_seconds'] <= 0 or cfg['cooldown_scope'] not in {'stage', 'subject'}:
        raise ValueError('The production queue requires a positive rest duration, per stage or per subject')
    if cfg.get('selection_policy', 'internal_generation') not in {'fixed_epoch004', 'internal_generation'}:
        raise ValueError('Unknown checkpoint selection policy')
    if cfg['test_max_items'] < 0 or cfg['validation_images'] != 100:
        raise ValueError('Use nonnegative test_max_items and 100 internal validation images')
    if (cfg['seed'], cfg['num_steps'], cfg['num_samples'], cfg['rgb_mode'], cfg['device']) != (42, 38, 1, 'standard', 'cuda'):
        raise ValueError('This queue fixes seed42, 38 steps, one standard-RGB sample and CUDA')
    return cfg, resolve_path(cfg['experiment_root'], ROOT)


def write_yaml(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != value:
        raise ValueError(f'Configuration already exists with different settings: {path}')
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def initialize(cfg, experiment):
    experiment.mkdir(parents=True, exist_ok=True)
    plan_path = experiment / 'plan.json'
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if saved['settings'] != cfg:
            ledger = experiment / 'stage_ledger.json'
            if ledger.exists() and json.loads(ledger.read_text()).get('stages'):
                raise ValueError('Execution already started with different settings; use a new experiment_root')
        else:
            for name, digest in saved['config_sha256'].items():
                if file_sha256(Path(name)) != digest:
                    raise ValueError(f'Generated config changed: {name}')
            return saved
    configs, subjects = {}, {}
    template = yaml.safe_load((ROOT / 'configs/clip_mapping_ppbe_local_adatoken.yaml').read_text())
    semantic = yaml.safe_load((ROOT / 'configs/clip_mapping_adatoken_semantic_w010.yaml').read_text())
    for subject in cfg['subjects']:
        folder = experiment / f'subj{subject:02d}'
        generated = folder / 'configs'
        generated.mkdir(parents=True, exist_ok=True)
        if subject == 1:
            brain_config = ROOT / 'configs/brainvae_scheme1_ppbe.yaml'
            brain_weight = ROOT / 'outputs_new/baselines/brainvae_ppbe_local/subj01/checkpoints/best.pt'
            source_kind = 'local_ppbe'
        elif subject in range(3, 8):
            brain_config = ROOT / f'outputs_new/ppbe_multisubject_v1/configs/subj{subject:02d}.yaml'
            brain_weight = ROOT / f'outputs_new/ppbe_multisubject_v1/subj{subject:02d}/runs/brainvae_ppbe_local_fast_v1/checkpoints/best.pt'
            source_kind = 'local_ppbe'
        else:
            raw = yaml.safe_load((ROOT / 'configs/brainvae_ppbe_multisubject_fast.yaml').read_text())
            raw.update(output_root=str(experiment / 'stage1'), log_root=str(experiment / 'logs/stage1'))
            brain_config = generated / 'brainvae.yaml'
            write_yaml(raw, brain_config)
            configs[str(brain_config)] = file_sha256(brain_config)
            if subject == 2:
                brain_weight = ROOT / 'checkpoints/ppbe_author/subj02/brainvae.pth'
                source_kind = 'ppbe_author'
            else:
                brain_cfg, _ = load_scheme1_config(brain_config, ROOT)
                brain_weight = training_directory(brain_cfg, ROOT, subject) / 'checkpoints/best.pt'
                source_kind = 'local_ppbe_to_train'
                if not cfg['train_missing_subj08'] and not brain_weight.is_file():
                    raise FileNotFoundError('Subject 08 needs a matching PPBE Stage1; the legacy project VAE is a different model')
        common = {'brainvae_config': str(brain_config), 'brainvae_checkpoint': str(brain_weight),
                  'output_root': str(experiment), 'log_root': str(experiment / 'logs'),
                  'mean_cache_directory': str(folder / 'features')}
        if subject == 2:
            common.update(brainvae_source='ppbe_external', external_source={
                'subject': subject, 'author_model_config': '',
                'origin': 'user_provided_PPBE_Stage1',
                'training_data_provenance': 'author_checkpoint_not_fully_audited'})
        base = copy.deepcopy(template)
        base.pop('mean_cache_run', None)
        base.update(common, run_name='adatoken_base_v1')
        base['training'].update(num_workers=2, persistent_workers=True, prefetch_factor=2)
        base['training']['encode_batch_size'] = 6
        base['evaluation'].update(num_samples=1, num_steps=38, seed=42, rgb_mode='standard')
        if subject != 1 and cfg.get('fast_head_training'):
            base['model']['gradient_checkpointing'] = False
            base['training'].update(micro_batch_size=12, fused_adamw=True)
            base['run_name'] = 'adatoken_base_fast_v2'
        base_path = generated / 'adatoken.yaml'
        write_yaml(base, base_path)
        refined = copy.deepcopy(semantic)
        refined.update(common, run_name='adatoken_w010_v1')
        refined['training'].update(num_workers=2, persistent_workers=True, prefetch_factor=2,
            initialize_from=str(folder / 'runs/adatoken_base_v1/checkpoints/best.pt'))
        refined['training']['encode_batch_size'] = 6
        if subject != 1 and cfg.get('fast_head_training'):
            refined['model']['gradient_checkpointing'] = False
            refined['training'].update(micro_batch_size=12, fused_adamw=True,
                initialize_from=str(folder / 'runs/adatoken_base_fast_v2/checkpoints/best.pt'))
            refined['run_name'] = 'adatoken_w010_fast_v2'
        refined_path = generated / 'w010.yaml'
        write_yaml(refined, refined_path)
        for path in (base_path, refined_path):
            configs[str(path)] = file_sha256(path)
        subjects[str(subject)] = {'subject': subject, 'source_kind': source_kind,
            'brain_config': str(brain_config), 'brain_checkpoint': str(brain_weight),
            'base_config': str(base_path), 'w010_config': str(refined_path),
            'reuse_head': subject == 1 and cfg['reuse_subj01_head']}
    plan = {'protocol': 'w010_multisubject_pipeline_v1', 'settings': cfg,
            'subjects': subjects, 'config_sha256': configs,
            'selection': 'Fixed epoch004, no validation image generation.' if cfg.get('selection_policy') == 'fixed_epoch004' else 'Highest internal-val100 CLIP among w010 epochs 1..4; Inception and SSIM break ties. Original AdaToken is a reference.',
            'storage': 'All new artifacts accessible below outputs_new; completed resumable head optimizer states and reproducible source-mean banks are removed after final evaluation.'}
    atomic_json(plan, plan_path)
    return plan


def preflight(cfg, experiment, plan, gpu=False):
    rows = []
    for subject in cfg['subjects']:
        spec = plan['subjects'][str(subject)]
        brain_cfg, paths = load_scheme1_config(spec['brain_config'], ROOT)
        directory = paths.nsd_root / f'subj{subject:02d}'
        ids = {}
        row = {'subject': subject, 'source_kind': spec['source_kind']}
        for split in ('train', 'test'):
            ids[split] = np.load(directory / f'nsd_{split}_image_id_sub{subject}.npy')
            fmri = np.load(directory / f'nsd_{split}_fmri_scale_sub{subject}.npy', mmap_mode='r')
            tokens = np.load(directory / f'nsd_sdxl_clip_{split}_sub{subject}.npy', mmap_mode='r')
            if (fmri.ndim != 3 or len(ids[split]) != len(fmri) or len(np.unique(ids[split])) != len(ids[split])
                    or tokens.shape != (len(fmri), 256, 1664)):
                raise ValueError(f'Invalid source arrays: subject {subject}/{split}')
            row[split + '_images'] = len(ids[split])
        if np.intersect1d(ids['train'], ids['test']).size:
            raise ValueError(f'Training/test image overlap: subject {subject}')
        row['test_targets'] = min(cfg['test_max_items'], row['test_images']) if cfg['test_max_items'] else row['test_images']
        weight = Path(spec['brain_checkpoint'])
        if not weight.is_file() and subject != 8:
            raise FileNotFoundError(weight)
        if subject != 8 or weight.is_file():
            row['brainvae_sha256'] = file_sha256(weight)
        frozen_record = experiment / f'subj{subject:02d}/frozen_source.json'
        if frozen_record.is_file() and json.loads(frozen_record.read_text())['brainvae_checkpoint_sha256'] != row.get('brainvae_sha256'):
            raise ValueError(f'Frozen BrainVAE changed after the pipeline started: subject {subject}')
        prepared = data_directory(brain_cfg, ROOT, subject)
        if (prepared / 'manifest.json').is_file():
            manifest, _, _ = load_prepared(prepared)
            check_source_files(paths.nsd_root, subject, manifest,
                               ('train_fmri', 'train_ids', 'train_clip', 'test_fmri', 'test_ids', 'test_clip'))
        for path in (paths.openclip_version, Path(brain_cfg['evaluation']['unclip_checkpoint'])):
            if path is None or not path.is_file():
                raise FileNotFoundError(f'Missing generation checkpoint: {path}')
        rows.append(row)
        print('[核验] ' + json.dumps(row), flush=True)
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use the reconstruction environment on the GPU host.')
    report = {'subjects': rows, 'cuda_checked': gpu, 'cooldown_seconds': cfg['cooldown_seconds'],
              'cooldown_scope': cfg['cooldown_scope'], 'plan_sha256': file_sha256(experiment / 'plan.json')}
    atomic_json(report, experiment / 'data_audit.json')
    return report


def child(experiment, subject, phase, arguments):
    global ACTIVE
    log = experiment / 'logs' / f'subj{subject:02d}' / f'{phase}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    state_path = experiment / 'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    atomic_json({**state, 'subject': subject, 'log': str(log), 'command': arguments}, state_path)
    with log.open('a') as handle:
        handle.write('\nCOMMAND ' + json.dumps([sys.executable, '-u', *arguments]) + '\n')
        handle.flush()
        ACTIVE = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT,
                                  stdout=handle, stderr=subprocess.STDOUT)
        code = ACTIVE.wait()
        ACTIVE = None
    if code:
        raise RuntimeError(f'{phase} exit={code}; log={log}\n' + '\n'.join(log.read_text(errors='replace').splitlines()[-15:]))


def prepared_ok(spec, subject):
    config, _ = load_scheme1_config(spec['brain_config'], ROOT)
    directory = data_directory(config, ROOT, subject)
    if not (directory / 'manifest.json').is_file():
        return False
    manifest, _, _ = load_prepared(directory)
    return manifest['subject'] == subject and manifest.get('source_sample_check', {}).get('status') == 'verified'


def training_ok(config_path, subject):
    config, _, _ = load_mapping_config(config_path, ROOT)
    run = training_directory(config, ROOT, subject)
    done, best, protocol = run / 'training_complete.json', run / 'checkpoints/best.pt', run / 'protocol.json'
    if not (done.is_file() and best.is_file() and protocol.is_file()):
        return False
    saved = json.loads(protocol.read_text())
    if saved['subject'] != subject or saved.get('debug_training') or saved.get('test_used_for_selection'):
        raise ValueError(f'Not a formal subject-matched head: {run}')
    if saved['config'] != config:
        raise ValueError(f'Completed head has a different configuration: {run}')
    return True


def head_train(experiment, subject, phase, path):
    config, _, _ = load_mapping_config(path, ROOT)
    run = training_directory(config, ROOT, subject)
    args = ['scripts/train_clip_mapping.py', '--config', str(path), '--subject', str(subject), '--device', 'cuda']
    if (run / 'checkpoints/last.pt').exists():
        args.append('--resume')
    child(experiment, subject, phase, args)


def evaluation_paths(config_path, checkpoint, dataset, label, maximum):
    run = Path(checkpoint).parent.parent
    tag = f'{label}_{dataset}_{maximum if maximum else "all"}_s42'
    return run / 'evaluation' / tag / dataset, tag


def decode_command(subject, config_path, checkpoint, dataset, tag, maximum, subset=None):
    command = ['scripts/decode_clip_mapping.py', '--config', str(config_path), '--checkpoint', str(checkpoint),
        '--subject', str(subject), '--device', 'cuda', '--dataset', dataset, '--max-items', str(maximum),
        '--num-samples', '1', '--num-steps', '38', '--seed', '42', '--rgb-mode', 'standard',
        '--eval-tag', tag, '--token-source-tag', tag, '--resume']
    if subset:
        command += ['--validation-subset', str(subset)]
    return command


def images_ok(directory, subject, dataset, ids, expected_checkpoint_sha=None):
    metadata_path, manifest_path = directory / 'metadata.json', directory / 'sample00/manifest.json'
    if not metadata_path.is_file() or not manifest_path.is_file():
        return False
    metadata, manifest = json.loads(metadata_path.read_text()), json.loads(manifest_path.read_text())
    if expected_checkpoint_sha is not None and metadata.get('checkpoint_sha256') != expected_checkpoint_sha:
        raise ValueError(f'Images refer to a different checkpoint: {directory}')
    if any(manifest.get(key) != metadata.get(key) for key in ('checkpoint_sha256', 'data_fingerprint', 'dataset', 'num_samples')):
        raise ValueError(f'Manifest and generation metadata disagree: {directory}')
    if (metadata.get('subject') != subject or metadata.get('dataset') != dataset or metadata.get('debug_training')
            or metadata.get('num_samples') != 1 or metadata.get('num_steps') != 38 or metadata.get('seed') != 42
            or metadata.get('rgb_mode') != 'standard' or metadata.get('aggregation') != 'mean_fmri'):
        raise ValueError(f'Inconsistent generation protocol: {directory}')
    items = manifest['items']
    if [row['image_id'] for row in items] != list(map(int, ids)):
        return False
    for index, row in enumerate(items):
        path = directory / 'sample00' / row['path']
        if row['seed'] != 42 + index or row['task_type'] != dataset or not path.is_file() or not path.stat().st_size:
            return False
    return True


def metrics_ok(directory, subject, count):
    path = directory / 'image_metrics.json'
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    return (record.get('subject') == subject and record.get('n_targets') == count
            and record.get('dataset') == directory.name
            and record.get('n_samples') == 1 and record.get('debug_training') is False
            and all(name in record['metrics'] and math.isfinite(record['metrics'][name]) for name in METRICS))


def evaluate(experiment, subject, label, directory):
    child(experiment, subject, label, ['scripts/eval_scheme1_reconstruction.py', '--dataset-dir', str(directory),
        '--models', 'Alex2,Alex5,Incep,CLIP,Eff,SwAV', '--batch-size', '16', '--device', 'cuda'])


def export_subj01(spec, experiment):
    """Preserve model tensors; give relocated provenance a new, explicit export."""
    parent = ROOT / 'outputs_new/baselines/w010/subj01/checkpoints/epoch004.pt'
    evidence_path = ROOT / 'outputs_new/retained/adatoken_refinement/semantic_v2/selections/all.json'
    evidence = json.loads(evidence_path.read_text())
    if evidence['selected']['checkpoint_sha256'] != file_sha256(parent):
        raise ValueError('Existing subj01 w010 is not the recorded validation-selected checkpoint')
    destination = experiment / 'subj01/runs/w010_reused/checkpoints/selected.pt'
    selection_path = experiment / 'subj01/selection.json'
    if selection_path.is_file():
        return json.loads(selection_path.read_text())
    config, brain, _ = load_mapping_config(spec['w010_config'], ROOT)
    source = load_brain_source(config, brain, ROOT, 1)
    saved = torch.load(parent, map_location='cpu', weights_only=False, mmap=True)
    if not equal_relocated_source(saved['protocol_metadata']['source_identity'], source['identity']):
        raise ValueError('Relocation changed more than the BrainVAE path')
    if saved['config'] != config['model']:
        raise ValueError('Reused w010 model geometry differs')
    saved = dict(saved)
    protocol = copy.deepcopy(saved['protocol_metadata'])
    protocol.update(source_identity=source['identity'], reused_without_training=True,
                    export_parent_checkpoint=str(parent), export_parent_sha256=file_sha256(parent))
    saved['protocol_metadata'] = protocol
    saved['export_note'] = 'Same w010 tensors; normalized relocated frozen-source path. Original checkpoint is unchanged.'
    from src.data.scheme1 import atomic_torch_save
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(saved, destination)
    atomic_json(protocol, destination.parent.parent / 'protocol.json')
    atomic_json({'status': 'complete', 'reused_without_training': True}, destination.parent.parent / 'training_complete.json')
    selection = {'protocol': 'reuse_subj01_existing_internal_selection_v1',
        'test_used_for_selection': False, 'source_evidence': str(evidence_path),
        'selected': {'checkpoint': str(destination), 'checkpoint_sha256': file_sha256(destination),
                     'config': spec['w010_config'], 'config_sha256': file_sha256(Path(spec['w010_config'])),
                     'label': 'w010_epoch004_reused'}, 'original_checkpoint_sha256': file_sha256(parent)}
    atomic_json(selection, selection_path)
    return selection


def cleanup(experiment, subject):
    folder = experiment / f'subj{subject:02d}'
    removed = []
    for path in [folder / 'features/posterior_means.npy', folder / 'features/cache_state.json',
                 folder / 'runs/adatoken_base_v1/checkpoints/last.pt',
                 folder / 'runs/adatoken_w010_v1/checkpoints/last.pt',
                 folder / 'runs/adatoken_base_fast_v2/checkpoints/last.pt',
                 folder / 'runs/adatoken_w010_fast_v2/checkpoints/last.pt']:
        if path.is_file():
            if experiment.resolve() not in path.resolve().parents:
                raise ValueError(f'Cleanup escaped the new experiment: {path}')
            removed.append({'path': str(path), 'bytes': path.stat().st_size})
            path.unlink()
    atomic_json({'reason': 'Final subject reconstruction/evaluation complete; all selected and candidate weights retained',
                 'removed_reproducible_cache_and_completed_optimizer_states': removed}, folder / 'cleanup.json')


def subject_work(subject, spec, cfg, experiment, queue, audit):
    folder = experiment / f'subj{subject:02d}'
    done_path = folder / 'complete.json'
    row_audit = next(row for row in audit['subjects'] if row['subject'] == subject)
    if done_path.is_file():
        done = json.loads(done_path.read_text())
        selected = json.loads((folder / 'selection.json').read_text())['selected']
        dataset_dir = Path(done['dataset_directory'])
        ids = np.load(ROOT / f'data/processed/nsd/subj{subject:02d}/nsd_test_image_id_sub{subject}.npy')[:row_audit['test_targets']]
        if (file_sha256(Path(selected['checkpoint'])) != selected['checkpoint_sha256']
                or not images_ok(dataset_dir, subject, 'nsd_test', ids, selected['checkpoint_sha256'])
                or not metrics_ok(dataset_dir, subject, len(ids))):
            raise ValueError(f'Completed subject artifacts failed verification: {subject}')
        if cfg['cleanup_completed_training_state']:
            cleanup(experiment, subject)
        print(f'[复用已完成被试] subj{subject:02d}', flush=True)
        return
    rest = cfg['cooldown_scope'] == 'stage'

    def stage(label, action, verify):
        return queue.run(f'subj{subject:02d}/{label}', action, verify, rest=rest)

    if not prepared_ok(spec, subject):
        stage('prepare', lambda: child(experiment, subject, 'prepare', ['scripts/prepare_scheme1_data.py',
            '--config', spec['brain_config'], '--subject', str(subject), '--reuse']), lambda: prepared_ok(spec, subject))
    if subject == 8:
        brain_cfg, _ = load_scheme1_config(spec['brain_config'], ROOT)
        run = training_directory(brain_cfg, ROOT, subject)
        if not ((run / 'training_complete.json').is_file() and Path(spec['brain_checkpoint']).is_file()):
            arguments = ['scripts/train_brainvae_scheme1.py', '--config', spec['brain_config'], '--subject', '8', '--device', 'cuda']
            if (run / 'checkpoints/last.pt').exists():
                arguments.append('--resume')
            stage('train_missing_brainvae', lambda: child(experiment, subject, 'train_missing_brainvae', arguments),
                  lambda: (run / 'training_complete.json').is_file() and Path(spec['brain_checkpoint']).is_file())
    base_config, brain_cfg, paths = load_mapping_config(spec['base_config'], ROOT)
    source = load_brain_source(base_config, brain_cfg, ROOT, subject)
    manifest, split = source['manifest'], source['split']
    source_identity = source['identity']
    atomic_json(source_identity, folder / 'frozen_source.json')
    del source
    if spec['reuse_head']:
        selection = export_subj01(spec, experiment)
    else:
        bank = folder / 'features'
        stage('cache_frozen_means', lambda: child(experiment, subject, 'cache_frozen_means',
            ['scripts/train_clip_mapping.py', '--config', spec['base_config'], '--subject', str(subject), '--device', 'cuda', '--cache-only']),
            lambda: verified_completed_mean_cache(bank, source_identity, manifest['n_source_images'], (256, 1664)) is not None)
        stage('train_adatoken', lambda: head_train(experiment, subject, 'train_adatoken', spec['base_config']),
              lambda: training_ok(spec['base_config'], subject))
        stage('train_w010', lambda: head_train(experiment, subject, 'train_w010', spec['w010_config']),
              lambda: training_ok(spec['w010_config'], subject))
        original_run = training_directory(base_config, ROOT, subject)
        refined_config, _, _ = load_mapping_config(spec['w010_config'], ROOT)
        refined_run = training_directory(refined_config, ROOT, subject)
        if cfg.get('selection_policy') == 'fixed_epoch004':
            checkpoint = refined_run / 'checkpoints/epoch004.pt'
            config_path = Path(spec['w010_config'])
            selection = {'protocol': 'predeclared_fixed_w010_epoch004_v2',
                'test_used_for_selection': False, 'validation_images_generated': False,
                'selected': {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
                             'config': str(config_path), 'config_sha256': file_sha256(config_path),
                             'label': 'w010_epoch004'}}
            atomic_json(selection, folder / 'selection.json')
        else:
            val_ids = split['source_image_ids'][split['val_indices']][:100]
            subset = folder / 'validation_subset_100.json'
            if not subset.exists():
                atomic_json({'protocol': 'fixed_internal_validation_generation_subset_v1', 'subject': subject,
                    'data_fingerprint': manifest['data_fingerprint'], 'image_ids': val_ids.tolist()}, subset)
            records = []
            candidates = [('baseline', Path(spec['base_config']), original_run / 'checkpoints/best.pt'),
                *[(f'w010_epoch{epoch:03d}', Path(spec['w010_config']), refined_run / f'checkpoints/epoch{epoch:03d}.pt') for epoch in range(1, 5)]]
            for label, config_path, checkpoint in candidates:
                candidate_sha = file_sha256(checkpoint)
                directory, tag = evaluation_paths(config_path, checkpoint, 'nsd_val', label, 100)
                command = decode_command(subject, config_path, checkpoint, 'nsd_val', tag, 100, subset)
                stage(f'{label}_validation_reconstruction',
                      lambda c=command, l=label: child(experiment, subject, l + '_validation_reconstruction', c),
                      lambda d=directory, sha=candidate_sha: images_ok(d, subject, 'nsd_val', val_ids, sha))
                stage(f'{label}_validation_metrics', lambda d=directory, l=label: evaluate(experiment, subject, l + '_validation_metrics', d),
                      lambda d=directory: metrics_ok(d, subject, 100))
                records.append(read_candidate(label, directory, checkpoint, config_path))
            selection = choose_w010(records[0], records[1:])
            atomic_json(selection, folder / 'selection.json')
    selected = selection['selected']
    checkpoint, config_path = Path(selected['checkpoint']), Path(selected['config'])
    if file_sha256(checkpoint) != selected['checkpoint_sha256'] or file_sha256(config_path) != selected['config_sha256']:
        raise ValueError('Selected checkpoint/config changed after internal selection')
    maximum = cfg['test_max_items']
    ids = split['test_image_ids'][:row_audit['test_targets']]
    directory, tag = evaluation_paths(config_path, checkpoint, 'nsd_test', 'w010_selected', maximum)
    command = decode_command(subject, config_path, checkpoint, 'nsd_test', tag, maximum)
    stage('nsd_test_reconstruction', lambda: child(experiment, subject, 'nsd_test_reconstruction', command),
          lambda: images_ok(directory, subject, 'nsd_test', ids, selected['checkpoint_sha256']))
    stage('nsd_test_metrics', lambda: evaluate(experiment, subject, 'nsd_test_metrics', directory),
          lambda: metrics_ok(directory, subject, len(ids)))
    metrics = json.loads((directory / 'image_metrics.json').read_text())
    atomic_json({'subject': subject, 'status': 'complete', 'source_kind': spec['source_kind'],
                'n_targets': len(ids), 'partial_test': bool(maximum), 'dataset_directory': str(directory),
                'selection_file': str(folder / 'selection.json'), 'metrics': metrics['metrics']}, done_path)
    if cfg['cleanup_completed_training_state']:
        cleanup(experiment, subject)
    if cfg['cooldown_scope'] == 'subject':
        queue.ledger.update(last_stage=f'subj{subject:02d}/complete', rest_until=queue.clock() + cfg['cooldown_seconds'])
        atomic_json(queue.ledger, queue.path)


def summarize(cfg, experiment):
    rows = []
    for subject in cfg['subjects']:
        path = experiment / f'subj{subject:02d}/complete.json'
        if path.is_file():
            row = json.loads(path.read_text())
            rows.append({**{k: row[k] for k in ('subject', 'source_kind', 'n_targets', 'partial_test')}, **row['metrics']})
    atomic_json({'subjects': rows, 'completed': len(rows), 'requested': len(cfg['subjects'])}, experiment / 'summary.json')
    with (experiment / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['subject', 'source_kind', 'n_targets', 'partial_test', *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/w010_multisubject_suite.yaml')
    parser.add_argument('--action', choices=('all', 'init', 'preflight', 'plan'), default='all')
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--test-max-items', type=int)
    parser.add_argument('--cooldown-scope', choices=('stage', 'subject'))
    parser.add_argument('--experiment-root')
    args = parser.parse_args()
    cfg, experiment = load_suite(args)
    plan = initialize(cfg, experiment)
    if args.action in ('init', 'plan'):
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    queue = StageQueue(experiment, cfg['cooldown_seconds']) if args.action == 'all' else None
    if queue:
        queue.state(status='preflight', formal_pipeline_started=True, failures=[],
                    cooldown_seconds=cfg['cooldown_seconds'], selection_policy=cfg.get('selection_policy', 'internal_generation'))
    try:
        audit = preflight(cfg, experiment, plan, gpu=args.action == 'all')
    except Exception as error:
        if queue:
            queue.state(status='failed', error=str(error))
        raise
    if args.action == 'preflight':
        return

    def stop(signum, frame):
        if ACTIVE is not None and ACTIVE.poll() is None:
            ACTIVE.send_signal(signum)
            # Do not re-enter Popen.wait from a signal handler while its lock is held.
        queue.state(status='interrupted', signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    failures = []
    try:
        for subject in cfg['subjects']:
            try:
                spec = plan['subjects'][str(subject)]
                if shutil.disk_usage(experiment).free < 35 * 1024**3 and not (experiment / f'subj{subject:02d}/complete.json').exists():
                    raise RuntimeError('Need 35 GiB free for the next subject; prior completed banks/optimizer states may need cleanup')
                subject_work(subject, spec, cfg, experiment, queue, audit)
            except Exception as error:
                failures.append({'subject': subject, 'error': str(error)})
                print(f'[被试失败] subj{subject:02d}: {error}', flush=True)
                # Failed computations also leave the GPU idle before the next subject.
                queue.ledger.update(last_stage=f'subj{subject:02d}/failed', rest_until=queue.clock() + cfg['cooldown_seconds'])
                atomic_json(queue.ledger, queue.path)
            summarize(cfg, experiment)
        queue.state(status='failed' if failures else 'complete', subject=None, failures=failures,
                    cooldown_remaining_seconds=0, resume_at=None)
        if failures:
            raise RuntimeError(f'Unfinished subjects: {[row["subject"] for row in failures]}')
    except BaseException as error:
        if not isinstance(error, SystemExit):
            queue.state(status='failed', error=str(error), failures=failures)
        raise
    print('[全部完成] w010 train, internal selection, NSD test reconstruction and eight metrics', flush=True)


if __name__ == '__main__':
    main()
