"""NeuroFlow's original session z-scores, reconstructed from verified NSD betas.

The upstream preparation normalizes all 750 unlabeled trials in each session,
including held-out fMRI. It does not use image features. This is intentionally
different from the local SynBrain beta/2000 and training-only z-score protocols.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as spio
import torch

from src.data.scheme1 import (
    Scheme1NSDDataset, check_source_files, file_sha256, file_signature,
)

PREPROCESSING_PROTOCOL = 'neuroflow_within_session_zscore_v1'


def recover_integer_betas(values, stored_divisor=2000.0):
    """Reverse the existing lossless-to-integer NSD beta/2000 representation.

    A full FP32 round trip is mandatory. Refuse non-integer or differently
    scaled data rather than approximating the external checkpoint's inputs.
    """
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all() or stored_divisor != 2000.0:
        raise ValueError('NeuroFlow reconstruction requires verified finite beta/2000 arrays')
    raw = np.rint(values.astype(np.float64) * stored_divisor).astype(np.float32)
    if not np.array_equal(raw / np.float32(stored_divisor), values):
        raise ValueError('Cached beta values fail the exact integer/2000 round trip')
    return raw


def session_statistics(raw_betas):
    # nibabel's beta_f[mask].transpose() in the source is Fortran-contiguous.
    # Preserve its layout and FP32 reduction order, in addition to the formula.
    scaled = np.asfortranarray(raw_betas, dtype=np.float32) / np.float32(300.0)
    mean = np.mean(scaled, axis=0)
    std = np.clip(np.std(scaled, axis=0), 1e-8, 10000)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_repeats(values, trial_indices, means, stds):
    scaled = recover_integer_betas(values) / np.float32(300.0)
    session_indices = np.asarray(trial_indices, dtype=np.int64) // 750
    normalized = np.nan_to_num((scaled - means[session_indices]) / stds[session_indices])
    if not np.isfinite(normalized).all():
        raise FloatingPointError('Invalid NeuroFlow normalized fMRI')
    return normalized.astype(np.float32, copy=False)


def chronological_trial_indices(expdesign, subject, sessions, source_ids, test_ids):
    master = np.asarray(expdesign['masterordering']).reshape(-1)
    subjectim = np.asarray(expdesign['subjectim'])
    if not 1 <= subject <= len(subjectim) or sessions <= 0 or sessions * 750 > len(master):
        raise ValueError('Invalid subject/session count in NSD experiment design')
    lookups = {'train': {}, 'test': {}}
    for trial, master_id in enumerate(master[:sessions * 750]):
        image_id = int(subjectim[subject - 1, int(master_id) - 1]) - 1
        split = 'train' if master_id > 1000 else 'test'
        lookups[split].setdefault(image_id, []).append(trial)
    result = {}
    for name, ids in (('train', source_ids), ('test', test_ids)):
        if not np.array_equal(np.asarray(list(lookups[name])), ids):
            raise ValueError(f'{name} IDs/order differ from original NeuroFlow NSD grouping')
        if any(len(lookups[name][int(image_id)]) != 3 for image_id in ids):
            raise ValueError('This comparison requires all three repeats and complete sessions')
        result[name + '_trials'] = np.asarray([lookups[name][int(i)] for i in ids], dtype=np.int64)
    complete = np.concatenate([result['train_trials'].ravel(), result['test_trials'].ravel()])
    if not np.array_equal(np.sort(complete), np.arange(sessions * 750)):
        raise ValueError('Grouped source/test trials do not exactly partition every NSD session')
    return result


def preprocessing_specification(paths, subject, manifest, sessions=40):
    if manifest['subject'] != subject or manifest.get('source_sample_check', {}).get('status') != 'verified':
        raise ValueError('First verify the matching subject against raw NSD betas')
    if manifest.get('voxel_order') != 'native_XYZ_mask_C_order':
        raise ValueError('NeuroVAE requires the original native XYZ ROI voxel order')
    prep = manifest['input_spec']['preprocessing']
    if (prep['normalization'] != 'beta_scale' or prep['common_divisor'] != 2000
            or prep['nsd_stored_divisor'] != 2000):
        raise ValueError('Session reconstruction expects the verified unstandardized beta/2000 source')
    check_source_files(paths.nsd_root, subject, manifest, ('train_fmri', 'train_ids', 'test_fmri', 'test_ids'))
    design = paths.raw_nsd_root / 'nsddata/experiments/nsd/nsd_expdesign.mat'
    return {'protocol': PREPROCESSING_PROTOCOL, 'subject': subject, 'sessions': sessions,
            'base_data_fingerprint': manifest['data_fingerprint'],
            'source_files': {key: manifest['input_spec']['files'][key]
                             for key in ('train_fmri', 'train_ids', 'test_fmri', 'test_ids')},
            'experiment_design': file_signature(design), 'experiment_design_sha256': file_sha256(design),
            'formula': 'FP32 beta/300; per-session mean/std(ddof=0); std clip [1e-8,10000]; nan_to_num',
            'statistics_scope': 'all_750_unlabeled_trials_in_each_session',
            'test_fmri_used_for_session_statistics': True, 'target_images_or_features_used': False,
            'repeat_order': 'chronological_within_image', 'aggregation': 'zscore_each_repeat_then_mean_fmri'}


def prepare_session_statistics(paths, subject, manifest, split, directory: Path, sessions=40):
    """Produce only small session statistics; never duplicate grouped fMRI/CLIP."""
    from src.utils.clip_mapping import atomic_json, mapping_lock

    specification = preprocessing_specification(paths, subject, manifest, sessions)
    directory.mkdir(parents=True, exist_ok=True)
    with mapping_lock(directory):
        spec_path = directory / 'preparation_spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
            raise ValueError('NeuroFlow preparation inputs changed; use a new preprocessing directory')
        if (directory / 'manifest.json').exists():
            load_session_statistics(paths, subject, manifest, directory, sessions)
            return directory / 'session_statistics.npz'
        atomic_json(specification, spec_path)
        design = spio.loadmat(specification['experiment_design']['path'], squeeze_me=True)
        trials = chronological_trial_indices(design, subject, sessions,
                                              split['source_image_ids'], split['test_image_ids'])
        arrays = {name: np.load(manifest['input_spec']['files'][name + '_fmri']['path'], mmap_mode='r')
                  for name in ('train', 'test')}
        voxel_dim = int(manifest['n_voxels'])
        for name, values in arrays.items():
            if values.shape != (*trials[name + '_trials'].shape, voxel_dim):
                raise ValueError('Grouped fMRI shape differs from the complete session trial map')
        means, stds = [], []
        for session in range(sessions):
            raw_session = np.empty((750, voxel_dim), dtype=np.float32, order='F')
            for name, values in arrays.items():
                rows, repeats = np.where(trials[name + '_trials'] // 750 == session)
                frames = trials[name + '_trials'][rows, repeats] % 750
                raw_session[frames] = recover_integer_betas(values[rows, repeats])
            mean, std = session_statistics(raw_session)
            means.append(mean)
            stds.append(std)
            print(f'[NeuroFlow session z-score] {session + 1}/{sessions}', flush=True)
        target = directory / 'session_statistics.npz'
        temporary = directory / 'session_statistics.npz.tmp'
        with temporary.open('wb') as handle:
            np.savez(handle, session_mean=np.stack(means), session_std=np.stack(stds), **trials)
        temporary.replace(target)
        atomic_json({'specification': specification, 'statistics_sha256': file_sha256(target),
                     'integer_beta_roundtrip': 'verified_all_train_and_test_voxels'}, directory / 'manifest.json')
    return target


def load_session_statistics(paths, subject, manifest, directory: Path, sessions=40):
    metadata = json.loads((directory / 'manifest.json').read_text())
    if metadata['specification'] != preprocessing_specification(paths, subject, manifest, sessions):
        raise ValueError('NeuroFlow session preprocessing source mismatch')
    path = directory / 'session_statistics.npz'
    if file_sha256(path) != metadata['statistics_sha256']:
        raise ValueError('NeuroFlow session statistics changed')
    with np.load(path, allow_pickle=False) as values:
        statistics = {key: values[key] for key in values.files}
    mean, std = statistics['session_mean'], statistics['session_std']
    if (mean.shape != (sessions, manifest['n_voxels']) or std.shape != mean.shape
            or mean.dtype != np.float32 or std.dtype != np.float32
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('Invalid NeuroFlow session statistics')
    return statistics, metadata


class NeuroFlowNSDDataset(Scheme1NSDDataset):
    def __init__(self, *args, session_statistics, validation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_statistics = session_statistics
        self.session_trials = session_statistics[self.split + '_trials']
        self.validation = validation
        if self.session_trials.shape != self.fmri.shape[:2]:
            raise ValueError('NeuroFlow session map and grouped fMRI do not align')

    def __getitem__(self, index):
        row = self.image_index(index)
        normalized = normalize_repeats(self.fmri[row], self.session_trials[row],
                                       self.session_statistics['session_mean'], self.session_statistics['session_std'])
        values = normalized.mean(axis=0) if self.average_repeats else normalized[index % self.repeats]
        item = {'fmri': torch.from_numpy(np.array(values, dtype=np.float32, copy=True)),
                'index': index, 'image_id': int(self.image_ids[row])}
        if self.clip is not None:
            item['clip_vision'] = torch.from_numpy(np.array(self.clip[row], dtype=np.float32, copy=True))
        return item

    def record(self, index):
        record = super().record(index)
        return {**record, 'task_type': 'nsd_val'} if self.validation else record
