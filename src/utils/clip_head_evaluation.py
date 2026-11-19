"""Shared head provenance and genuine internal-validation inference datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.scheme1 import Scheme1NSDDataset, Scheme1ImageryDataset, check_source_files, digest_json
from src.utils.clip_mapping import atomic_json, load_brain_source, source_nsd_dataset


def load_trained_head_source(checkpoint_path, config, brain_config, root, subject,
                             expected_protocol, brainvae_override=''):
    if not (checkpoint_path.parent.parent/'training_complete.json').is_file():
        raise RuntimeError('Head training has not completed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    protocol = checkpoint.get('protocol_metadata', {})
    if protocol.get('protocol') != expected_protocol or protocol.get('subject') != subject:
        raise ValueError('Head checkpoint protocol / subject mismatch')
    if checkpoint['config'] != config['model']:
        raise ValueError('Head checkpoint model config mismatch')
    saved_source = protocol['source_identity']
    source = load_brain_source(config, brain_config, root, subject,
                               brainvae_override or saved_source['brainvae_checkpoint'])
    if source['identity'] != saved_source or protocol['data_fingerprint'] != source['manifest']['data_fingerprint']:
        raise ValueError('Head was trained with a different frozen BrainVAE / data protocol')
    for key, value in source['split'].items():
        if key not in checkpoint['split'] or not torch.equal(checkpoint['split'][key], torch.from_numpy(value)):
            raise ValueError('Head / BrainVAE image split mismatch')
    return checkpoint, source, protocol


class ValidationNSDDataset(Scheme1NSDDataset):
    def record(self, index):
        return {**super().record(index), 'task_type':'nsd_val'}


def head_evaluation_dataset(name, source, paths, subject, max_items=0, validation_subset=''):
    if name == 'nsd_test':
        check_source_files(paths.nsd_root, subject, source['manifest'], ('test_fmri','test_ids'))
        selection_metadata = {}
        if source['identity'].get('source_kind') == 'neuroflow_external':
            if not 1 <= max_items <= 250:
                raise ValueError('NeuroFlow NSD250 comparison requires --max-items between 1 and 250')
            selection_metadata = source['evaluation_selection']
        return source_nsd_dataset(source, paths.nsd_root, subject, 'test',
                                average_repeats=True, load_targets=False,
                                expected_ids=source['split']['test_image_ids']), selection_metadata
    if name != 'nsd_val':
        if source['identity'].get('source_kind') == 'neuroflow_external':
            raise ValueError('NeuroFlow session normalization is implemented for NSD only; choose nsd_test or nsd_val')
        if validation_subset:
            raise ValueError('validation-subset is only valid for nsd_val')
        return Scheme1ImageryDataset(source['prepared']/'imagery_trials.npz', subject,
                                    source['normalizer'], name, average_repeats=True), {}
    split = source['split']
    pool_ids = split['source_image_ids']
    train_ids, val_ids = pool_ids[split['train_indices']], pool_ids[split['val_indices']]
    if np.intersect1d(val_ids, train_ids).size or np.intersect1d(val_ids, split['test_image_ids']).size:
        raise ValueError('Internal validation identities overlap training or test')
    count = min(max_items, len(val_ids)) if max_items else len(val_ids)
    path = (Path(validation_subset) if validation_subset else
            source['prepared']/'evaluation_subsets'/f'nsd_val_first{count}.json')
    if path.is_file():
        saved = json.loads(path.read_text())
        selected = np.asarray(saved['image_ids'], dtype=np.int64)
    else:
        selected = val_ids[:count]
        saved = {'protocol':'fixed_internal_validation_generation_subset_v1', 'subject':subject,
                 'data_fingerprint':source['manifest']['data_fingerprint'], 'image_ids':selected.tolist()}
    if (saved.get('protocol') != 'fixed_internal_validation_generation_subset_v1'
            or saved.get('subject') != subject or saved.get('data_fingerprint') != source['manifest']['data_fingerprint']
            or selected.ndim != 1 or len(selected) != count or len(np.unique(selected)) != count
            or not np.isin(selected, val_ids).all()):
        raise ValueError('Fixed validation subset source / identities / count mismatch')
    indices_by_id = {int(image_id):int(row) for image_id,row in zip(val_ids, split['val_indices'])}
    rows = np.asarray([indices_by_id[int(image_id)] for image_id in selected], dtype=np.int64)
    check_source_files(paths.nsd_root, subject, source['manifest'], ('train_fmri','train_ids'))
    if source['identity'].get('source_kind') == 'neuroflow_external':
        dataset = source_nsd_dataset(source, paths.nsd_root, subject, 'train', rows, validation=True,
                                    average_repeats=True, load_targets=False, expected_ids=pool_ids)
    else:
        dataset = ValidationNSDDataset(paths.nsd_root, subject, 'train', source['normalizer'], rows,
                                      average_repeats=True, load_targets=False, expected_ids=pool_ids)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(saved, path)
    return dataset, {'validation_subset_sha256':digest_json(saved), 'validation_subset_path':str(path.resolve()),
                     'validation_image_ids':selected.tolist(), 'validation_source':'NSD_train_internal_image_holdout'}
