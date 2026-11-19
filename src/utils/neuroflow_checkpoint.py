"""Author NeuroFlow checkpoint and preprocessing identity, separate from SynBrain."""
from __future__ import annotations

import json

import torch

from src.data.neuroflow import load_session_statistics
from src.data.scheme1 import (
    Scheme1NSDDataset, check_prepared_config, data_directory, file_sha256, load_prepared, resolve_path,
)
from src.models.brainvae_factory import build_brainvae
from src.utils.config import load_yaml

UPSTREAM_COMMIT = '9aeae32f9f487bde1e45dbd575891d79900b895e'


def read_neuroflow_weights(path, model_config):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    weights = checkpoint.get('model')
    if not isinstance(weights, dict) or not weights or any(not torch.is_tensor(v) for v in weights.values()):
        raise ValueError('Expected the original NeuroFlow checkpoint[model] state dictionary')
    with torch.device('meta'):
        model = build_brainvae(model_config)
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(weights)), sorted(set(weights) - set(expected))
    mismatched = {key: [list(weights[key].shape), list(expected[key].shape)]
                  for key in set(weights) & set(expected) if weights[key].shape != expected[key].shape}
    if missing or unexpected or mismatched:
        raise ValueError(f'NeuroVAE checkpoint is not strictly compatible: missing={missing}, '
                         f'unexpected={unexpected}, shapes={mismatched}')
    model.load_state_dict(weights, strict=True, assign=True)
    metadata = {'epoch_zero_based': int(checkpoint.get('epoch', -1)), 'state_tensors': len(weights),
                'parameters': sum(value.numel() for value in weights.values()),
                'checkpoint_training_data_metadata': 'not_recorded_in_original_checkpoint'}
    return weights, metadata


def load_external_neuroflow_source(config, brain_config, root, subject, checkpoint_override=''):
    from src.utils.clip_mapping import MAPPING_PROTOCOL

    specification = config.get('external_source', {})
    if specification.get('subject') != subject:
        raise ValueError('NeuroFlow checkpoint subject must match external_source.subject')
    if specification.get('representation') != 'generation_posterior_mean':
        raise ValueError('Use NeuroFlow generation posterior mean; the retrieval projector is a different branch')
    model_path = resolve_path(specification['model_config'], root)
    model_config = load_yaml(model_path)
    prepared = data_directory(brain_config, root, subject)
    manifest, split, normalizer = load_prepared(prepared)
    check_prepared_config(brain_config, manifest)
    if model_config['architecture'] != 'neuroflow' or model_config['ddconfig']['voxel_dim'] != manifest['n_voxels']:
        raise ValueError('NeuroFlow model configuration does not match prepared NSD voxels')
    if any(model_config[key] != config['model'][key] for key in ('latent_tokens', 'embed_dim')):
        raise ValueError('NeuroFlow / mapping head dimensions differ')
    requested = checkpoint_override or config.get('brainvae_checkpoint', '')
    if not requested:
        raise ValueError('Set brainvae_checkpoint to the frozen NeuroFlow last.pth')
    checkpoint_path = resolve_path(requested, root)
    weights, metadata = read_neuroflow_weights(checkpoint_path, model_config)
    statistics_dir = resolve_path(specification['preprocessing_directory'], root)
    from src.data.scheme1 import load_scheme1_config
    _, paths = load_scheme1_config(config['brainvae_config'], root)
    statistics, preprocessing = load_session_statistics(paths, subject, manifest, statistics_dir,
                                                       int(specification.get('sessions', 40)))
    # Check the small trial map against the design again at preparation time;
    # its content hash subsequently protects row/repeat/session correspondence.
    for name, ids in (('train', split['source_image_ids']), ('test', split['test_image_ids'])):
        if statistics[name + '_trials'].shape != (len(ids), 3):
            raise ValueError('NeuroFlow session map differs from the prepared image identities')
    digest = file_sha256(checkpoint_path)
    if specification.get('checkpoint_sha256') and specification['checkpoint_sha256'] != digest:
        raise ValueError('NeuroFlow frozen checkpoint SHA256 differs from the pinned source')
    selection_path = resolve_path(config['comparison']['selection_manifest'], root)
    selection = json.loads(selection_path.read_text())
    reference_dataset = Scheme1NSDDataset(paths.nsd_root, subject, 'test', normalizer,
                                         average_repeats=True, load_targets=False,
                                         expected_ids=split['test_image_ids'])
    if (selection.get('subject') != subject or selection.get('dataset') != 'nsd_test'
            or selection.get('n_targets') != 250 or selection.get('aggregation') != 'mean_fmri'
            or selection.get('data_fingerprint') != manifest['data_fingerprint']
            or selection.get('records') != [reference_dataset.record(i) for i in range(250)]):
        raise ValueError('NeuroFlow comparison must use exactly the existing NSD250 identities/order')
    identity = {'protocol': MAPPING_PROTOCOL, 'source_kind': 'neuroflow_external', 'subject': subject,
                'brainvae_checkpoint': str(checkpoint_path.resolve()), 'brainvae_checkpoint_sha256': digest,
                'brainvae_model_config': model_config, 'data_fingerprint': manifest['data_fingerprint'],
                'preprocessing_sha256': manifest['artifact_sha256']['preprocessing.npz'],
                'source_debug_training': False, 'external_source': specification,
                'author_checkpoint_metadata': metadata, 'author_model_config_sha256': file_sha256(model_path),
                'upstream_commit': UPSTREAM_COMMIT,
                'encoder_representation': 'generation_posterior_mean',
                'encoder_input_normalization': 'neuroflow_within_session_zscore_then_repeat_mean',
                'neuroflow_preprocessing': preprocessing,
                'comparison_selection_manifest_sha256': file_sha256(selection_path),
                'encoder_training_data_provenance': specification.get('training_data_provenance', 'unknown'),
                'upstream_encoder_validation_uses_nsd_test': True}
    return {'identity': identity, 'weights': weights, 'manifest': manifest, 'split': split,
            'normalizer': normalizer, 'prepared': prepared, 'neuroflow_statistics': statistics,
            'evaluation_selection': {'selection_manifest': str(selection_path.resolve()),
                                     'selection_manifest_sha256': file_sha256(selection_path)}}
