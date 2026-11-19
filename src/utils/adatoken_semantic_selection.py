"""Select on generated INTERNAL validation images; test metrics are never inputs."""
import json
import math
from pathlib import Path

from src.data.scheme1 import file_sha256

METRICS = ('PixCorr', 'SSIM', 'Alex2_2way', 'Alex5_2way', 'Incep_2way',
           'CLIP_2way', 'Eff_corr_distance', 'SwAV_corr_distance')
POLICY = {'n_targets': 100, 'seed': 42, 'num_steps': 38,
          'max_pixcorr_drop': 0.01, 'max_ssim_drop': 0.01,
          'max_inception_drop': 0.0,
          'ranking': ['CLIP_2way', 'Incep_2way', 'SSIM']}


def read_candidate(label, directory, checkpoint, config_path, policy=POLICY):
    directory, checkpoint, config_path = map(Path, (directory, checkpoint, config_path))
    metrics_path = directory / 'image_metrics.json'
    scores = json.loads(metrics_path.read_text())
    metadata = json.loads((directory / 'metadata.json').read_text())
    manifest = json.loads((directory / 'sample00/manifest.json').read_text())
    if scores.get('dataset') != 'nsd_val' or metadata.get('dataset') != 'nsd_val':
        raise ValueError('Semantic selection accepts nsd_val only, never nsd_test')
    if (scores.get('n_targets') != policy['n_targets'] or scores.get('n_samples') != 1
            or metadata.get('num_samples') != 1 or scores.get('debug_training') is not False
            or metadata.get('debug_training') is not False):
        raise ValueError('Validation target count / sampling / formal-training identity differs')
    if (metadata.get('seed') != policy['seed'] or metadata.get('num_steps') != policy['num_steps']
            or metadata.get('rgb_mode') != 'standard' or metadata.get('aggregation') != 'mean_fmri'
            or not metadata.get('validation_subset_sha256')):
        raise ValueError('Validation generation protocol differs')
    sha = file_sha256(checkpoint)
    if metadata.get('checkpoint_sha256') != sha:
        raise ValueError('Validation scores refer to a different checkpoint')
    for key in ('checkpoint_sha256', 'data_fingerprint', 'dataset', 'num_samples'):
        if manifest.get(key) != metadata.get(key):
            raise ValueError('Validation manifest and metadata differ')
    items = manifest['items']
    if len(items) != policy['n_targets'] or len({i['image_id'] for i in items}) != len(items):
        raise ValueError('Validation identities are missing or duplicated')
    for index, item in enumerate(items):
        if item.get('task_type') != 'nsd_val' or item['seed'] != policy['seed'] + index:
            raise ValueError('Validation task or seed mismatch')
        image = directory / 'sample00' / item['path']
        if not image.is_file() or image.stat().st_size == 0:
            raise ValueError('Validation image missing or empty')
    values = {k: float(scores['metrics'][k]) for k in METRICS}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite validation metrics')
    return {'label': label, 'metrics': values, 'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': sha, 'config': str(config_path.resolve()),
            'config_sha256': file_sha256(config_path), 'metrics_file': str(metrics_path.resolve()),
            'metrics_sha256': file_sha256(metrics_path),
            'identity': {k: metadata.get(k) for k in ('subject','data_fingerprint',
                'brainvae_checkpoint_sha256','validation_subset_sha256')},
            'image_ids': [i['image_id'] for i in items]}


def select_candidate(baseline, candidates, policy=POLICY):
    rows = [baseline] + list(candidates)
    if len({r['label'] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate labels')
    reference = baseline['metrics']
    audited, eligible = [], []
    for row in rows:
        if row['identity'] != baseline['identity'] or row['image_ids'] != baseline['image_ids']:
            raise ValueError('Candidates must share validation identities and source BrainVAE')
        m = row['metrics']
        valid = (m['PixCorr'] >= reference['PixCorr'] - policy['max_pixcorr_drop']
                 and m['SSIM'] >= reference['SSIM'] - policy['max_ssim_drop']
                 and m['Incep_2way'] >= reference['Incep_2way'] - policy['max_inception_drop'])
        audited.append({**row, 'eligible': valid})
        if valid:
            eligible.append(row)
    # Stable lexical tie break, independent of command-line candidate order.
    eligible.sort(key=lambda r: r['label'])
    winner = max(eligible, key=lambda r: tuple(r['metrics'][k] for k in policy['ranking']))
    return {'protocol': 'adatoken_semantic_validation_selection_v2', 'selection_dataset': 'nsd_val',
            'policy': policy, 'baseline': baseline, 'selected': winner,
            'candidates': audited, 'test_used_for_selection': False}
