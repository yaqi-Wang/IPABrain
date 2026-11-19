"""Load the same 12 subj01 stimulus images and held-out task trials as B0."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_scheme1_clip import load_imagery_truth
from src.data.scheme1 import file_sha256


class ImageTaskPairs(Dataset):
    def __init__(self, tokens, fmri, records):
        self.tokens, self.fmri, self.records = tokens, torch.as_tensor(fmri, dtype=torch.float32), records
        if len(tokens) != len(fmri) or len(fmri) != len(records):
            raise ValueError('Stimulus / task response count mismatch')
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        return {'clip_vision':self.tokens[i], 'fmri':self.fmri[i], 'image_id':i}


def imagery_pairs(source, paths, subject):
    if subject != 1:
        raise ValueError('Imagery trial matching is currently implemented for subj01 only')
    vision_manifest = ROOT/'output_1104/imagery_k60_ridge_subj01_v1/inputs/vision_mu_manifest.json'
    imagery_manifest = ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs/imagery_mu_manifest.json'
    vm, im = [json.loads(p.read_text()) for p in (vision_manifest, imagery_manifest)]
    for m in (vm,im):
        if m['status']!='complete' or m['brainvae_sha256']!=source['identity']['brainvae_checkpoint_sha256'] or m['data_fingerprint']!=source['identity']['data_fingerprint']:
            raise ValueError('Imagery trial manifest differs from selected BrainVAE/data')
    trials = source['prepared']/'imagery_trials.npz'
    if file_sha256(trials) != im['trials_sha256']:
        raise ValueError('Imagery trials changed since the existing split was recorded')
    with np.load(trials,allow_pickle=False) as f:
        raw={name:f[name] for name in f.files}
    cache=paths.imagery_root/f'subj{subject:02d}'/'imagery_trials.npz'
    tokens, token_records, _=load_imagery_truth(cache,subject)
    keys=[(r['stimulus_type'],r['stimulus_id']) for r in token_records]
    if len(keys)!=12 or len(set(keys))!=12: raise ValueError('Expected 12 unique imagery image tokens')
    lookup={k:tokens[i] for i,k in enumerate(keys)}
    gid=np.asarray(raw['global_trial_idx'],dtype=np.int64)
    if len(set(gid.tolist()))!=len(gid): raise ValueError('Duplicate global trial indices')
    id_to_row={int(v):i for i,v in enumerate(gid)}
    result={}
    for task, manifest_rows in (('vision',vm['records']),('imagery',im['rows'])):
        token_rows=[]; fmri_rows=[]; records=[]
        for row in manifest_rows:
            key=(row['stimulus_type'],row['stimulus_id'])
            if key not in lookup: raise ValueError(f'No target-image CLIP for {key}')
            trial_ids=[int(x) for x in row['source_trial_ids']]
            if len(trial_ids)!=(8 if task=='vision' else 4) or len(set(trial_ids))!=len(trial_ids):
                raise ValueError('Unexpected trial count')
            indices=[id_to_row[i] for i in trial_ids]
            for idx in indices:
                if (str(raw['stimulus_id'][idx])!=key[1] or str(raw['stimulus_type'][idx])!=key[0]
                    or str(raw['task_type'][idx])!=task or (task=='vision' and raw['condition_index'][idx]!=0)):
                    raise ValueError('Trial identity/condition mismatch')
            values=np.asarray(raw['fmri_scaled'][indices],dtype=np.float32).mean(0)
            fmri_rows.append(source['normalizer'].transform(values,'imagery_scaled'))
            token_rows.append(lookup[key])
            records.append({'stimulus_type':key[0], 'stimulus_id':key[1], 'task_type':task,
                            'source_trial_ids':trial_ids})
        if len(records)!=12 or set((r['stimulus_type'],r['stimulus_id']) for r in records)!=set(keys):
            raise ValueError('Imagery and vision stimulus sets differ')
        result[task]=ImageTaskPairs(torch.stack(token_rows), np.stack(fmri_rows),records)
    identity={'source_imagery_trials_sha256':file_sha256(trials),
              'target_image_clip_cache_sha256':file_sha256(cache),
              'vision_manifest_sha256':file_sha256(vision_manifest),
              'imagery_manifest_sha256':file_sha256(imagery_manifest)}
    return result,identity


def stratified_image_split(pairs, seed):
    """Split identities, not trials: 4/1/1 per natural and geometric type."""
    from torch.utils.data import Subset
    reference = pairs['vision'].records
    keys = [(r['stimulus_type'], r['stimulus_id']) for r in reference]
    if len(keys) != 12 or len(set(keys)) != 12:
        raise ValueError('Expected 12 unique stimuli')
    rng = np.random.default_rng(seed)
    split_keys = {'train': [], 'val': [], 'test': []}
    for kind in ('complex', 'simple'):
        candidates = sorted(key for key in keys if key[0] == kind)
        if len(candidates) != 6:
            raise ValueError('Expected six natural and six geometric stimuli')
        order = rng.permutation(len(candidates))
        for name, indices in (('train', order[:4]), ('val', order[4:5]), ('test', order[5:])):
            split_keys[name].extend(candidates[int(i)] for i in indices)
    result = {}
    for task, dataset in pairs.items():
        by_key = {(r['stimulus_type'], r['stimulus_id']): i for i, r in enumerate(dataset.records)}
        if set(by_key) != set(keys) or len(by_key) != len(dataset):
            raise ValueError('Task stimulus identities differ')
        for key in keys:
            if not torch.equal(dataset.tokens[by_key[key]], pairs['vision'].tokens[keys.index(key)]):
                raise ValueError(f'Task image tokens differ for {key}')
        result[task] = {name: Subset(dataset, [by_key[key] for key in selected])
                        for name, selected in split_keys.items()}
    info = {name: [{'stimulus_type': kind, 'stimulus_id': sid} for kind, sid in values]
            for name, values in split_keys.items()}
    return result, info
