#!/usr/bin/env python3
"""Encode held-out natural and geometric imagery trials with the matching Ridge BrainVAE."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
import yaml
import ridge_multisubject_three_text as ridge
import run_multisubject_imagery_natural as pilot
from src.data.scheme1 import file_sha256, validate_tag
from src.utils.clip_mapping import atomic_json, frozen_brain, mapping_lock
from src.utils.imagery_w010_pilot import split_trials
from multisubject_resume import archive_incomplete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, choices=range(2,9), required=True)
    parser.add_argument('--ridge-run', default='ridge_mu_three_text_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='archive incomplete known outputs and retry')
    args = parser.parse_args()
    validate_tag(args.ridge_run)
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    root = ROOT/f'output_1104/multisubject_nsd_imagery_v1/subj{args.subject:02d}'
    cfg = yaml.safe_load((root/'config_imagery_natural_k60_3samples.yaml').read_text())
    exp = ROOT/cfg['experiment_root']
    source = pilot.run_source(cfg)
    identity = pilot.require_identity(cfg, exp, source)
    data_path = source['prepared']/'imagery_trials.npz'
    if file_sha256(data_path) != identity['imagery_trials_sha256']:
        raise ValueError('Imagery trial data differs from the frozen split protocol')
    data = pilot.load_trials(source)
    natural = json.loads((exp/'protocol/split.json').read_text())
    if natural != split_trials(data, cfg['split_seed'], 'complex'):
        # prepare adds provenance columns; compare the actual partition below instead.
        expected = split_trials(data, cfg['split_seed'], 'complex')
        if (natural['targets'] != expected['targets'] or any(natural['rows'][name][key] != expected['rows'][name][key]
                for name in expected['targets'] for key in ('train','val','test','vision_train','vision_val'))):
            raise ValueError('Natural trial split differs from deterministic subject protocol')
    geometry = split_trials(data, cfg['split_seed'], 'simple')
    rows, means = [], []
    for label, split in [('natural', natural), ('geometry', geometry)]:
        if split['subject'] != args.subject or len(split['targets']) != 6:
            raise ValueError('Trial split subject or target count differs')
        for target in split['targets']:
            partition = split['rows'][target]
            indices = np.asarray(partition['test'], dtype=np.int64)
            if len(indices)!=4 or set(indices) & (set(partition['train']) | set(partition['val'])):
                raise ValueError('Imagery trial leakage')
            if not np.all(data['stimulus_id'][indices].astype(str)==target) or not np.all(data['task_type'][indices].astype(str)=='imagery'):
                raise ValueError('Imagery trial identity mismatch')
            values = np.asarray(data['fmri_scaled'][indices], dtype=np.float32).mean(axis=0)
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite fMRI mean')
            means.append(values)
            rows.append({'row_index':len(rows),'group':label,'stimulus_id':target,
                         'stimulus_type':split['stimulus_type'],'test_rows':indices.tolist(),
                         'source_trial_ids':[int(v) for v in data['global_trial_idx'][indices]],
                         'runs':[int(v) for v in data['run'][indices]]})
    model_dir = root/'ridge'/args.ridge_run/'model'
    model = json.loads((model_dir/'model_complete.json').read_text())
    if (model.get('status')!='complete' or model.get('debug')
            or model['brain_source']['subject'] != args.subject
            or model['brain_source']['brainvae_checkpoint_sha256'] != identity['brainvae_sha256']
            or model['brain_source']['data_fingerprint'] != identity['data_fingerprint']):
        raise ValueError('Ridge BrainVAE identity differs from imagery source')
    print(f'[input] subject={args.subject} natural=6 geometry=6 held-out trials=4 each',flush=True)
    if args.dry_run:
        return
    device = torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    output = root/'imagery_ridge_inputs_3samples_v1'
    with mapping_lock(output):
        if (output/'imagery_mu_manifest.json').exists():
            raise FileExistsError('Completed imagery μ output is immutable')
        existing=[p for p in output.iterdir() if p.name not in ('.mapping.lock','incomplete_attempts')]
        if existing:
            if not args.resume:raise FileExistsError('Existing output must be preserved and inspected')
            archived=archive_incomplete(output,('imagery_mu.npy','imagery_mu.npy.tmp'))
            print('[resume] preserved incomplete μ files in {}'.format(archived),flush=True)
        brain_source,_ = ridge.load_subject_context(args.subject)
        if brain_source['identity'] != model['brain_source']:
            raise ValueError('Loaded BrainVAE differs from Ridge training source')
        brain = frozen_brain(brain_source,device)
        del brain_source['weights']
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result=[]
        with torch.inference_mode():
            for start in range(0,len(means),args.batch_size):
                batch=torch.from_numpy(np.stack(means[start:start+args.batch_size])).to(device)
                mu=brain.encode(batch).mode().float()
                if tuple(mu.shape[1:])!=(256,1664) or not torch.isfinite(mu).all():
                    raise FloatingPointError('Invalid BrainVAE mean')
                result.append(mu.cpu().numpy())
        values=np.concatenate(result)
        path=output/'imagery_mu.npy'
        tmp=path.with_name(path.name+'.tmp')
        with tmp.open('wb') as handle:
            np.save(handle,values,allow_pickle=False)
        os.replace(tmp,path)
        atomic_json({'status':'complete','subject':args.subject,'dataset':'imagery',
                     'representation':'raw_unadapted_mean_then_brainvae_posterior_mean',
                     'mu_sha256':file_sha256(path),'shape':list(values.shape),'rows':rows,
                     'brainvae_sha256':identity['brainvae_sha256'],'data_fingerprint':identity['data_fingerprint'],
                     'trials_sha256':file_sha256(data_path),'natural_split_sha256':file_sha256(exp/'protocol/split.json'),
                     'geometry_split':geometry,'ridge_model_metadata_sha256':file_sha256(model_dir/'model_complete.json'),
                     'test_images_or_text_used_as_input':False},output/'imagery_mu_manifest.json')
        print('[complete] '+str(path),flush=True)


if __name__=='__main__':
    main()
