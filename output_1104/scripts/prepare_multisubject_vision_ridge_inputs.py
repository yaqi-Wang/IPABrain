#!/usr/bin/env python3
"""Prepare subj02-08 unadapted vision BrainVAE means for their own three-branch Ridge."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
import yaml
import ridge_multisubject_three_text as ridge
import run_multisubject_imagery_natural as natural
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import frozen_brain,mapping_lock
from src.utils.staged_queue import atomic_json

OUT=ROOT/'output_1104/multisubject_linear_imagery_v1'

def read(p):return json.loads(Path(p).read_text())

def source_inputs(subject):
    root=ROOT/f'output_1104/multisubject_nsd_imagery_v1/subj{subject:02d}'
    cfg=yaml.safe_load((root/'config_imagery_natural_k60_3samples.yaml').read_text())
    source=natural.run_source(cfg);exp=ROOT/cfg['experiment_root'];ident=natural.require_identity(cfg,exp,source)
    datap=source['prepared']/'imagery_trials.npz'
    if file_sha256(datap)!=ident['imagery_trials_sha256']:raise ValueError('Imagery data source changed')
    data=natural.load_trials(source)
    info=read(root/'imagery_ridge_inputs_3samples_v1/imagery_mu_manifest.json')
    if info.get('status')!='complete' or info.get('subject')!=subject or info.get('brainvae_sha256')!=ident['brainvae_sha256']:
        raise ValueError('Imagery input source changed')
    if info['mu_sha256']!=file_sha256(root/'imagery_ridge_inputs_3samples_v1/imagery_mu.npy'):
        raise ValueError('Imagery mean checksum mismatch')
    ridge_run=read(root/'ridge_selected_run.json')['run_name'];model=root/'ridge'/ridge_run/'model/model_complete.json'
    meta=read(model)
    if meta.get('status')!='complete' or meta.get('debug') or meta['brain_source']['subject']!=subject or meta['brain_source']['brainvae_checkpoint_sha256']!=ident['brainvae_sha256']:
        raise ValueError('Ridge model subject or BrainVAE mismatch')
    task=data['task_type'].astype(str);kinds=data['stimulus_type'].astype(str);ids=data['stimulus_id'].astype(str)
    rows=[];means=[]
    for i,target in enumerate(info['rows']):
        sid=target['stimulus_id'];kind=target['stimulus_type']
        indices=np.flatnonzero((task=='vision') & (kinds==kind) & (ids==sid))
        if len(indices)!=8 or np.any(data['condition_index'][indices]!=0):raise ValueError(f'Expected eight vision trials: {sid}')
        if set(data['stim_set'][indices].astype(str))!=({'B'} if kind=='complex' else {'A'}):raise ValueError('Incorrect stimulus set')
        values=np.asarray(data['fmri_scaled'][indices],dtype=np.float32).mean(0)
        if not np.isfinite(values).all():raise ValueError('Nonfinite input')
        rows.append({'row_index':i,'group':target['group'],'stimulus_id':sid,'stimulus_type':kind,
                     'vision_rows':indices.tolist(),'source_trial_ids':[int(x) for x in data['global_trial_idx'][indices]],
                     'runs':[int(x) for x in data['run'][indices]]})
        means.append(values)
    if len(rows)!=12 or [r['group'] for r in rows]!=['natural']*6+['geometry']*6:raise ValueError('Unexpected targets')
    return root,source,ident,model,rows,np.stack(means)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True,type=int,choices=range(2,9));p.add_argument('--device',default='cuda:0');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    root,source,ident,model,rows,values=source_inputs(a.subject)
    print(json.dumps({'status':'preflight_ok','subject':a.subject,'targets':len(rows),'n_trials_per_target':8,'voxels':values.shape[1]}),flush=True)
    if a.dry_run:return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=OUT/f'subj{a.subject:02d}'/'vision_ridge_inputs';target=output/'vision_mu.npy';manifest=output/'imagery_mu_manifest.json'
    with mapping_lock(output):
        if manifest.exists():
            old=read(manifest)
            if old.get('status')!='complete' or not target.is_file() or old['mu_sha256']!=file_sha256(target) or old['rows']!=rows or old['brainvae_sha256']!=ident['brainvae_sha256'] or old['ridge_model_metadata_sha256']!=file_sha256(model):raise ValueError('Existing vision means changed')
            print('[reuse] verified vision means',flush=True);return
        if target.exists():
            from multisubject_resume import archive_file
            print('[resume] preserving unverified means:',archive_file(target),flush=True)
        brain_source,_=ridge.load_subject_context(a.subject)
        if brain_source['identity']!=read(model)['brain_source']:raise ValueError('Ridge training BrainVAE changed')
        brain=frozen_brain(brain_source,torch.device(a.device));del brain_source['weights']
        parts=[]
        with torch.inference_mode():
            for start in range(0,12,4):
                mu=brain.encode(torch.from_numpy(values[start:start+4]).to(a.device)).mode().float()
                if tuple(mu.shape[1:])!=(256,1664) or not torch.isfinite(mu).all():raise ValueError('Invalid BrainVAE mean')
                parts.append(mu.cpu().numpy())
        array=np.concatenate(parts);tmp=target.with_suffix('.npy.tmp')
        with tmp.open('wb') as f:np.save(f,array,allow_pickle=False)
        os.replace(tmp,target)
        atomic_json({'status':'complete','subject':a.subject,'dataset':'vision','representation':'raw_unadapted_vision_mean_then_brainvae_mean',
                     'mu_sha256':file_sha256(target),'shape':list(array.shape),'rows':rows,'brainvae_sha256':ident['brainvae_sha256'],
                     'data_fingerprint':ident['data_fingerprint'],'trials_sha256':ident['imagery_trials_sha256'],
                     'ridge_model_metadata_sha256':file_sha256(model),'test_images_or_text_used_as_input':False},manifest)
        print('[complete]',target,flush=True)
if __name__=='__main__':main()
