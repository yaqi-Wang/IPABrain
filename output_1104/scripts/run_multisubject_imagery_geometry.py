#!/usr/bin/env python3
"""Transfer each subject's natural R/V/R-MSE adapters to held-out geometry imagery."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
import yaml
import run_multisubject_imagery_natural as natural
from src.data.scheme1 import file_sha256
from src.utils.imagery_w010_pilot import split_trials
from src.utils.staged_queue import atomic_json

METHODS=('B0','R','V','R_MSE')


def setup(subject):
    base=ROOT/f'output_1104/multisubject_nsd_imagery_v1/subj{subject:02d}'
    cfg=yaml.safe_load((base/'config_imagery_natural_k60_3samples.yaml').read_text())
    exp=ROOT/cfg['experiment_root']
    source=natural.run_source(cfg)
    identity=natural.require_identity(cfg,exp,source)
    lock=exp/'protocol/code_lock.json'
    if not lock.is_file() or json.loads(lock.read_text())['files']!=natural.code_fingerprint():
        raise ValueError('Natural training code identity differs')
    data=natural.load_trials(source)
    split=split_trials(data,cfg['split_seed'],'simple')
    if split['subject']!=subject:raise ValueError('Geometry split subject mismatch')
    inputs=[];records=[]
    for name in split['targets']:
        part=split['rows'][name];rows=part['test']
        if len(rows)!=4 or set(rows)&(set(part['train'])|set(part['val'])):
            raise ValueError('Invalid held-out geometry trials')
        inputs.append(np.asarray(data['fmri_scaled'][rows],dtype=np.float32).mean(axis=0))
        records.append({'subject_id':subject,'stimulus_id':name,'stimulus_type':'simple',
                        'task_type':'imagery','n_trials':4,
                        'source_trial_ids':[int(data['global_trial_idx'][i]) for i in rows],
                        'runs':[int(data['run'][i]) for i in rows]})
    out=base/'imagery_geometry_transfer_k60_3samples_v1'
    return cfg,exp,source,identity,split,np.stack(inputs),records,out


def predict(cfg,exp,source,identity,split,values,records,out,device):
    out.mkdir(parents=True,exist_ok=True)
    protocol={'subject':cfg['subject'],'geometry_split':split,'natural_identity_sha256':file_sha256(exp/'protocol/identity.json'),
              'natural_code_lock_sha256':file_sha256(exp/'protocol/code_lock.json'),
              'role':'natural-adapter transfer to held-out geometry imagery',
              'script_sha256':file_sha256(Path(__file__))}
    p=out/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Geometry protocol differs')
    if not p.exists():atomic_json(protocol,p)
    brain,mapper,_=natural.load_frozen(cfg,source,device,sdpa=False)
    mapping=natural.FrozenMapping(brain,mapper)
    x=torch.from_numpy(values.astype(np.float32)).to(device)
    for method in METHODS:
        folder=out/'runs'/method/'evaluation/imagery'
        token=folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
        adapter=sha=None
        if method!='B0':
            cp=exp/'runs'/method/'best.pt'
            complete=json.loads((cp.parent/'training_complete.json').read_text())
            sha=file_sha256(cp)
            if sha!=complete['best_sha256']:raise ValueError('Adapter checkpoint checksum differs')
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            if saved['signature']!=complete['signature']:raise ValueError('Adapter signature differs')
            adapter=natural.make_adapter(cfg,x.shape[-1],device).eval().requires_grad_(False)
            adapter.load_state_dict(saved['adapter'],strict=True)
        with torch.inference_mode():
            adapted=x if adapter is None else adapter(x)
            value=torch.cat([mapping(chunk) for chunk in adapted.split(2)]).float().cpu()
        if value.shape!=(6,256,1664) or not torch.isfinite(value).all():raise FloatingPointError('Invalid predicted tokens')
        payload={'mean':value,'records':records,'representation':'mapped_raw_clip_tokens',
                 'mapping_checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':sha}
        if token.exists():
            old=torch.load(token,map_location='cpu',weights_only=False)
            if not torch.equal(old['mean'],value) or old['records']!=records or old['adapter_sha256']!=sha:
                raise ValueError('Existing token predictions differ')
        else:natural.atomic_checkpoint(payload,token)
        meta={'protocol':'multisubject_natural_to_geometry_transfer_v1','subject':cfg['subject'],'dataset':'imagery',
              'method':method,'checkpoint_sha256':sha or identity['w010_sha256'],'adapter_sha256':sha,
              'w010_sha256':identity['w010_sha256'],'brainvae_sha256':identity['brainvae_sha256'],
              'data_fingerprint':identity['data_fingerprint'],'aggregation':'mean_fmri','num_samples':3,
              'num_steps':38,'seed':48,'rgb_mode':'standard','n_items':6,'partial_test':False,
              'debug_training':False,'token_sha256':file_sha256(token),'protocol_sha256':file_sha256(p),
              'target_usage':'reference images attached after fMRI prediction for evaluation',
              'evaluation_role':'natural_adapter_transfer'}
        m=folder/'metadata.json'
        if m.exists() and json.loads(m.read_text())!=meta:raise ValueError('Geometry metadata differs')
        if not m.exists():atomic_json(meta,m)
        print(f'[predict] geometry {method}',flush=True)
    atomic_json({'status':'complete','methods':list(METHODS)},out/'prediction_complete.json')
    del mapping,brain,mapper
    gc.collect();torch.cuda.empty_cache()


def generate(cfg,source,out,device):
    from torchvision.utils import save_image
    from src.utils.clip_mapping import load_mapping_config
    from multisubject_generate import generate_samples_verified
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    _,_,paths=load_mapping_config(source['config_path'],ROOT)
    target_root=paths.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages/setA'
    targets={p.stem:p for p in target_root.glob('*.png')}
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute():unclip=ROOT/unclip
    engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
    for method in METHODS:
        folder=out/'runs'/method/'evaluation/imagery'
        meta=json.loads((folder/'metadata.json').read_text())
        token=folder/'tokens/chunk00000.pt'
        if file_sha256(token)!=meta['token_sha256']:raise ValueError('Token checksum differs')
        generate_samples_verified([token],folder,meta,engine,suffix,offset,
                         partial(unclip_decode_batch,rgb_mode='standard'),save_image,3,meta['seed'],targets)
        for sample in range(3):
            sample_dir=folder/f'sample{sample:02d}'
            manifest=json.loads((sample_dir/'manifest.json').read_text())
            ledger={int(x['index']):x for x in map(json.loads,(sample_dir/'progress.jsonl').read_text().splitlines())}
            if (len(manifest['items'])!=6 or any(not (sample_dir/x['path']).is_file() or
                    x['seed']!=meta['seed']+sample*100000+i or i not in ledger or
                    ledger[i]['sha256']!=file_sha256(sample_dir/x['path']) for i,x in enumerate(manifest['items']))):
                raise ValueError('Geometry generation incomplete')
    atomic_json({'status':'complete','images':72,'methods':list(METHODS)},out/'generation_complete.json')
    del engine;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,choices=range(2,9),required=True)
    p.add_argument('--stage',choices=('predict','generate'),required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    cfg,exp,source,identity,split,values,records,out=setup(args.subject)
    if args.stage=='predict':predict(cfg,exp,source,identity,split,values,records,out,torch.device(args.device))
    else:generate(cfg,source,out,torch.device(args.device))


if __name__=='__main__':main()
