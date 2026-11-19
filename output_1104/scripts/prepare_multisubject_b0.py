#!/usr/bin/env python3
"""Prepare three held-out imagery B0 samples without training an adapter."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from functools import partial
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
import yaml
from torchvision.utils import save_image
import run_multisubject_imagery_natural as natural
import prepare_subj01_imagery_three_samples as previous
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import load_mapping_config, mapping_lock
from src.utils.staged_queue import atomic_json
from multisubject_generate import generate_samples_verified


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def paths(subject):
    root=ROOT/f'output_1104/multisubject_nsd_imagery_v1/subj{subject:02d}'
    return root,root/'imagery_b0_initial_3samples_v1'


def immutable(path,value):
    path=Path(path)
    if path.exists() and read(path)!=value:raise ValueError('Existing B0 identity differs: '+str(path))
    if not path.exists():atomic_json(value,path)


def audit_existing_subj01():
    mu=read(ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs/imagery_mu_manifest.json')
    expected=[(r['stimulus_id'],r['source_trial_ids']) for r in mu['rows']]
    ref=previous.old.OUT/'refined/B0/imagery'
    info=read(ref/'metadata.json'); manifest=read(ref/'sample00/manifest.json')
    if len(expected)!=12 or len(manifest['items'])!=12:raise ValueError('Expected 12 subj01 imagery stimuli')
    for sample in range(3):
        rows=previous.original('B0',sample)
        if [(r['stimulus_id'],r['source_trial_ids']) for r in rows]!=expected:
            raise ValueError('Existing subj01 B0 trials differ')
        if sample==0:
            for index,row in enumerate(rows):
                item=manifest['items'][index];image=ref/'sample00'/item['path']
                if (info['provenance']['input_images'][row['stimulus_id']]!=row['input_sha256']
                    or item['stimulus_id']!=row['stimulus_id'] or item['source_trial_ids']!=row['source_trial_ids']
                    or item['seed']!=42+index or read(image.with_suffix('.json'))['sha256']!=file_sha256(image)):
                    raise ValueError('Existing subj01 B0 refinement provenance differs')
    print('[audit] subj01 B0 initial samples 00-02 and refined sample00 verified',flush=True)


def reuse_subj01():
    audit_existing_subj01()
    _root,out=paths(1);folder=out/'initial/B0/imagery';folder.mkdir(parents=True,exist_ok=True)
    _cfg,_source,identity=previous.old.source()
    meta=previous.metadata('B0',identity,identity['w010_sha256'])
    immutable(folder/'metadata.json',meta)
    hashes=[]
    for sample in range(3):
        rows=previous.original('B0',sample);sample_dir=folder/f'sample{sample:02d}';sample_dir.mkdir(parents=True,exist_ok=True)
        items=[{**{k:r[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')},
                'item_index':i,'sample_index':sample,'seed':42+sample*100000+i,'path':r['input_path'],
                'source_sha256':r['input_sha256']} for i,r in enumerate(rows)]
        immutable(sample_dir/'manifest.json',dict(meta,level='stimulus_level',sample_index=sample,n_images=12,items=items))
        hashes.append({r['stimulus_id']:r['input_sha256'] for r in rows})
    atomic_json({'status':'complete','methods':['B0'],'samples':3,'targets':12,
                 'image_sha256':{'B0':hashes},'policy':'verified reuse of existing subj01 B0 samples'},out/'generation_complete.json')


def predict(subject,device):
    root,out=paths(subject);cfg=yaml.safe_load((root/'config_imagery_natural_k60_3samples.yaml').read_text())
    source=natural.run_source(cfg);exp=ROOT/cfg['experiment_root'];identity=natural.require_identity(cfg,exp,source)
    mu_dir=root/'imagery_ridge_inputs_3samples_v1';info=read(mu_dir/'imagery_mu_manifest.json');mu_path=mu_dir/'imagery_mu.npy'
    mu=np.load(mu_path,mmap_mode='r',allow_pickle=False);rows=info['rows']
    if (info.get('status')!='complete' or info['subject']!=subject or info['mu_sha256']!=file_sha256(mu_path)
        or info['brainvae_sha256']!=identity['brainvae_sha256'] or info['data_fingerprint']!=identity['data_fingerprint']
        or mu.shape!=(12,256,1664) or mu.dtype!=np.float32 or not np.isfinite(mu).all()
        or len(rows)!=12 or [r['group'] for r in rows]!=['natural']*6+['geometry']*6):
        raise ValueError('B0 source means differ from subject identity')
    _head,_brain_cfg,project=load_mapping_config(source['config_path'],ROOT)
    target_root=project.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages'
    records=[];target_index={}
    for index,row in enumerate(rows):
        target=target_root/('setB' if row['group']=='natural' else 'setA')/(row['stimulus_id']+'.png')
        if row['row_index']!=index or not target.is_file() or len(row['source_trial_ids'])!=4:
            raise ValueError('Invalid held-out imagery target')
        records.append({'subject_id':subject,'stimulus_id':row['stimulus_id'],'stimulus_type':row['stimulus_type'],
                        'task_type':'imagery','n_trials':4,'source_trial_ids':row['source_trial_ids'],'runs':row['runs']})
        target_index[row['stimulus_id']]=str(target)
    brain,mapper,_=natural.load_frozen(cfg,source,device,sdpa=False);del brain
    with torch.inference_mode():
        values=torch.cat([natural.map_tokens(mapper,torch.from_numpy(np.array(mu[i:i+2])).to(device)).float().cpu()
                          for i in range(0,12,2)])
    if values.shape!=(12,256,1664) or not torch.isfinite(values).all():raise ValueError('Invalid B0 tokens')
    folder=out/'initial/B0/imagery';token=folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
    payload={'mean':values,'records':records,'representation':'mapped_raw_clip_tokens',
             'mapping_checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':None}
    if token.exists():
        old=torch.load(token,map_location='cpu',weights_only=False)
        if old['records']!=records or old['mapping_checkpoint_sha256']!=identity['w010_sha256'] or not torch.equal(old['mean'],values):
            raise ValueError('Existing B0 tokens differ')
    else:natural.atomic_checkpoint(payload,token)
    meta={'protocol':'multisubject_b0_imagery_initial_3samples_v1','subject':subject,'dataset':'imagery','method':'B0',
          'aggregation':'mean_fmri','checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':None,
          'w010_sha256':identity['w010_sha256'],'brainvae_sha256':identity['brainvae_sha256'],
          'data_fingerprint':identity['data_fingerprint'],'num_samples':3,'num_steps':38,'seed':42,
          'partial_test':False,'debug_training':False,'n_items':12,'token_sha256':file_sha256(token),
          'mu_sha256':info['mu_sha256'],'target_usage':'evaluation_only','geometry':'direct B0'}
    immutable(folder/'metadata.json',meta);immutable(out/'targets.json',target_index)
    atomic_json({'status':'complete','methods':['B0'],'token_sha256':file_sha256(token)},out/'prediction_complete.json')
    del mapper;gc.collect();torch.cuda.empty_cache()


def generate(subject,device):
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    root,out=paths(subject);cfg=yaml.safe_load((root/'config_imagery_natural_k60_3samples.yaml').read_text())
    source=natural.run_source(cfg);_head,_brain_cfg,project=load_mapping_config(source['config_path'],ROOT)
    folder=out/'initial/B0/imagery';meta=read(folder/'metadata.json');token=folder/'tokens/chunk00000.pt'
    if file_sha256(token)!=meta['token_sha256']:raise ValueError('B0 token digest mismatch')
    targets={name:Path(path) for name,path in read(out/'targets.json').items()}
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute():unclip=ROOT/unclip
    engine,suffix,offset=load_sdxl_unclip(project.generative_models_path,unclip,project.openclip_version,device,38)
    generate_samples_verified([token],folder,meta,engine,suffix,offset,partial(unclip_decode_batch,rgb_mode='standard'),
                              save_image,3,42,targets)
    hashes=[]
    for sample in range(3):
        sample_dir=folder/f'sample{sample:02d}';manifest=read(sample_dir/'manifest.json')
        progress={int(r['index']):r for r in map(json.loads,(sample_dir/'progress.jsonl').read_text().splitlines())}
        if len(manifest['items'])!=12 or set(progress)!=set(range(12)):raise ValueError('Incomplete B0 samples')
        group={}
        for i,item in enumerate(manifest['items']):
            image=sample_dir/item['path'];digest=file_sha256(image)
            if digest!=progress[i]['sha256'] or item['seed']!=42+sample*100000+i:raise ValueError('B0 progress mismatch')
            group[item['stimulus_id']]=digest
        hashes.append(group)
    atomic_json({'status':'complete','methods':['B0'],'samples':3,'targets':12,
                 'image_sha256':{'B0':hashes},'policy':'subject-specific B0 from held-out imagery mean'},out/'generation_complete.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',type=int,choices=range(1,9),required=True)
    p.add_argument('--stage',choices=('audit','reuse','predict','generate'),required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.set_num_threads(4)
    if args.stage=='audit':
        if args.subject!=1:raise ValueError('Audit reuse applies to subj01')
        audit_existing_subj01();return
    _root,out=paths(args.subject)
    with mapping_lock(out):
        if args.stage=='reuse':
            if args.subject!=1:raise ValueError('Reuse applies to subj01')
            reuse_subj01();return
        device=torch.device(args.device)
        if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if args.stage=='predict':predict(args.subject,device)
        else:generate(args.subject,device)


if __name__=='__main__':main()
