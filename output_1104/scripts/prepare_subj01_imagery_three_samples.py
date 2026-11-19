#!/usr/bin/env python3
"""Preserve subj01 K60 imagery samples and fill missing initial reconstructions."""
from __future__ import annotations
import argparse
import gc
import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
from torchvision.utils import save_image
import prepare_imagery_r_k60 as setup_k60
import run_k60_vision_imagery as old
import run_imagery_w010_pilot as pilot
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import load_mapping_config
from src.utils.staged_queue import atomic_json

OUT=ROOT/'output_1104/multisubject_nsd_imagery_v1/subj01/imagery_initial_k60_3samples_v1'
METHODS=('B0','R','V','R_MSE')


def read(path):return json.loads(Path(path).read_text())


def immutable(path,value):
    if path.exists() and read(path)!=value:raise ValueError('Existing protocol differs: '+str(path))
    if not path.exists():atomic_json(value,path)


def original(method,sample):
    paths=[old.PILOT/'runs'/method/'evaluation/imagery'/f'sample{sample:02d}'/'manifest.json',
           old.TRANSFER/'imagery_geometry'/method/'evaluation/imagery'/f'sample{sample:02d}'/'manifest.json']
    rows=[];source_identity=None
    for path in paths:
        manifest=read(path)
        if manifest['subject']!=1 or manifest['method']!=method or manifest['sample_index']!=sample:
            raise ValueError('Existing initial sample metadata mismatch')
        identity=(manifest['checkpoint_sha256'],manifest['w010_sha256'],manifest['brainvae_sha256'],manifest.get('adapter_sha256'))
        if source_identity is not None and source_identity!=identity:raise ValueError('Natural/geometry source mismatch')
        source_identity=identity
        for item in manifest['items']:
            if item['seed']!=42+sample*100000+len(rows):
                raise ValueError('Existing initial sample seed/order mismatch')
            image=(path.parent/item['path']).resolve()
            if not image.is_file():raise FileNotFoundError(image)
            rows.append({**item,'input_path':str(image),'input_sha256':file_sha256(image)})
    return rows


def metadata(method,identity,digest,token=None):
    return {'protocol':'subj01_k60_imagery_initial_3samples_v1','subject':1,'dataset':'imagery','method':method,
            'aggregation':'mean_fmri','checkpoint_sha256':digest,'adapter_sha256':None if method=='B0' else digest,
            'w010_sha256':identity['w010_sha256'],'brainvae_sha256':identity['brainvae_sha256'],
            'data_fingerprint':identity['data_fingerprint'],'num_samples':3,'num_steps':38,'seed':42,
            'partial_test':False,'debug_training':False,'n_items':12,'token_sha256':token,
            'target_usage':'target images are used only for evaluation after target-conditioned R training',
            'geometry':'natural-adapter transfer'}


def audit_existing():
    for method in ('B0','V'):
        for sample in range(3):
            rows=original(method,sample)
            if len(rows)!=12:raise ValueError('Existing subj01 initial sample must contain 12 stimuli')
            if sample==0:
                refined=read(old.OUT/'refined'/method/'imagery/metadata.json')
                if any(refined['provenance']['input_images'][row['stimulus_id']]!=row['input_sha256'] for row in rows):
                    raise ValueError('Existing subj01 refined inputs changed')
    r=old.OUT/'initial/R/imagery';manifest=read(r/'sample00/manifest.json')
    refined=read(old.OUT/'refined/R/imagery/metadata.json')
    if len(manifest['items'])!=12:raise ValueError('Existing K60 R sample is incomplete')
    for item in manifest['items']:
        if file_sha256(r/'sample00'/item['path'])!=refined['provenance']['input_images'][item['stimulus_id']]:
            raise ValueError('Existing K60 R input image changed')
    for method in ('B0','R','V'):
        folder=old.OUT/'refined'/method/'imagery';manifest=read(folder/'sample00/manifest.json')
        if len(manifest['items'])!=12:raise ValueError('Existing refined sample is incomplete')
        for item in manifest['items']:
            image=folder/'sample00'/item['path'];receipt=image.with_suffix('.json')
            if read(receipt)['sha256']!=file_sha256(image):raise ValueError('Existing refined sample changed')
    print('[audit] subj01 B0/V initial samples 00–02 and K60 R initial/refined sample00 verified',flush=True)


def train(device):
    cfg,source=setup_k60.prepare()
    pilot.train_adapter(cfg,setup_k60.OUT,source,device,'R_MSE')


def predict(device):
    cfg,source,identity=old.source(); records,values=old.task_values(source)['imagery']
    if len(records)!=12:raise ValueError('Expected 12 held-out imagery stimuli')
    OUT.mkdir(parents=True,exist_ok=True)
    # B0/V already have three original initial images: record and reuse them.
    for method in ('B0','V'):
        sample0=original(method,0)
        if [(r['stimulus_id'],r['source_trial_ids']) for r in sample0]!=[(r['stimulus_id'],r['source_trial_ids']) for r in records]:
            raise ValueError('Existing baseline trials differ')
        src_meta=read(old.PILOT/'runs'/method/'evaluation/imagery/metadata.json')
        digest=src_meta['checkpoint_sha256']
        folder=OUT/'initial'/method/'imagery';folder.mkdir(parents=True,exist_ok=True)
        meta=metadata(method,identity,digest)
        immutable(folder/'metadata.json',meta)
        for sample in range(3):
            rows=original(method,sample)
            if [(r['stimulus_id'],r['source_trial_ids']) for r in rows]!=[(r['stimulus_id'],r['source_trial_ids']) for r in records]:
                raise ValueError('Existing sample stimulus/trial order differs')
            # The source sample itself remains immutable; use absolute paths.
            items=[{**{k:r[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')},
                    'item_index':i,'sample_index':sample,'seed':42+sample*100000+i,'path':r['input_path'],
                    'source_sha256':r['input_sha256']} for i,r in enumerate(rows)]
            immutable(folder/f'sample{sample:02d}'/'manifest.json',dict(meta,level='stimulus_level',sample_index=sample,n_images=12,items=items))
    # R already has K60 text-independent tokens and one initial sample.
    r_old=old.OUT/'initial/R/imagery';r_meta=read(r_old/'metadata.json');r_token=r_old/'tokens/chunk00000.pt'
    if file_sha256(r_token)!=r_meta['token_sha256']:raise ValueError('K60 R token checksum mismatch')
    r_complete=read(setup_k60.OUT/'runs/R/training_complete.json')
    if r_complete['best_sha256']!=r_meta['adapter_sha256']:raise ValueError('K60 R adapter mismatch')
    r_folder=OUT/'initial/R/imagery';token=r_folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
    if token.exists() and file_sha256(token)!=file_sha256(r_token):raise ValueError('Existing R token differs')
    if not token.exists():shutil.copyfile(r_token,token)
    immutable(r_folder/'metadata.json',metadata('R',identity,r_meta['adapter_sha256'],file_sha256(token)))
    # R_MSE must be K60; the older K1000 branch is not substituted.
    folder=setup_k60.OUT/'runs/R_MSE';complete=read(folder/'training_complete.json');ckpt=folder/'best.pt';digest=file_sha256(ckpt)
    if complete.get('status')!='complete' or complete['best_sha256']!=digest or complete['signature']['method']!='R_MSE':
        raise ValueError('K60 R_MSE training incomplete')
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    if saved['signature']!=complete['signature']:raise ValueError('R_MSE checkpoint identity mismatch')
    brain,mapper,_=pilot.load_frozen(cfg,source,device,sdpa=False);mapping=pilot.FrozenMapping(brain,mapper)
    adapter=pilot.make_adapter(cfg,values.shape[-1],device).eval().requires_grad_(False)
    adapter.load_state_dict(saved['adapter'],strict=True)
    with torch.inference_mode():
        x=torch.from_numpy(values).to(device)
        pred=torch.cat([mapping(adapter(chunk)) for chunk in x.split(2)]).float().cpu()
    if pred.shape!=(12,256,1664) or not torch.isfinite(pred).all():raise FloatingPointError('Invalid R_MSE tokens')
    m_folder=OUT/'initial/R_MSE/imagery';token=m_folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
    payload={'mean':pred,'records':records,'representation':'mapped_raw_clip_tokens',
             'mapping_checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':digest}
    if token.exists():
        previous=torch.load(token,map_location='cpu',weights_only=False)
        if previous['records']!=records or previous['adapter_sha256']!=digest or not torch.equal(previous['mean'],pred):
            raise ValueError('Existing R_MSE tokens differ')
    else:pilot.atomic_checkpoint(payload,token)
    immutable(m_folder/'metadata.json',metadata('R_MSE',identity,digest,file_sha256(token)))
    del mapping,brain,mapper,adapter;gc.collect();torch.cuda.empty_cache()
    atomic_json({'status':'complete','methods':list(METHODS)},OUT/'prediction_complete.json')


def generate(device):
    from multisubject_generate import generate_samples_verified
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    cfg,source,_=old.source();_,_,paths=load_mapping_config(source['config_path'],ROOT)
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute():unclip=ROOT/unclip
    # Preserve the already generated R sample00. Verify it against the old
    # refinement's recorded input hashes before copying into the new run.
    folder=OUT/'initial/R/imagery';old_folder=old.OUT/'initial/R/imagery'
    old_manifest=read(old_folder/'sample00/manifest.json')
    old_refined=read(old.OUT/'refined/R/imagery/metadata.json')
    expected=old_refined['provenance']['input_images']
    token_records=torch.load(folder/'tokens/chunk00000.pt',map_location='cpu',weights_only=False)['records']
    if (len(token_records)!=len(old_manifest['items']) or
            [(r['stimulus_id'],r['source_trial_ids']) for r in token_records] !=
            [(r['stimulus_id'],r['source_trial_ids']) for r in old_manifest['items']]):
        raise ValueError('Existing K60 R image and token identities differ')
    reuse={}
    for index,item in enumerate(old_manifest['items']):
        src=old_folder/'sample00'/item['path']
        if item['seed']!=42+index:raise ValueError('Existing K60 R seed mismatch')
        if file_sha256(src)!=expected[item['stimulus_id']]:raise ValueError('Existing R input image changed')
        reuse[(0,index)]=(src,expected[item['stimulus_id']])
    target={row['stimulus_id']:Path(row['target_image']) for row in old.original_rows('imagery','B0')}
    engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
    for method in ('R','R_MSE'):
        folder=OUT/'initial'/method/'imagery';meta=read(folder/'metadata.json');token=folder/'tokens/chunk00000.pt'
        if file_sha256(token)!=meta['token_sha256']:raise ValueError('Token checksum mismatch')
        generate_samples_verified([token],folder,meta,engine,suffix,offset,partial(unclip_decode_batch,rgb_mode='standard'),
                         save_image,3,42,target,reuse=reuse if method=='R' else None)
    del engine;gc.collect();torch.cuda.empty_cache()
    hashes={}
    for method in METHODS:
        folder=OUT/'initial'/method/'imagery'
        hashes[method]=[]
        for sample in range(3):
            p=folder/f'sample{sample:02d}'/'manifest.json';manifest=read(p)
            if len(manifest['items'])!=12 or any(not (p.parent/x['path']).is_file() for x in manifest['items']):
                raise ValueError('Initial images incomplete')
            hashes[method].append({x['stimulus_id']:file_sha256(p.parent/x['path']) for x in manifest['items']})
    atomic_json({'status':'complete','methods':list(METHODS),'samples':3,'targets':12,
                 'image_sha256':hashes,
                 'policy':'reuse existing B0/V samples and K60 R sample00; K60 R_MSE is newly trained'},OUT/'generation_complete.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('audit','train_R_MSE','predict','generate'),required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    if args.stage=='audit':audit_existing();return
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    if args.stage=='train_R_MSE':train(device)
    elif args.stage=='predict':predict(device)
    else:generate(device)


if __name__=='__main__':main()
