#!/usr/bin/env python3
"""Refine matching vision or imagery samples with subject-specific three-text Ridge."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
import refine_w010_mindeye2_img2img as base
import sweep_subj01_retrieval_k as sweep
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import mapping_lock
from src.utils.staged_queue import atomic_json
SHAPES={'clip_l':(77,768),'bigg_tokens':(77,1280),'bigg_pooled':(1280,)}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def immutable(path,value):
    if path.exists() and read(path)!=value:raise ValueError(f'Existing provenance changed: {path}')
    if not path.exists():atomic_json(value,path)

def checked(source,pred,mu_manifest,subject,task):
    source=source.resolve();pred=pred.resolve();mu_manifest=mu_manifest.resolve()
    info=read(source/'metadata.json');condition=read(pred/'prediction_complete.json');mu=read(mu_manifest)
    if info.get('subject')!=subject or info.get('dataset')!=task or info.get('n_items')!=12 or info.get('num_samples')!=3 or info.get('partial_test') or info.get('debug_training'):
        raise ValueError('Unexpected initial image metadata')
    if condition.get('status')!='complete' or condition.get('test_text_used_as_input') is not False:
        raise ValueError('Invalid Ridge predictions')
    if mu.get('status')!='complete' or mu.get('dataset',task)!=task or mu.get('subject',subject)!=subject:
        raise ValueError('Invalid subject or task for Ridge inputs')
    if condition['input']['sha256']!=mu['mu_sha256']:raise ValueError('Ridge input is not the recorded mean')
    if file_sha256(Path(condition['input']['path']))!=mu['mu_sha256']:raise ValueError('Ridge input was modified')
    if condition['input'].get('manifest_sha256') and condition['input']['manifest_sha256']!=file_sha256(mu_manifest):
        raise ValueError('Ridge input manifest was modified')
    if mu.get('ridge_model_metadata_sha256') and condition.get('model_metadata_sha256')!=mu['ridge_model_metadata_sha256']:
        raise ValueError('Ridge prediction model does not match the input manifest')
    for key in ('brainvae_sha256','data_fingerprint'):
        if info.get(key) and mu.get(key) and info[key]!=mu[key]:raise ValueError('Ridge source differs in '+key)
    rows=mu.get('rows',mu.get('records'))
    if len(rows)!=12:raise ValueError('Expected 12 Ridge inputs')
    arrays={}
    for key,shape in SHAPES.items():
        path=pred/f'predicted_{key}.npy';x=np.load(path,mmap_mode='r')
        if x.shape!=(12,*shape) or not np.isfinite(x).all() or file_sha256(path)!=condition['prediction_sha256'][key]:raise ValueError('Invalid '+key)
        arrays[key]=x
    inputs=[]
    for sample in range(3):
        folder=source/f'sample{sample:02d}'
        if not sweep.checked_sample(source,sample):raise ValueError(f'Unverified initial images sample {sample}')
        items=read(folder/'manifest.json')['items']
        for i,(item,row) in enumerate(zip(items,rows)):
            if item['stimulus_id']!=row['stimulus_id'] or item['stimulus_type']!=row['stimulus_type'] or item['source_trial_ids']!=row['source_trial_ids'] or item['seed']!=42+sample*100000+i:
                raise ValueError(f'Text/trial/seed mismatch sample={sample} item={i}')
        inputs.append(items)
    if file_sha256(base.ZAVY)!=base.ZAVY_SHA:raise ValueError('Zavy weights mismatch')
    identity={'protocol':'multisubject_two_conditions_three_text_v1','subject':subject,'dataset':task,
              'input_metadata_sha256':file_sha256(source/'metadata.json'),
              'input_manifest_sha256':[file_sha256(source/f'sample{s:02d}'/'manifest.json') for s in range(3)],
              'input_images_sha256':[[file_sha256(source/f'sample{s:02d}'/item['path']) for item in inputs[s]] for s in range(3)],
              'ridge_prediction_metadata_sha256':file_sha256(pred/'prediction_complete.json'),
              'ridge_prediction_sha256':condition['prediction_sha256'],'ridge_input_manifest_sha256':file_sha256(mu_manifest),
              'brainvae_sha256':info['brainvae_sha256'],'checkpoint_sha256':info['checkpoint_sha256'],
              'data_fingerprint':info['data_fingerprint'],'zavy_sha256':base.ZAVY_SHA,
              'samples':3,'targets':12,'steps':38,'timepoint':13,'cfg_scale':5.,'seed':'42+100000*sample+stimulus_index',
              'text_condition':'NSD-trained three-branch Ridge from unadapted task fMRI','target_image_used_for_generation':False,
              'script_sha256':file_sha256(Path(__file__)),'base_sha256':file_sha256(Path(base.__file__))}
    return info,arrays,inputs,identity

def reuse_sample00(identity_path,folder,identity,inputs,task):
    old=read(identity_path);meta=read(folder/'metadata.json')
    if (old['zavy_sha256']!=identity['zavy_sha256'] or old['img2img_steps']!=38 or old['timepoint']!=13
            or old['cfg_scale']!=5. or old['seed']!='42 + stimulus_index'
            or old['ridge_predictions'][task]['prediction_sha256']!=identity['ridge_prediction_sha256']
            or old['input_images'][f'linear_{task}']['input_image_sha256']!=identity['input_images_sha256'][0]
            or meta['checkpoint_sha256']!=identity['checkpoint_sha256'] or not sweep.checked_sample(folder,0)):
        raise ValueError('Existing sample00 refinement is incompatible')
    items=read(folder/'sample00/manifest.json')['items']
    for i,(old_item,new_item) in enumerate(zip(items,inputs[0])):
        if any(old_item[k]!=new_item[k] for k in ('stimulus_id','stimulus_type','source_trial_ids','seed')):
            raise ValueError(f'Existing sample00 refinement mismatch at {i}')
    paths=[folder/'sample00'/r['path'] for r in items]
    return paths,{'identity_sha256':file_sha256(identity_path),'manifest_sha256':file_sha256(folder/'sample00/manifest.json'),
                  'image_sha256':[file_sha256(p) for p in paths]}

def refine(source,pred,mu_manifest,out,subject,task,device,reuse_dir=None,reuse_identity=None):
    meta,arrays,inputs,ident=checked(source,pred,mu_manifest,subject,task)
    reused=None
    if reuse_dir is not None:
        reused,reuse_info=reuse_sample00(reuse_identity,reuse_dir,ident,inputs,task)
        ident['reused_refined_sample00']=reuse_info
    immutable(out/'identity.json',ident)
    negcross,negpool,audit=base.negative_condition(device);immutable(out/'negative_condition_audit.json',audit)
    engine,suffix=base.load_engine(device,38,5.)
    metadata={'protocol':ident['protocol'],'subject':subject,'dataset':task,'method':meta['method'],
              'aggregation':'mean_fmri','n_items':12,'num_samples':3,'partial_test':False,'debug_training':False,
              'checkpoint_sha256':ident['checkpoint_sha256'],'brainvae_sha256':ident['brainvae_sha256'],
              'data_fingerprint':ident['data_fingerprint'],'generator_backend':'zavy_sdxl_img2img',
              'text_condition':'nsd_ridge_three_branch_unadapted_fmri','text_oracle':False,'oracle':False,
              'target_usage':'target image used during adapter training; not used during generation',
              'source_identity_sha256':base.digest_json(ident)}
    immutable(out/'metadata.json',metadata)
    for sample in range(3):
        dest=out/f'sample{sample:02d}';(dest/'images').mkdir(parents=True,exist_ok=True)
        outputs=[]
        for i,row in enumerate(inputs[sample]):
            keys=('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')
            outputs.append({**{k:row[k] for k in keys},'item_index':i,'sample_index':sample,'seed':42+sample*100000+i,'path':f'images/item{i:05d}.png'})
        immutable(dest/'manifest.json',{**metadata,'level':'stimulus_level','sample_index':sample,'n_images':12,'items':outputs})
        ledger=dest/'progress.jsonl';completed={}
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                rec=json.loads(line);i=int(rec['index'])
                if not 0<=i<12 or rec['seed']!=outputs[i]['seed'] or rec['stimulus_id']!=outputs[i]['stimulus_id']:raise ValueError('Invalid progress')
                completed[i]=rec
        with ledger.open('a',encoding='utf-8') as progress:
            for i,row in enumerate(inputs[sample]):
                target=dest/outputs[i]['path'];rec=completed.get(i)
                if rec:
                    if not target.is_file() or file_sha256(target)!=rec['sha256']:raise ValueError('Existing refined image changed')
                    continue
                if target.exists():
                    from multisubject_resume import archive_file
                    print('[resume] preserving unverified image:',archive_file(target),flush=True)
                image_path=source/f'sample{sample:02d}'/row['path']
                if file_sha256(image_path)!=ident['input_images_sha256'][sample][i]:raise ValueError('Initial image changed')
                temp=target.with_name(target.stem+'.tmp.png')
                if sample==0 and reused is not None:
                    if file_sha256(reused[i])!=ident['reused_refined_sample00']['image_sha256'][i]:raise ValueError('Reused refinement changed')
                    shutil.copyfile(reused[i],temp)
                else:
                    with Image.open(image_path) as im:
                        tensor=pil_to_tensor(ImageOps.exif_transpose(im).convert('RGB')).float().div_(255).unsqueeze(0).to(device)
                    cond,uncond=base.conditions(arrays,i,suffix,negcross,negpool,device)
                    image=base.img2img(engine,tensor,cond,uncond,outputs[i]['seed'],38,13)
                    save_image(image.cpu(),str(temp))
                os.replace(temp,target)
                progress.write(json.dumps({'index':i,'stimulus_id':row['stimulus_id'],'seed':outputs[i]['seed'],'sha256':file_sha256(target)},ensure_ascii=False)+'\n');progress.flush()
                print(f'[精修] subj{subject:02d} {task} sample{sample:02d} {i+1}/12',flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()
    if not all(sweep.checked_sample(out,s) for s in range(3)):raise ValueError('Refinement incomplete')
    atomic_json({'status':'complete','subject':subject,'task':task,'images':36,'identity_sha256':base.digest_json(ident)},out/'refinement_complete.json')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','prediction','mu-manifest','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--subject',type=int,choices=range(1,9),required=True);p.add_argument('--task',choices=('vision','imagery'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--reuse-dir',type=Path);p.add_argument('--reuse-identity',type=Path);a=p.parse_args()
    if (a.reuse_dir is None)!=(a.reuse_identity is None):p.error('Both reuse paths are required')
    source=a.source.resolve();pred=a.prediction.resolve();mu=a.mu_manifest.resolve();out=a.output.resolve()
    info,arrays,rows,ident=checked(source,pred,mu,a.subject,a.task)
    print(json.dumps({'status':'preflight_ok','subject':a.subject,'task':a.task,'samples':len(rows)},ensure_ascii=False),flush=True)
    if a.dry_run:return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    with mapping_lock(out):refine(source,pred,mu,out,a.subject,a.task,torch.device(a.device),a.reuse_dir,a.reuse_identity)
if __name__=='__main__':main()
