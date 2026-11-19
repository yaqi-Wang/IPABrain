#!/usr/bin/env python3
"""Refine subject imagery B0/R/V/R_MSE sample00 with NSD-trained Ridge predictions.

All three methods use the same unadapted held-out imagery BrainVAE μ prediction
per stimulus. Qwen/GIT reference embeddings and target images are not read by
the generation code. An imagery evaluator may use manifest target paths later.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'output_1104/scripts'))

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

import refine_w010_mindeye2_img2img as base
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import mapping_lock
from multisubject_resume import archive_file


RUN = ROOT / 'output_1104/imagery_ridge_refine_subj01_v1'
MU = RUN / 'inputs'
PRED = ROOT / 'output_1104/nsd_subj01/ridge_three_text/predictions/imagery_subj01_12stim_v1'
NATURAL = ROOT / 'outputs_new/imagery_w010_subj01_natural_v1/runs'
GEOMETRY = ROOT / 'outputs_new/imagery_w010_subj01_expansion_v1/transfer/imagery_geometry'
METHODS = ('B0', 'R', 'V', 'R_MSE')
STOP = 0


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def on_signal(signum, _frame):
    global STOP
    STOP = signum
    print('[signal] {} received; stop after the current image'.format(signum), flush=True)


def checked_inputs(mu_dir, prediction_dir):
    mu_path=mu_dir/'imagery_mu.npy';mu_meta=read_json(mu_dir/'imagery_mu_manifest.json')
    pred_meta=read_json(prediction_dir/'prediction_complete.json')
    mu=np.load(mu_path,mmap_mode='r',allow_pickle=False);rows=mu_meta['rows'];subject=mu_meta['subject']
    if (subject not in range(1,9) or mu_meta.get('status')!='complete' or mu.shape!=(12,256,1664)
            or mu.dtype!=np.float32 or not np.isfinite(mu).all() or mu_meta['mu_sha256']!=file_sha256(mu_path)
            or len(rows)!=12 or [r['row_index'] for r in rows]!=list(range(12))
            or [r['group'] for r in rows]!=['natural']*6+['geometry']*6
            or mu_meta.get('test_images_or_text_used_as_input') is not False):
        raise ValueError('Invalid imagery BrainVAE means')
    if (pred_meta.get('status')!='complete' or pred_meta.get('test_text_used_as_input') is not False
            or pred_meta['input'].get('kind')!='user_supplied_brainvae_mu' or pred_meta['input'].get('images')!=12
            or pred_meta['input'].get('sha256')!=mu_meta['mu_sha256']
            or pred_meta.get('model_metadata_sha256')!=mu_meta['ridge_model_metadata_sha256']):
        raise ValueError('Ridge prediction does not match imagery μ')
    arrays,hashes={},{}
    for name,shape in base.BRANCHES.items():
        path=prediction_dir/('predicted_'+name+'.npy');arr=np.load(path,mmap_mode='r',allow_pickle=False)
        digest=file_sha256(path)
        if arr.shape!=(12,*shape) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or digest!=pred_meta['prediction_sha256'][name]:
            raise ValueError('Invalid Ridge prediction: '+name)
        arrays[name],hashes[name]=arr,digest
    by_id={r['stimulus_id']:r for r in rows}
    if len(by_id)!=12:raise ValueError('Duplicate stimuli')
    catalog,provenance={},{}
    combined_input=GEOMETRY is None
    initial_receipt=read_json(NATURAL.parent/'generation_complete.json') if combined_input else None
    if initial_receipt is not None and (initial_receipt.get('status')!='complete' or initial_receipt.get('samples')!=3):
        raise ValueError('Subj01 initial image generation incomplete')
    for method in METHODS:
        method_meta=None;files=[];samples=[]
        for sample_index in range(3):
            combined=[]
            groups=([('all',NATURAL,None)] if combined_input else [('natural',NATURAL,'complex'),('geometry',GEOMETRY,'simple')])
            for group,root,kind in groups:
                directory=(root/method/'imagery' if combined_input else root/method/'evaluation/imagery');meta_path=directory/'metadata.json'
                manifest_path=directory/f'sample{sample_index:02d}'/'manifest.json'
                meta,manifest=read_json(meta_path),read_json(manifest_path)
                if (meta.get('subject')!=subject or meta.get('dataset')!='imagery' or meta.get('method')!=method
                        or meta.get('brainvae_sha256')!=mu_meta['brainvae_sha256']
                        or meta.get('data_fingerprint')!=mu_meta['data_fingerprint'] or meta.get('aggregation')!='mean_fmri'
                        or meta.get('num_samples')!=3 or meta.get('partial_test') or meta.get('debug_training')
                        or manifest.get('sample_index')!=sample_index or len(manifest.get('items',[]))!=(12 if combined_input else 6)):
                    raise ValueError(f'Baseline metadata mismatch {group} {method} sample{sample_index:02d}')
                identity=(meta['checkpoint_sha256'],meta['w010_sha256'],meta.get('adapter_sha256'))
                if method_meta is not None and identity!=method_meta:raise ValueError('Adapter identity mismatch')
                method_meta=identity
                files.append({'group':group,'sample_index':sample_index,'metadata_sha256':file_sha256(meta_path),
                              'manifest_sha256':file_sha256(manifest_path)})
                expected=[r['stimulus_id'] for r in rows if group=='all' or r['group']==group]
                if [item['stimulus_id'] for item in manifest['items']]!=expected:raise ValueError('Stimulus order mismatch')
                for item in manifest['items']:
                    row=by_id[item['stimulus_id']]
                    if (item['source_trial_ids']!=row['source_trial_ids'] or item['n_trials']!=4
                            or item['task_type']!='imagery' or item['stimulus_type']!=row['stimulus_type']):
                        raise ValueError('Held-out trial mismatch')
                    sample_dir=manifest_path.parent.resolve();image_path=(sample_dir/item['path']).resolve()
                    if (subject!=1 and sample_dir not in image_path.parents) or not image_path.is_file():raise ValueError('Missing input image')
                    if item.get('source_sha256') and file_sha256(image_path)!=item['source_sha256']:
                        raise ValueError('Reused input image checksum mismatch')
                    if initial_receipt is not None and file_sha256(image_path)!=initial_receipt['image_sha256'][method][sample_index][item['stimulus_id']]:
                        raise ValueError('Subj01 initial image changed')
                    with Image.open(image_path) as image:
                        if image.size!=(768,768):raise ValueError('Initial image is not 768x768')
                        image.verify()
                    combined.append({'index':row['row_index'],'group':group,'stimulus_id':row['stimulus_id'],
                        'stimulus_type':row['stimulus_type'],'task_type':'imagery','subject_id':subject,'n_trials':4,
                        'source_trial_ids':row['source_trial_ids'],'input_path':str(image_path),
                        'input_sha256':file_sha256(image_path),'target_image':item['target_image']})
            if [x['index'] for x in combined]!=list(range(12)):raise ValueError('Invalid combined order')
            samples.append(combined)
        catalog[method]=samples
        provenance[method]={'checkpoint_sha256':method_meta[0],'w010_sha256':method_meta[1],
                            'adapter_sha256':method_meta[2],'source_manifests':files}
    identity={'dataset':'imagery','subject':subject,'targets':12,'samples':3,'methods':list(METHODS),
              'mu_sha256':mu_meta['mu_sha256'],'mu_manifest_sha256':file_sha256(mu_dir/'imagery_mu_manifest.json'),
              'ridge_prediction_metadata_sha256':file_sha256(prediction_dir/'prediction_complete.json'),
              'ridge_prediction_sha256':hashes,'brainvae_sha256':mu_meta['brainvae_sha256'],
              'data_fingerprint':mu_meta['data_fingerprint'],'baseline_sources':provenance,
              'baseline_images':{m:[{x['stimulus_id']:x['input_sha256'] for x in sample} for sample in catalog[m]] for m in METHODS},
              'prediction_policy':'same NSD-Ridge prediction from raw held-out imagery μ across methods and samples',
              'test_images_or_text_used_as_generation_input':False}
    return catalog,arrays,identity

def progress(path, items, sample_dir, seed):
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        index = int(row['index'])
        if not 0 <= index < 12 or row['stimulus_id'] != items[index]['stimulus_id'] or row['seed'] != seed + index:
            raise ValueError('Progress record belongs to a different run')
        latest[index]=row
    done={}
    for index,row in latest.items():
        path_out = sample_dir / ('images/item%05d.png' % index)
        if path_out.is_file() and file_sha256(path_out) == row['image_sha256']:
            with Image.open(path_out) as image:
                image.verify()
            done[index] = row
    return done


def main():
    global RUN,MU,PRED,NATURAL,GEOMETRY,METHODS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,choices=range(1,9),required=True)
    p.add_argument('--mu-dir',type=Path);p.add_argument('--prediction-dir',type=Path);p.add_argument('--output-dir',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=38);p.add_argument('--timepoint',type=int,default=13)
    p.add_argument('--cfg-scale',type=float,default=5.0);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--b0-only',action='store_true',help='process B0 only in its separate output directory')
    args=p.parse_args();root=ROOT/f'output_1104/multisubject_nsd_imagery_v1/subj{args.subject:02d}'
    RUN=root/'imagery_refine_k60_3samples_v1'
    if args.subject==1:
        MU=ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs'
        PRED=ROOT/'output_1104/nsd_subj01/ridge_three_text/predictions/imagery_subj01_12stim_v1'
        NATURAL=root/'imagery_initial_k60_3samples_v1/initial';GEOMETRY=None
    else:
        MU=root/'imagery_ridge_inputs_3samples_v1';PRED=root/'ridge/predictions/imagery_12stim_3samples_v1'
        NATURAL=root/'imagery_natural_k60_3samples_v1/runs';GEOMETRY=root/'imagery_geometry_transfer_k60_3samples_v1/runs'
    if args.b0_only:
        METHODS=('B0',);RUN=root/'imagery_b0_refine_3samples_v1'
        NATURAL=root/'imagery_b0_initial_3samples_v1/initial';GEOMETRY=None
    args.mu_dir=args.mu_dir or MU;args.prediction_dir=args.prediction_dir or PRED;args.output_dir=args.output_dir or RUN/'refinement'
    if args.steps<2 or not 2<=args.timepoint<=args.steps+1 or args.seed<0 or not math.isfinite(args.cfg_scale) or args.cfg_scale<=0:
        p.error('invalid img2img settings')
    catalog,arrays,identity=checked_inputs(args.mu_dir.resolve(),args.prediction_dir.resolve())
    if identity['subject']!=args.subject or file_sha256(base.ZAVY)!=base.ZAVY_SHA:raise ValueError('Subject or Zavy mismatch')
    print('[preflight] 12 stimuli × {} methods × 3 initial samples matched'.format(len(METHODS)),flush=True)
    if args.dry_run:return
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.output_dir.resolve()
    identity.update({'zavy_sha256':base.ZAVY_SHA,'steps':args.steps,'timepoint':args.timepoint,
                     'cfg_scale':args.cfg_scale,'seed_base':args.seed,'seed_policy':'seed_base+sample*100000+stimulus_index',
                     'negative_prompt':base.NEGATIVE,'backend':'MindEye2-style Zavy SDXL img2img',
                     'script_sha256':file_sha256(Path(__file__)),'base_img2img_script_sha256':file_sha256(Path(base.__file__))})
    if args.subject==1:
        previous=ROOT/'output_1104/imagery_k60_ridge_subj01_v1/refined'
        identity['reused_sample00_sources']={method:{'metadata_sha256':file_sha256(previous/method/'imagery/metadata.json'),
                                                      'manifest_sha256':file_sha256(previous/method/'imagery/sample00/manifest.json')}
                                             for method in METHODS if method in ('B0','R','V')}
    run_hash=base.digest_json(identity)
    with mapping_lock(out):
        pending=[]
        for method in METHODS:
            folder=out/method/'imagery';folder.mkdir(parents=True,exist_ok=True)
            metadata={'protocol':'imagery_nsd_ridge_w010_zavy_3samples_v1','subject':args.subject,
                      'dataset':'imagery','method':method,'aggregation':'mean_fmri','n_items':12,'num_samples':3,
                      'partial_test':False,'debug_training':False,
                      'checkpoint_sha256':identity['baseline_sources'][method]['checkpoint_sha256'],
                      'data_fingerprint':identity['data_fingerprint'],'brainvae_sha256':identity['brainvae_sha256'],
                      'text_condition':'nsd_ridge_predicted_from_unadapted_imagery_mu','text_oracle':False,
                      'oracle':False,'target_usage':'evaluation_only','generator_backend':'mindeye2_zavy_sdxl_img2img',
                      'source_identity_sha256':run_hash,'source_identity':identity}
            meta=folder/'metadata.json'
            if meta.exists() and read_json(meta)!=metadata:raise ValueError('Existing metadata differs')
            if not meta.exists():base.write_json(meta,metadata)
            for sample_index in range(3):
                sample=folder/f'sample{sample_index:02d}';(sample/'images').mkdir(parents=True,exist_ok=True)
                seed_base=args.seed+sample_index*100000;rows=catalog[method][sample_index]
                items=[{**{k:r[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')},
                        'item_index':r['index'],'sample_index':sample_index,'seed':seed_base+r['index'],
                        'path':'images/item%05d.png'%r['index']} for r in rows]
                manifest=dict(metadata,level='stimulus_level',sample_index=sample_index,n_images=12,items=items)
                path=sample/'manifest.json'
                if path.exists() and read_json(path)!=manifest:raise ValueError('Existing manifest differs')
                if not path.exists():base.write_json(path,manifest)
                done=progress(sample/'progress.jsonl',rows,sample,seed_base)
                if args.subject==1 and sample_index==0 and method in ('B0','R','V'):
                    if (args.seed,args.steps,args.timepoint,args.cfg_scale)!=(42,38,13,5.0):
                        raise ValueError('Existing subj01 refinements use seed=42, steps=38, timepoint=13, CFG=5')
                    prior=ROOT/'output_1104/imagery_k60_ridge_subj01_v1/refined'/method/'imagery'
                    prior_meta=read_json(prior/'metadata.json');prior_manifest=read_json(prior/'sample00/manifest.json')
                    prov=prior_meta['provenance']
                    if (prior_meta.get('subject')!=1 or prior_meta.get('text_oracle') is not False
                            or prov['zavy_sha256']!=base.ZAVY_SHA or prov['ridge_prediction_sha256']!=identity['ridge_prediction_sha256']):
                        raise ValueError('Existing subj01 refined sample provenance differs')
                    with (sample/'progress.jsonl').open('a',encoding='utf-8') as handle:
                        for index,row in enumerate(rows):
                            old_item=prior_manifest['items'][index];src=prior/'sample00'/old_item['path']
                            receipt=src.with_suffix('.json')
                            if (old_item['stimulus_id']!=row['stimulus_id'] or old_item['source_trial_ids']!=row['source_trial_ids']
                                    or old_item['seed']!=seed_base+index or prov['input_images'][row['stimulus_id']]!=row['input_sha256']
                                    or read_json(receipt)['sha256']!=file_sha256(src)):
                                raise ValueError('Existing subj01 refined image or source changed')
                            if index in done:continue
                            dest=sample/('images/item%05d.png'%index);temp=dest.with_name(dest.name+'.tmp')
                            if dest.exists():archive_file(dest)
                            shutil.copyfile(src,temp);os.replace(temp,dest)
                            if file_sha256(dest)!=file_sha256(src):raise ValueError('Refined image copy checksum differs')
                            handle.write(json.dumps({'index':index,'stimulus_id':row['stimulus_id'],'seed':seed_base+index,
                                                     'image_sha256':file_sha256(dest)})+'\n');handle.flush()
                    done=progress(sample/'progress.jsonl',rows,sample,seed_base)
                pending.extend((method,sample_index,i) for i in range(12) if i not in done)
                print('[resume] {} sample={} complete={}/12'.format(method,sample_index,len(done)),flush=True)
        if pending:
            signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
            negative_cross,negative_pool,negative_audit=base.negative_condition(device)
            audit_path=out/'negative_condition_audit.json'
            if audit_path.exists() and read_json(audit_path)!=negative_audit:raise ValueError('Negative conditions changed')
            if not audit_path.exists():base.write_json(audit_path,negative_audit)
            if STOP:return 128+STOP
            engine,suffix=base.load_engine(device,args.steps,args.cfg_scale);started=time.monotonic()
            for number,(method,sample_index,index) in enumerate(pending,1):
                item=catalog[method][sample_index][index];sample=out/method/'imagery'/f'sample{sample_index:02d}'
                with Image.open(item['input_path']) as source:
                    image=ImageOps.exif_transpose(source).convert('RGB')
                    tensor=pil_to_tensor(image).float().div_(255.0).unsqueeze(0).to(device)
                seed=args.seed+sample_index*100000+index
                cond,uncond=base.conditions(arrays,index,suffix,negative_cross,negative_pool,device)
                rgb=base.img2img(engine,tensor,cond,uncond,seed,args.steps,args.timepoint)
                dest=sample/('images/item%05d.png'%index);temp=dest.with_name(dest.stem+'.tmp.png')
                if dest.exists():archive_file(dest)
                save_image(rgb.cpu(),str(temp));os.replace(str(temp),str(dest))
                record={'index':index,'stimulus_id':item['stimulus_id'],'seed':seed,'image_sha256':file_sha256(dest)}
                with (sample/'progress.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(record)+'\n');handle.flush()
                print('[image] {}/{} {} sample={} {} {:.2f}s/image'.format(number,len(pending),method,sample_index,item['stimulus_id'],(time.monotonic()-started)/number),flush=True)
                if STOP:return 128+STOP
            del engine;gc.collect();torch.cuda.empty_cache()
        for method in METHODS:
            folder=out/method/'imagery';receipts={}
            for sample_index in range(3):
                name=f'sample{sample_index:02d}';sample=folder/name
                if len(progress(sample/'progress.jsonl',catalog[method][sample_index],sample,args.seed+sample_index*100000))!=12:
                    raise RuntimeError('Incomplete images for '+method)
                receipts[name]={'manifest_sha256':file_sha256(sample/'manifest.json'),'progress_sha256':file_sha256(sample/'progress.jsonl')}
            base.write_json(folder/'generation_complete.json',{'status':'complete','images':36,'targets':12,'samples':3,
                'source_identity_sha256':run_hash,'sample_receipts':receipts,'text_oracle':False})
        base.write_json(out/'generation_complete.json',{'status':'complete','images':36*len(METHODS),
                        'methods':list(METHODS),'source_identity_sha256':run_hash})
        print('[complete] {} imagery refinements {}'.format(36*len(METHODS),out),flush=True)

if __name__ == '__main__':
    main()
