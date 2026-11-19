#!/usr/bin/env python3
"""Complete 3-sample B0 vision reconstruction, refinement and four-subject statistics."""
import argparse
import csv
import gc
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from functools import partial
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
import run_multisubject_linear_imagery as work
import run_multisubject_imagery_natural as nat
import run_k60_vision_imagery as previous
import prepare_multisubject_vision_ridge_inputs as vision
import queue_multisubject_linear_imagery as linear_queue
import refine_multisubject_vision_imagery_three as refiner
import refine_w010_mindeye2_img2img as base
import sweep_subj01_retrieval_k as sweep
from multisubject_generate import generate_samples_verified
from multisubject_resume import archive_file
from reconstruct_subj01_imagery_train_val import head_checkpoint_alias
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import load_mapping_config,mapping_lock
from src.utils.roi_timed_stage_queue import ROITimedStageQueue
from src.utils.staged_queue import atomic_json

SUBJECTS=(1,2,5,7)
OUT=ROOT/'output_1104/vision_b0_four_subjects_v1'
REFERENCE=ROOT/'outputs_new/w010_subj01_vision_reference_v1/evaluation/vision'
OLD_REFINED=ROOT/'output_1104/imagery_three_text_direct_v1/B0/refined/vision'
OLD_IDENTITY=ROOT/'output_1104/imagery_three_text_direct_v1/identity.json'
ACTIVE=None

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def immutable(p,v):
    p=Path(p)
    if p.exists() and read(p)!=v:raise ValueError('Existing provenance mismatch: '+str(p))
    if not p.exists():atomic_json(v,p)
def subject_dir(s):return OUT/f'subj{s:02d}'
def initial_dir(s):return subject_dir(s)/'initial/vision'
def refined_dir(s):return subject_dir(s)/'refined/vision'
def equal_rows(a,b):
    return all(a[k]==b[k] for k in ('stimulus_id','stimulus_type','source_trial_ids'))

def subject_inputs(s):
    cfg,source,ident=work.context(s)
    if s==1:
        rows,values=previous.task_values(source)['vision']
        mu_path=previous.OUT/'inputs/vision_mu.npy';mu_manifest=previous.OUT/'inputs/vision_mu_manifest.json'
    else:
        _,checked_source,checked_ident,_model,original,values=vision.source_inputs(s)
        if checked_ident['brainvae_sha256']!=ident['brainvae_sha256'] or checked_ident['w010_sha256']!=ident['w010_sha256']:
            raise ValueError('Subject model source mismatch')
        rows=[{'subject_id':s,'stimulus_id':r['stimulus_id'],'stimulus_type':r['stimulus_type'],'task_type':'vision',
               'n_trials':8,'source_trial_ids':r['source_trial_ids'],'runs':r['runs']} for r in original]
        mu_path=work.subject_root(s)/'vision_ridge_inputs/vision_mu.npy';mu_manifest=mu_path.parent/'imagery_mu_manifest.json'
    info=read(mu_manifest);mu=np.load(mu_path,mmap_mode='r',allow_pickle=False)
    recorded=info.get('records',info.get('rows'))
    if (info.get('status')!='complete' or info['brainvae_sha256']!=ident['brainvae_sha256']
            or info['data_fingerprint']!=ident['data_fingerprint'] or info['mu_sha256']!=file_sha256(mu_path)
            or mu.shape!=(12,256,1664) or mu.dtype!=np.float32 or not np.isfinite(mu).all()
            or len(rows)!=12 or len(recorded)!=12 or any(not equal_rows(a,b) for a,b in zip(rows,recorded))
            or any(len(r['source_trial_ids'])!=8 for r in rows)):
        raise ValueError('Unadapted vision mu or trial identities differ')
    if not linear_queue.text_ok(s,'vision'):raise ValueError('Verified three-branch vision Ridge predictions unavailable')
    pred,pred_mu_manifest=linear_queue.text_paths(s,'vision')
    if pred_mu_manifest.resolve()!=mu_manifest.resolve():raise ValueError('Text and visual branches use different fMRI means')
    # Match the original multisubject adapter evaluation's trial and random-seed protocol.
    for method in ('w010_mse','mu_mse'):
        folder=work.method_root(s,method)/'initial/vision'
        for sample in range(3):
            manifest=read(folder/f'sample{sample:02d}/manifest.json');items=manifest['items']
            if len(items)!=12 or any(not equal_rows(a,b) or b['seed']!=42+100000*sample+i
                                      for i,(a,b) in enumerate(zip(rows,items))):
                raise ValueError('B0 and adapter comparison trial/seed differ')
    _,_,project=load_mapping_config(source['config_path'],ROOT)
    targets={}
    for r in rows:
        p=project.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages'/('setB' if r['stimulus_type']=='complex' else 'setA')/(r['stimulus_id']+'.png')
        if not p.is_file():raise FileNotFoundError(p)
        targets[r['stimulus_id']]=str(p)
    return cfg,source,ident,rows,mu_path,mu_manifest,pred,targets

def subj01_reuse(rows,ident,pred):
    meta=read(REFERENCE/'metadata.json');token=REFERENCE/'tokens/chunk00000.pt'
    if (meta['subject']!=1 or meta['dataset']!='vision' or meta['method']!='B0' or meta['n_items']!=12
            or meta['num_samples']!=3 or meta['brainvae_sha256']!=ident['brainvae_sha256']
            or meta['w010_sha256']!=ident['w010_sha256'] or meta['data_fingerprint']!=ident['data_fingerprint']
            or meta['token_sha256']!=file_sha256(token)):raise ValueError('Existing subj01 B0 source changed')
    old=read(OLD_IDENTITY);cond=read(pred/'prediction_complete.json')
    if (old['brainvae_sha256']!=ident['brainvae_sha256'] or old['w010_sha256']!=ident['w010_sha256']
            or old['source_fingerprint']!=ident['data_fingerprint']
            or old['zavy_sha256']!=base.ZAVY_SHA or old['img2img_steps']!=38 or old['timepoint']!=13 or old['cfg_scale']!=5.
            or old['seed']!='42 + stimulus_index' or old['ridge_predictions']['vision']['prediction_sha256']!=cond['prediction_sha256']):
        raise ValueError('Existing subj01 B0 refinement protocol differs')
    reuse={};hashes=[]
    for sample in range(3):
        sample_dir=REFERENCE/f'sample{sample:02d}';manifest=read(sample_dir/'manifest.json');items=manifest['items']
        if len(items)!=12 or manifest['sample_index']!=sample:raise ValueError('Old vision sample count mismatch')
        part=[]
        for i,(row,item) in enumerate(zip(rows,items)):
            path=sample_dir/item['path']
            if not equal_rows(row,item) or item['seed']!=42+100000*sample+i or not path.is_file():raise ValueError('Old vision trial/seed/image mismatch')
            with Image.open(path) as img:img.verify()
            h=file_sha256(path);reuse[(sample,i)]=(path,h);part.append(h)
        hashes.append(part)
    # Historical manifests do not contain a per-image hash ledger for initial sample01/02;
    # current hashes are recorded and sample00 is also checked against the prior experiment.
    if old['input_images']['B0_vision']['input_image_sha256']!=hashes[0]:raise ValueError('Existing B0 initial sample00 hashes changed')
    if not sweep.checked_sample(OLD_REFINED,0):raise ValueError('Existing B0 refinement ledger mismatch')
    old_items=read(OLD_REFINED/'sample00/manifest.json')['items']
    refined=[]
    for i,(row,item) in enumerate(zip(rows,old_items)):
        if not equal_rows(row,item) or item['seed']!=42+i:raise ValueError('Existing B0 refined trials mismatch')
        path=OLD_REFINED/'sample00'/item['path'];refined.append((path,file_sha256(path)))
    return reuse,refined,{'old_identity_sha256':file_sha256(OLD_IDENTITY),'old_refined_manifest_sha256':file_sha256(OLD_REFINED/'sample00/manifest.json'),
                           'initial_sha256':hashes,'refined_sample00_sha256':[h for _,h in refined]}

def prediction_ok(s):
    p=initial_dir(s);m=p/'metadata.json';token=p/'tokens/chunk00000.pt'
    if not m.is_file() or not token.is_file():return False
    info=read(m);selected=read(work.ROOT/f'outputs_new/w010_multisubject_v1/subj{s:02d}/selection.json')['selected']
    return (info.get('subject')==s and info.get('dataset')=='vision' and info.get('method')=='B0'
            and info.get('num_samples')==3 and info.get('n_items')==12
            and info.get('checkpoint_sha256')==selected['checkpoint_sha256']
            and info.get('token_sha256')==file_sha256(token))

def predict(s,device):
    cfg,source,ident,rows,mu_path,mu_manifest,pred,targets=subject_inputs(s)
    out=initial_dir(s);token=out/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
    if s==1:
        subj01_reuse(rows,ident,pred);old=REFERENCE/'tokens/chunk00000.pt';payload=torch.load(old,map_location='cpu',weights_only=False)
        if any(not equal_rows(a,b) for a,b in zip(rows,payload['records'])):raise ValueError('Old token records differ')
        values=payload['mean'].float().cpu()
    else:
        from src.models.clip_mapping import build_clip_mapper
        from src.utils.clip_head_evaluation import load_trained_head_source
        from src.utils.clip_mapping import MAPPING_PROTOCOL
        head,brain_cfg,_=load_mapping_config(source['config_path'],ROOT)
        with head_checkpoint_alias(source['protocol']['source_identity']):
            checkpoint,loaded,_=load_trained_head_source(source['checkpoint'],head,brain_cfg,ROOT,s,MAPPING_PROTOCOL)
        mapper=build_clip_mapper(checkpoint['config']).to(device).eval().requires_grad_(False)
        mapper.load_state_dict(checkpoint['model'],strict=True);del checkpoint,loaded
        mu=np.load(mu_path,mmap_mode='r',allow_pickle=False)
        with torch.inference_mode():
            values=torch.cat([nat.map_tokens(mapper,torch.from_numpy(np.array(mu[i:i+2],copy=True)).to(device)).float().cpu() for i in range(0,12,2)])
        del mapper;gc.collect();torch.cuda.empty_cache()
    if tuple(values.shape)!=(12,256,1664) or not torch.isfinite(values).all():raise ValueError('Invalid B0 vision tokens')
    payload={'mean':values,'records':rows,'representation':'mapped_raw_clip_tokens','mapping_checkpoint_sha256':ident['w010_sha256'],'adapter_sha256':None}
    if token.exists():
        existing=torch.load(token,map_location='cpu',weights_only=False)
        if existing['records']!=rows or existing['mapping_checkpoint_sha256']!=ident['w010_sha256'] or not torch.equal(existing['mean'],values):raise ValueError('Existing B0 tokens changed')
    else:nat.atomic_checkpoint(payload,token)
    meta={'protocol':'four_subject_vision_b0_3samples_v1','subject':s,'dataset':'vision','method':'B0','aggregation':'mean_fmri',
          'n_items':12,'num_samples':3,'num_steps':38,'seed':42,'partial_test':False,'debug_training':False,
          'checkpoint_sha256':ident['w010_sha256'],'w010_sha256':ident['w010_sha256'],'brainvae_sha256':ident['brainvae_sha256'],
          'data_fingerprint':ident['data_fingerprint'],'mu_sha256':file_sha256(mu_path),'token_sha256':file_sha256(token),
          'target_usage':'reference images for evaluation only','imagery_adapter':'none'}
    immutable(out/'metadata.json',meta);immutable(subject_dir(s)/'targets.json',targets)

def generation_ok(s):return prediction_ok(s) and all(sweep.checked_sample(initial_dir(s),i) for i in range(3))

def generate(s,device):
    if not prediction_ok(s):raise ValueError('Prediction missing')
    cfg,source,ident,rows,mu_path,manifest,pred,targets=subject_inputs(s)
    out=initial_dir(s);meta=read(out/'metadata.json');token=out/'tokens/chunk00000.pt';reuse={}
    if s==1:reuse,_,_=subj01_reuse(rows,ident,pred)
    missing=any((sample,i) not in reuse for sample in range(3) for i in range(12))
    engine=suffix=offset=None
    if missing:
        from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
        _,_,paths=load_mapping_config(source['config_path'],ROOT)
        unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint']);unclip=unclip if unclip.is_absolute() else ROOT/unclip
        engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
        decode=partial(unclip_decode_batch,rgb_mode='standard')
    else:decode=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Unexpected generation for reused image'))
    generate_samples_verified([token],out,meta,engine,suffix,offset,decode,save_image,3,42,{k:Path(v) for k,v in targets.items()},reuse=reuse)
    if not generation_ok(s):raise ValueError('Initial generation incomplete')
    atomic_json({'status':'complete','subject':s,'samples':3,'targets':12,'policy':'subj01 old initial 3 samples reused' if s==1 else 'new B0 samples'},subject_dir(s)/'initial_complete.json')
    if engine is not None:del engine;gc.collect();torch.cuda.empty_cache()

def refined_ok(s):
    out=refined_dir(s);marker=out/'refinement_complete.json';identity=out/'identity.json'
    if not marker.is_file() or not identity.is_file():return False
    info=read(marker)
    return (info.get('status')=='complete' and info.get('subject')==s and info.get('task')=='vision'
            and info.get('images')==36 and info.get('identity_sha256')==base.digest_json(read(identity))
            and all(sweep.checked_sample(out,i) for i in range(3)))

def refine(s,device):
    if not generation_ok(s):raise ValueError('Initial images incomplete')
    cfg,source,ident,rows,mu_path,mu_manifest,pred,targets=subject_inputs(s)
    meta,arrays,inputs,identity=refiner.checked(initial_dir(s),pred,mu_manifest,s,'vision')
    reused=None
    if s==1:
        _,reused,reuse_info=subj01_reuse(rows,ident,pred);identity['reused_existing_B0_sample00']=reuse_info
    identity['protocol']='four_subject_vision_b0_three_text_v1';identity['note']='B0 uses no imagery adapter; vision fMRI mean of eight trials'
    out=refined_dir(s);immutable(out/'identity.json',identity)
    negative_cross,negative_pool,audit=base.negative_condition(device);immutable(out/'negative_condition_audit.json',audit)
    engine,suffix=base.load_engine(device,38,5.)
    metadata={'protocol':identity['protocol'],'subject':s,'dataset':'vision','method':'B0','aggregation':'mean_fmri',
              'n_items':12,'num_samples':3,'partial_test':False,'debug_training':False,'checkpoint_sha256':ident['w010_sha256'],
              'brainvae_sha256':ident['brainvae_sha256'],'data_fingerprint':ident['data_fingerprint'],
              'generator_backend':'zavy_sdxl_img2img','text_condition':'nsd_ridge_three_branch_unadapted_fmri',
              'text_oracle':False,'oracle':False,'target_usage':'reference image for evaluation only; no imagery adapter',
              'source_identity_sha256':base.digest_json(identity)}
    immutable(out/'metadata.json',metadata)
    for sample in range(3):
        sample_dir=out/f'sample{sample:02d}';(sample_dir/'images').mkdir(parents=True,exist_ok=True);items=[]
        for i,row in enumerate(inputs[sample]):
            keys=('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')
            items.append({**{k:row[k] for k in keys},'item_index':i,'sample_index':sample,'seed':42+100000*sample+i,'path':f'images/item{i:05d}.png'})
        immutable(sample_dir/'manifest.json',{**metadata,'level':'stimulus_level','sample_index':sample,'n_images':12,'items':items})
        progress_path=sample_dir/'progress.jsonl';done={}
        if progress_path.exists():
            for line in progress_path.read_text().splitlines():
                r=json.loads(line);i=int(r['index'])
                if not 0<=i<12 or r['seed']!=items[i]['seed'] or r['stimulus_id']!=items[i]['stimulus_id']:raise ValueError('Invalid progress record')
                done[i]=r
        with progress_path.open('a',encoding='utf-8') as progress:
            for i,item in enumerate(items):
                path=sample_dir/item['path'];entry=done.get(i)
                if entry:
                    if not path.is_file() or file_sha256(path)!=entry['sha256']:raise ValueError('Existing refinement changed')
                    continue
                if path.exists():archive_file(path)
                temp=path.with_name(path.stem+'.tmp.png')
                if sample==0 and reused is not None:
                    if file_sha256(reused[i][0])!=reused[i][1]:raise ValueError('Old B0 refinement changed')
                    shutil.copyfile(reused[i][0],temp)
                else:
                    image=initial_dir(s)/f'sample{sample:02d}'/inputs[sample][i]['path']
                    if file_sha256(image)!=identity['input_images_sha256'][sample][i]:raise ValueError('Initial image changed')
                    with Image.open(image) as im:x=pil_to_tensor(ImageOps.exif_transpose(im).convert('RGB')).float().div_(255).unsqueeze(0).to(device)
                    cond,uncond=base.conditions(arrays,i,suffix,negative_cross,negative_pool,device)
                    rgb=base.img2img(engine,x,cond,uncond,item['seed'],38,13);save_image(rgb.cpu(),str(temp))
                os.replace(temp,path)
                progress.write(json.dumps({'index':i,'stimulus_id':item['stimulus_id'],'seed':item['seed'],'sha256':file_sha256(path)})+'\n');progress.flush()
                print('[refine] subj%02d sample%02d %d/12'%(s,sample,i+1),flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()
    atomic_json({'status':'complete','subject':s,'task':'vision','images':36,'identity_sha256':base.digest_json(identity)},out/'refinement_complete.json')
    if not refined_ok(s):raise ValueError('Refinement failed verification')

def eval_ok(s,phase):
    sweep.SAMPLES=3;sweep.experiment=lambda _k:subject_dir(s)
    return sweep.evaluation_ok(10,'vision',phase)

def evaluate(s,phase,device):
    sweep.SAMPLES=3;sweep.experiment=lambda _k:subject_dir(s)
    sweep.evaluate(10,'vision',phase,device)

def verify_paired(s,method,phase):
    b=subject_dir(s)/phase/'vision';a=work.method_root(s,method)/phase/'vision'
    if not eval_ok(s,phase):raise ValueError('B0 evaluation invalid')
    sweep.SAMPLES=3;sweep.experiment=lambda _k:work.method_root(s,method)
    if not sweep.evaluation_ok(10,'vision',phase):raise ValueError('Adapter evaluation invalid')
    for sample in range(3):
        x=read(b/f'sample{sample:02d}/manifest.json')['items'];y=read(a/f'sample{sample:02d}/manifest.json')['items']
        if len(x)!=12 or len(y)!=12 or any(not equal_rows(i,j) or i['seed']!=j['seed'] or file_sha256(Path(i['target_image']))!=file_sha256(Path(j['target_image'])) for i,j in zip(x,y)):
            raise ValueError('Methods do not have matched trials, seeds and target images')

def bh(values):
    n=len(values);order=sorted(range(n),key=lambda i:values[i]);out=[0.]*n;last=1.
    for rank in range(n,0,-1):
        index=order[rank-1];last=min(last,values[index]*n/rank);out[index]=last
    return out

def summarize():
    subject_rows=[];group_rows=[]
    for phase in ('initial','refined'):
        for s in SUBJECTS:
            for method in ('w010_mse','mu_mse'):verify_paired(s,method,phase)
        for group,prefix in (('all',''),('natural','complex_'),('geometry','simple_')):
            values={}
            for method in ('B0','w010_mse','mu_mse'):
                values[method]={}
                for s in SUBJECTS:
                    folder=subject_dir(s) if method=='B0' else work.method_root(s,method)
                    metrics=read(folder/phase/'vision/image_metrics.json')
                    if metrics['n_targets']!=12 or metrics['n_samples']!=3 or len(metrics['per_seed'])!=3:raise ValueError('Metric sample scope mismatch')
                    values[method][s]={}
                    for sample in range(3):
                        data={k:float(metrics['per_seed'][sample][prefix+k]) for k in sweep.KEYS}
                        if not all(math.isfinite(v) for v in data.values()):raise ValueError('Nonfinite metric')
                        subject_rows.append({'subject':s,'method':method,'phase':phase,'group':group,'sample':f'sample{sample:02d}',**data})
                    avg={k:float(np.mean([metrics['per_seed'][i][prefix+k] for i in range(3)])) for k in sweep.KEYS}
                    if any(not np.isclose(avg[k],metrics['metrics'][prefix+k],rtol=1e-8,atol=1e-8) for k in sweep.KEYS):raise ValueError('Three sample mean mismatch')
                    subject_rows.append({'subject':s,'method':method,'phase':phase,'group':group,'sample':'mean',**avg});values[method][s]=avg
            for method in ('w010_mse','mu_mse'):
                rows=[]
                for key in sweep.KEYS:
                    b=np.array([values['B0'][s][key] for s in SUBJECTS]);a=np.array([values[method][s][key] for s in SUBJECTS]);diff=a-b
                    observed=abs(float(diff.mean()));p=sum(abs(float(np.mean(diff*np.array(signs))))>=observed-1e-12 for signs in itertools.product((-1,1),repeat=4))/16
                    delta=float(diff.mean());half=3.182446305284263*float(diff.std(ddof=1))/2
                    rows.append({'phase':phase,'group':group,'comparison':method+' - B0','metric':key,
                                 'b0_mean':float(b.mean()),'b0_subject_sd':float(b.std(ddof=1)),
                                 'method_mean':float(a.mean()),'method_subject_sd':float(a.std(ddof=1)),
                                 'paired_difference_mean':delta,'paired_difference_sd':float(diff.std(ddof=1)),
                                 'paired_difference_ci95_low':delta-half,'paired_difference_ci95_high':delta+half,
                                 'paired_sign_flip_p':p})
                for r,q in zip(rows,bh([r['paired_sign_flip_p'] for r in rows])):r['bh_q_within_8_metrics']=q
                group_rows.extend(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows in [('subject_metrics.csv',subject_rows),('paired_group_comparison.csv',group_rows)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json({'status':'complete','subjects':list(SUBJECTS),'subject_n':4,'samples':3,'rows':group_rows,
                 'note':'Three diffusion samples averaged per subject; exact two-sided sign-flip test uses 4 paired subjects, minimum p=0.125. BH is separate for each phase, group and method over 8 metrics.'},OUT/'summary.json')

def stop(sig,frame):
    if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.send_signal(sig)
    raise SystemExit(128+sig)

def preflight():
    if not base.ZAVY.is_file():raise FileNotFoundError(base.ZAVY)
    for s in SUBJECTS:
        _,_,ident,rows,_,_,pred,_=subject_inputs(s)
        if s==1:subj01_reuse(rows,ident,pred)
        print('[preflight] subj%02d 12 targets, 8 trials, Ridge and comparison identities verified'%s,flush=True)

def queue(device):
    global ACTIVE
    OUT.mkdir(parents=True,exist_ok=True);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with mapping_lock(OUT):
        q=ROITimedStageQueue(OUT)
        preflight()
        protocol={'subjects':list(SUBJECTS),'samples':3,'n_targets':12,'trials_per_target':8,'seed':'42+100000*sample+stimulus_index',
                  'method':'B0 no imagery adapter','text':'existing subject-specific three-branch Ridge from same unadapted vision fMRI',
                  'script_sha256':file_sha256(Path(__file__)),'generator_sha256':file_sha256(Path(base.__file__)),
                  'evaluation_sha256':file_sha256(ROOT/'scripts/eval_scheme1_reconstruction.py'),
                  'alias_helper_sha256':file_sha256(ROOT/'output_1104/scripts/reconstruct_subj01_imagery_train_val.py')}
        immutable(OUT/'queue_identity.json',protocol)
        def stage(key,s,action,check,extra=()):
            def run():
                global ACTIVE
                if check():print('[reuse]',key,flush=True);return
                if shutil.disk_usage(ROOT).free<5*2**30:raise OSError('Less than 5 GiB free; preserve output and resume after cleanup')
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',action,'--subject',str(s),'--device',device,*extra]
                log=OUT/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as f:
                    f.write('\nCOMMAND '+json.dumps(cmd)+'\n');f.flush()
                    ACTIVE=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);code=ACTIVE.wait();ACTIVE=None
                if code:raise RuntimeError(f'{key} failed ({code}); see {log}')
            q.run(key,run,check)
        for s in SUBJECTS:
            prefix=f'subj{s:02d}'
            stage(prefix+'_predict',s,'predict',lambda s=s:prediction_ok(s))
            stage(prefix+'_generate',s,'generate',lambda s=s:generation_ok(s) and (subject_dir(s)/'initial_complete.json').is_file())
            stage(prefix+'_initial_eval',s,'evaluate',lambda s=s:eval_ok(s,'initial'),('--phase','initial'))
            stage(prefix+'_refine',s,'refine',lambda s=s:refined_ok(s))
            stage(prefix+'_refined_eval',s,'evaluate',lambda s=s:eval_ok(s,'refined'),('--phase','refined'))
        summarize();q.state(status='complete',stage=None,error=None)
    print('[complete]',OUT,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('preflight','queue','predict','generate','refine','evaluate','summary'),default='queue')
    p.add_argument('--subject',type=int,choices=SUBJECTS);p.add_argument('--phase',choices=('initial','refined'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='preflight':
        preflight();return
    if a.stage=='summary':summarize();return
    if not torch.cuda.is_available() or torch.device(a.device).type!='cuda':raise RuntimeError('CUDA required')
    if a.stage=='queue':queue(a.device);return
    if not a.subject:p.error('--subject required')
    device=torch.device(a.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='evaluate':
        if not a.phase:p.error('--phase required')
        evaluate(a.subject,a.phase,device)
    else:{'predict':predict,'generate':generate,'refine':refine}[a.stage](a.subject,device)

if __name__=='__main__':main()
