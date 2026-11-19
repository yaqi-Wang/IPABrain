#!/usr/bin/env python3
"""Subject-specific W010-MSE and mu-MSE linear imagery experiment worker."""
from __future__ import annotations
import argparse,copy,gc,hashlib,itertools,json,math,os,shutil,sys
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
import run_multisubject_imagery_natural as nat
import run_imagery_w010_pilot as pilot
import run_k60_vision_imagery as previous
import run_imagery_clipl_direct as wloss
import run_imagery_mu_clip_mse as mloss
import sweep_subj01_retrieval_k as sweep
import prepare_multisubject_vision_ridge_inputs as vision
from check_scheme1_clip import load_imagery_truth
from multisubject_generate import generate_samples_verified
from src.data.scheme1 import file_sha256
from src.models.imagery_residual_comparison import ImageryResidualAdapter
from src.utils.clip_mapping import load_mapping_config
from src.utils.imagery_w010_pilot import split_trials,balanced_schedule,pair_records
from src.utils.perception_anchor_prior import aggregate_anchor_parameters
from src.utils.staged_queue import atomic_json

OUT=ROOT/'output_1104/multisubject_linear_imagery_v1'
OLD=ROOT/'output_1104/multisubject_nsd_imagery_v1'
METHODS=('w010_mse','mu_mse');TASKS=('vision','imagery')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def immutable(p,value):
    p=Path(p)
    if p.exists() and read(p)!=value:raise ValueError(f'Immutable identity differs: {p}')
    if not p.exists():atomic_json(value,p)
def subject_root(s):return OUT/f'subj{s:02d}'
def method_root(s,m):return subject_root(s)/m

def context(s):
    if s==1:return previous.source()
    root=OLD/f'subj{s:02d}';cfg=yaml.safe_load((root/'config_imagery_natural_k60_3samples.yaml').read_text())
    source=nat.run_source(cfg);original=nat.require_identity(cfg,ROOT/cfg['experiment_root'],source)
    cfg=dict(cfg);cfg['experiment_root']=str((subject_root(s)/'retrieval60').relative_to(ROOT))
    return cfg,source,original

def data_parts(s):
    cfg,source,identity=context(s);data=nat.load_trials(source)
    if s==1:
        natural=read(wloss.NAT);geometry=read(wloss.GEO)
        rows=read(ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs/imagery_mu_manifest.json')['rows']
    else:
        natural=split_trials(data,42,'complex');geometry=split_trials(data,42,'simple')
        rows=read(OLD/f'subj{s:02d}/imagery_ridge_inputs_3samples_v1/imagery_mu_manifest.json')['rows']
    if len(rows)!=12 or [x['stimulus_type'] for x in rows]!=['complex']*6+['simple']*6:raise ValueError('Unexpected stimulus order')
    splits={}
    for row in rows:
        sid=row['stimulus_id'];z=(natural if row['stimulus_type']=='complex' else geometry)['rows'][sid]
        if [len(z[k]) for k in ('train','val','test')]!=[10,2,4] or z['test']!=row['test_rows']:raise ValueError('Imagery split mismatch')
        for k in ('train','val','test'):
            for ix in z[k]:
                if str(data['task_type'][ix])!='imagery' or str(data['stimulus_id'][ix])!=sid:raise ValueError('Invalid trial')
        if any(set(z[a]) & set(z[b]) for a,b in (('train','val'),('train','test'),('val','test'))):raise ValueError('Trial overlap')
        splits[sid]={k:z[k] for k in ('train','val','test')}
    return cfg,source,identity,data,rows,splits,natural

def ranked_plan(retrieval,split):
    # Preserve the original K10 rule when possible; if target neighborhoods overlap,
    # choose globally disjoint validation images deterministically within top ten.
    targets=split['targets'];groups=[copy.deepcopy(retrieval['queries'][t][:10]) for t in targets]
    if len(targets)!=6 or any(len(g)!=10 for g in groups):raise ValueError('Expected six top-10 trial lists')
    options=[]
    for group in groups:
        options.append([im for im in dict.fromkeys(r['image_id'] for r in group) if sum(r['image_id']==im for r in group)>=2])
    preferred=[g[4]['image_id'] for g in groups]
    options=[[x for x in [preferred[i],*opts] if x in opts] for i,opts in enumerate(options)]
    chosen=None
    for selection in itertools.product(*options):
        if len(set(selection))!=6:continue
        selected=set(selection)
        if all(any(r['image_id'] not in selected for r in g) for g in groups):chosen=selection;break
    if chosen is None:raise ValueError('K=10 cannot provide globally disjoint teacher validation and training; inspect retrieval')
    old_ids=sorted({r['teacher_index'] for g in groups for r in g});remap={v:i for i,v in enumerate(old_ids)}
    teachers=[]
    for i in old_ids:
        t=copy.deepcopy(retrieval['teachers'][i])
        if t['kind']!='nsd_trial' or t['teacher_index']!=i:raise ValueError('Teacher identity mismatch')
        t['original_teacher_index']=i;t['teacher_index']=remap[i];teachers.append(t)
    train_groups=[];val_groups=[];queries={};counts=[];held=set(chosen)
    for label,(target,group) in enumerate(zip(targets,groups)):
        chosen_val=[r['teacher_index'] for r in group if r['image_id']==chosen[label]][:2]
        for row in group:
            old=row['teacher_index'];teacher=retrieval['teachers'][old]
            if any(row[k]!=teacher[k] for k in ('image_id','source_row','repeat_index')):
                raise ValueError('Ranked trial does not match teacher provenance')
            row['original_teacher_index']=old;row['teacher_index']=remap[old]
            row['teacher_split']='val' if old in chosen_val else ('excluded_validation_image' if row['image_id'] in held else 'train')
        train=[r['teacher_index'] for r in group if r['teacher_split']=='train'];val=[r['teacher_index'] for r in group if r['teacher_split']=='val']
        if not train or len(val)!=2:raise ValueError('Insufficient teachers')
        parents=np.random.default_rng(1000+label).permutation(split['rows'][target]['train']).tolist()
        order=np.random.default_rng(2000+label).permutation(train).tolist();rng=np.random.default_rng(3000+label)
        pairs=[{'parent_row':int(parents[i%len(parents)]),'teacher_index':int(order[i%len(order)]),
                'augmented':i>=len(parents),'noise_seed':int(rng.integers(1,2**31-1)),
                'balance_repeat':i>=max(len(parents),len(order))} for i in range(18*6)]
        train_groups.append(pairs);val_groups.append(pair_records(split['rows'][target]['val'],val,9000+label));queries[target]=group
        counts.append({'target':target,'train_trials':len(train),'validation_trials':len(val),'train_images':len({r['image_id'] for r in group if r['teacher_split']=='train'}),'validation_image':int(chosen[label])})
    schedules={}
    for part,entries in [('train',train_groups),('val',val_groups)]:
        for attempt in range(512):
            try:batches=balanced_schedule(entries,6,42+attempt+(2000 if part=='val' else 0));break
            except ValueError:continue
        else:raise ValueError('Cannot balance K10 schedule')
        if part=='train' and len(batches)!=18:raise ValueError('Unexpected update count')
        schedules[part]={'protocol':'multisubject_k10_fixed_steps_v1','portion':part,'batches':batches,'attempt':attempt}
    plan={'overview':{'k_unit':'NSD fMRI trial','k_per_query':10,'epochs':30,'updates_per_epoch':18,
                      'validation_images':list(map(int,chosen)),'counts':counts,
                      'selection':'prefer second image in top five; fallback to globally disjoint top-10 image'},
          'queries':queries,'teachers':teachers}
    return plan,schedules

def prepare(s):
    if s==1:
        cfg,src,ident=previous.source();previous.predictions('vision');previous.predictions('imagery')
        return
    cfg,source,identity,data,rows,splits,natural=data_parts(s);exp=ROOT/cfg['experiment_root']
    nat.prepare(cfg,exp,source)

def retrieval(s,device):
    if s==1:return
    cfg,source,ident=context(s);exp=ROOT/cfg['experiment_root'];p=exp/'protocol/retrieval.json'
    if not p.is_file():nat.retrieval(cfg,exp,source,device)
    full=read(p);split=read(exp/'protocol/split.json');plan,schedules=ranked_plan(full,split)
    k10=subject_root(s)/'k10';immutable(k10/'protocol/split.json',split);immutable(k10/'protocol/retrieval.json',plan)
    for key,value in schedules.items():immutable(k10/f'protocol/R_{key}_pairs.json',value)
    immutable(k10/'protocol/source.json',{'subject':s,'source_retrieval_sha256':file_sha256(p),'brainvae_sha256':ident['brainvae_sha256'],'w010_sha256':ident['w010_sha256']})
    print('[K10]',json.dumps(plan['overview'],ensure_ascii=False),flush=True)

def teachers(s,device):
    if s==1:return
    cfg,source,_=context(s);nat.cache_teachers(cfg,subject_root(s)/'k10',source,device)

def prior(s,device):
    if s==1:return
    cfg,source,ident=context(s);k10=subject_root(s)/'k10';payload=k10/'teacher_prior.pt';receipt=k10/'teacher_prior.json'
    if receipt.exists():
        if file_sha256(payload)!=read(receipt)['sha256']:raise ValueError('Prior checksum mismatch')
        return
    if payload.exists():
        from multisubject_resume import archive_file
        print('[resume] preserving unverified prior:',archive_file(payload),flush=True)
    retrieval=read(k10/'protocol/retrieval.json');split=read(k10/'protocol/split.json');data=nat.load_trials(source)
    nsd=np.load(ROOT/f'data/processed/nsd/subj{s:02d}/nsd_train_fmri_scale_sub{s}.npy',mmap_mode='r')
    brain,mapper,_=nat.load_frozen(cfg,source,device,sdpa=False);del mapper
    means=[];logvars=[];details=[]
    with torch.inference_mode():
        for target in split['targets']:
            anchors=[r for r in retrieval['queries'][target] if r['teacher_split']=='train'];mu=[];lv=[]
            for start in range(0,len(anchors),4):
                rows=[retrieval['teachers'][r['teacher_index']] for r in anchors[start:start+4]]
                dist=brain.encode(nat.teacher_voxels(rows,nsd,data,source['normalizer']).to(device));mu.append(dist.mean.cpu());lv.append(dist.logvar.cpu())
            m,l,info=aggregate_anchor_parameters(torch.cat(mu),torch.cat(lv),[r['image_id'] for r in anchors],[r['similarity'] for r in anchors])
            means.append(m);logvars.append(l);details.append({'target':target,**info})
    nat.atomic_checkpoint({'mean':torch.stack(means),'logvar':torch.stack(logvars),'targets':split['targets']},payload)
    atomic_json({'status':'complete','subject':s,'sha256':file_sha256(payload),'brainvae_sha256':ident['brainvae_sha256'],
                 'retrieval_sha256':file_sha256(k10/'protocol/retrieval.json'),'anchor_groups':details},receipt)
    del brain;gc.collect();torch.cuda.empty_cache()

def old_run(m):
    return ROOT/('output_1104/imagery_clipl_direct_v1' if m=='w010_mse' else 'output_1104/imagery_mu_clip_mse_v1')

def train(s,m,device):
    if s==1:return
    cfg,source,ident,data,rows,splits,natural=data_parts(s);k10=subject_root(s)/'k10';folder=method_root(s,m)/'runs';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'training_complete.json';best=folder/'best.pt';last=folder/'last.pt'
    if marker.exists():
        if not best.exists() or file_sha256(best)!=read(marker)['best_sha256']:raise ValueError('Completed model changed')
        if last.exists():last.unlink()  # Only this run's completed temporary resume state.
        return
    v=data['fmri_scaled'].shape[-1];required=8*1024**3+7*(v*v+v)*4
    if shutil.disk_usage(OUT).free<required:raise OSError(f'Not enough free disk for safe resumable checkpoints: need {required/2**30:.1f} GiB')
    priorinfo=read(k10/'teacher_prior.json');path=k10/'teacher_prior.pt'
    if (file_sha256(path)!=priorinfo['sha256'] or priorinfo.get('subject')!=s
            or priorinfo['brainvae_sha256']!=ident['brainvae_sha256']
            or priorinfo['retrieval_sha256']!=file_sha256(k10/'protocol/retrieval.json')):raise ValueError('Prior changed')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['targets']!=[r['stimulus_id'] for r in rows[:6]]:raise ValueError('Prior target order mismatch')
    priors={k:p[k].to(device) for k in ('mean','logvar')}
    cache_meta=read(k10/'teacher_cache/state.json');cache_path=k10/'teacher_cache/teacher_tokens.npy'
    cache_spec=cache_meta['spec']
    if (not cache_meta.get('complete') or file_sha256(cache_path)!=cache_meta['cache_sha256']
            or cache_spec['brainvae_sha256']!=ident['brainvae_sha256'] or cache_spec['w010_sha256']!=ident['w010_sha256']
            or cache_spec['retrieval_sha256']!=file_sha256(k10/'protocol/retrieval.json')):raise ValueError('Teacher cache changed')
    cache=np.load(cache_path,mmap_mode='r');schedule=read(k10/'protocol/R_train_pairs.json')['batches']
    if len(schedule)!=18 or any(len(b)!=36 for b in schedule):raise ValueError('Unexpected K10 schedule')
    target_cache=ROOT/f'data/processed/imagery/subj{s:02d}/imagery_trials.npz'
    tokens,trows,_=load_imagery_truth(target_cache,s);lookup={(r['stimulus_type'],r['stimulus_id']):tokens[i] for i,r in enumerate(trows)}
    targets=torch.stack([lookup[(r['stimulus_type'],r['stimulus_id'])] for r in rows]).to(device)
    brain,mapper,_=nat.load_frozen(cfg,source,device,sdpa=True);mapping=nat.FrozenMapping(brain,mapper)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    adapter=ImageryResidualAdapter(v,'linear').to(device);optim=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=1.)
    signature={'protocol':'multisubject_w010_mu_mse_linear_v1','subject':s,'method':m,'source_brainvae':ident['brainvae_sha256'],
               'source_w010':ident['w010_sha256'],'data_fingerprint':ident['data_fingerprint'],
               'imagery_sha256':ident['imagery_trials_sha256'],'image_target_cache_sha256':file_sha256(target_cache),'prior_sha256':priorinfo['sha256'],
               'teacher_cache_sha256':cache_meta['cache_sha256'],'schedule_sha256':file_sha256(k10/'protocol/R_train_pairs.json'),
               'epochs':30,'updates_per_epoch':18,'batch':24,'micro':2,'seed':42,'g_natural':.01,'g_geometry':0.,
               'anchor_weight':.1,'kl_weight':.01,'validation':'w010 vs target image CLIP MSE',
               'worker_sha256':file_sha256(Path(__file__)),'loss_sha256':file_sha256(Path(wloss.__file__ if m=='w010_mse' else mloss.__file__))}
    start=0;bestval=float('inf')
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['signature']!=signature:raise ValueError('Checkpoint signature mismatch')
        adapter.load_state_dict(state['adapter']);optim.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);start=state['epoch'];bestval=state['best_val']
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    elif best.exists():
        # Interrupted before the first resumable epoch was saved: preserve the
        # unpaired best and restart this experiment from its original seed.
        from multisubject_resume import archive_file
        print('[resume] preserving unpaired best checkpoint:',archive_file(best),flush=True)
    loss=wloss if m=='w010_mse' else mloss
    for epoch in range(start,30):
        adapter.train()
        for batch_index,(x,y,teacher) in enumerate(wloss.batches(data,rows,splits,schedule,epoch)):
            x=x.to(device);y=y.to(device);optim.zero_grad(set_to_none=True)
            result=loss.backwards(adapter,mapping,x,y,teacher,cache,targets,priors,scaler)
            old=scaler.get_scale();scaler.step(optim);scaler.update()
            if scaler.get_scale()<old:raise FloatingPointError('AMP skipped an update')
            if batch_index%6==0:print(json.dumps({'subject':s,'method':m,'epoch':epoch+1,'batch':batch_index+1,**result}),flush=True)
        val=wloss.validate(adapter,mapping,data,rows,splits,targets,device)
        if val<bestval:
            bestval=val;nat.atomic_checkpoint({'adapter':adapter.state_dict(),'signature':signature,'epoch':epoch+1,'val_mse':val},best)
        nat.atomic_checkpoint({'adapter':adapter.state_dict(),'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),
              'signature':signature,'epoch':epoch+1,'best_val':bestval,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},last)
        print(json.dumps({'subject':s,'method':m,'epoch':epoch+1,'val_mse':val,'best_val':bestval}),flush=True)
    digest=file_sha256(best);atomic_json({'status':'complete','subject':s,'method':m,'signature':signature,'best_sha256':digest,
        'last_policy':'temporary resume checkpoint removed only after complete training and best checksum verification'},marker)
    if file_sha256(best)!=digest:raise ValueError('Best checkpoint changed before cleanup')
    last.unlink()
    del adapter,mapping,brain,mapper,optim,cache;gc.collect();torch.cuda.empty_cache()

def output_values(s):
    cfg,source,ident,data,rows,splits,natural=data_parts(s);tasks={}
    if s==1:return previous.task_values(source)
    _root,_source,_identity,_model,vr,v=vision.source_inputs(s)
    records=[{'subject_id':s,'stimulus_id':r['stimulus_id'],'stimulus_type':r['stimulus_type'],'task_type':'vision',
              'n_trials':8,'source_trial_ids':r['source_trial_ids'],'runs':r['runs']} for r in vr]
    tasks['vision']=(records,v)
    records=[];values=[]
    for r in rows:
        ids=r['test_rows'];records.append({'subject_id':s,'stimulus_id':r['stimulus_id'],'stimulus_type':r['stimulus_type'],
             'task_type':'imagery','n_trials':4,'source_trial_ids':r['source_trial_ids'],'runs':r['runs']})
        values.append(np.asarray(data['fmri_scaled'][ids],dtype=np.float32).mean(0))
    tasks['imagery']=(records,np.stack(values));return tasks

def predict(s,m,device):
    cfg,source,ident=context(s);tasks=output_values(s);dest=method_root(s,m)
    if s==1:
        root=old_run(m);marker=read(root/'linear/runs/training_complete.json');best=root/'linear/runs/best.pt'
    else:
        root=dest;marker=read(root/'runs/training_complete.json');best=root/'runs/best.pt'
    if file_sha256(best)!=marker['best_sha256']:raise ValueError('Model checksum mismatch')
    if s!=1:
        state=torch.load(best,map_location='cpu',weights_only=False);brain,mapper,_=nat.load_frozen(cfg,source,device,sdpa=True)
        mapping=nat.FrozenMapping(brain,mapper);adapter=ImageryResidualAdapter(tasks['vision'][1].shape[-1],'linear').to(device).eval();adapter.load_state_dict(state['adapter'])
    for task,(records,values) in tasks.items():
        folder=dest/'initial'/task;token=folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
        if s==1:
            origin=root/'linear/initial'/task/'tokens/chunk00000.pt';oldmeta=read(root/'linear/initial'/task/'metadata.json')
            if file_sha256(origin)!=oldmeta['token_sha256'] or oldmeta['checkpoint_sha256']!=marker['best_sha256']:raise ValueError('Subj01 original tokens differ')
            payload=torch.load(origin,map_location='cpu',weights_only=False)
            if payload['records']!=records:raise ValueError('Original trial order differs')
            if token.exists() and file_sha256(token)!=file_sha256(origin):raise ValueError('Existing tokens changed')
            if not token.exists():shutil.copyfile(origin,token)
        else:
            with torch.inference_mode():
                result=torch.cat([mapping(adapter(x.to(device))).float().cpu() for x in torch.from_numpy(values.astype(np.float32)).split(1)])
            if tuple(result.shape)!=(12,256,1664) or not torch.isfinite(result).all():raise ValueError('Invalid predicted tokens')
            payload={'mean':result,'records':records,'representation':'mapped_raw_clip_tokens','adapter_sha256':marker['best_sha256'],'mapping_checkpoint_sha256':ident['w010_sha256']}
            if token.exists():
                old=torch.load(token,map_location='cpu',weights_only=False)
                if old['records']!=records or old['adapter_sha256']!=marker['best_sha256'] or not torch.equal(old['mean'],result):raise ValueError('Existing tokens differ')
            else:nat.atomic_checkpoint(payload,token)
        meta={'protocol':'multisubject_linear_mse_3samples_v1','subject':s,'dataset':task,'method':m,'K':10,
              'aggregation':'mean_fmri','n_items':12,'num_samples':3,'num_steps':38,'seed':42,'partial_test':False,'debug_training':False,
              'checkpoint_sha256':marker['best_sha256'],'adapter_sha256':marker['best_sha256'],'w010_sha256':ident['w010_sha256'],
              'brainvae_sha256':ident['brainvae_sha256'],'data_fingerprint':ident['data_fingerprint'],'token_sha256':file_sha256(token),
              'target_usage':'all 12 stimuli used in adapter training; imagery test uses held-out trials'}
        immutable(folder/'metadata.json',meta)
    atomic_json({'status':'complete','subject':s,'method':m,'checkpoint_sha256':marker['best_sha256']},dest/'prediction_complete.json')
    if s!=1:del mapping,brain,mapper,adapter;gc.collect();torch.cuda.empty_cache()

def sample00_reuse(s,m,task,records):
    if s!=1:return {}
    root=old_run(m);folder=root/'linear/initial'/task
    saved=sweep.SAMPLES;sweep.SAMPLES=1;sweep.experiment=lambda _k:root/'linear'
    try:
        if not sweep.evaluation_ok(10,task,'initial'):raise ValueError('Existing subj01 initial evaluation invalid')
    finally:sweep.SAMPLES=saved
    items=read(folder/'sample00/manifest.json')['items'];reused={}
    for i,(row,item) in enumerate(zip(records,items)):
        if any(row[k]!=item[k] for k in ('stimulus_id','stimulus_type','source_trial_ids')) or item['seed']!=42+i:raise ValueError('Subj01 initial image trial mismatch')
        path=folder/'sample00'/item['path'];reused[(0,i)]=(path,file_sha256(path))
    return reused

def generate(s,m,device):
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    cfg,source,ident=context(s);_,_,paths=load_mapping_config(source['config_path'],ROOT)
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint']);unclip=unclip if unclip.is_absolute() else ROOT/unclip
    engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
    for task in TASKS:
        folder=method_root(s,m)/'initial'/task;meta=read(folder/'metadata.json');token=folder/'tokens/chunk00000.pt'
        if file_sha256(token)!=meta['token_sha256']:raise ValueError('Tokens changed')
        payload=torch.load(token,map_location='cpu',weights_only=False);records=payload['records'];targets={}
        for row in records:
            path=paths.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages'/('setB' if row['stimulus_type']=='complex' else 'setA')/(row['stimulus_id']+'.png')
            if not path.is_file():raise FileNotFoundError(path)
            targets[row['stimulus_id']]=path
        reuse=sample00_reuse(s,m,task,records)
        generate_samples_verified([token],folder,meta,engine,suffix,offset,partial(unclip_decode_batch,rgb_mode='standard'),save_image,3,42,targets,reuse=reuse)
        if not all(sweep.checked_sample(folder,i) for i in range(3)):raise ValueError('Generation incomplete')
    atomic_json({'status':'complete','subject':s,'method':m,'samples':3,'targets':12},method_root(s,m)/'generation_complete.json')
    del engine;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--subject',required=True,type=int,choices=range(1,9));p.add_argument('--method',choices=METHODS)
    p.add_argument('--stage',required=True,choices=('preflight','prepare','retrieval','teachers','prior','train','predict','generate'))
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='preflight':
        cfg,source,ident,data,rows,splits,natural=data_parts(a.subject)
        print(json.dumps({'status':'ok','subject':a.subject,'voxels':data['fmri_scaled'].shape[-1],'targets':len(rows),'brainvae':ident['brainvae_sha256'],'w010':ident['w010_sha256']}),flush=True);return
    if a.stage=='prepare':prepare(a.subject);return
    if a.stage in ('train','predict','generate') and not a.method:p.error('--method required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device=torch.device(a.device)
    {'retrieval':lambda:retrieval(a.subject,device),'teachers':lambda:teachers(a.subject,device),'prior':lambda:prior(a.subject,device),
     'train':lambda:train(a.subject,a.method,device),'predict':lambda:predict(a.subject,a.method,device),'generate':lambda:generate(a.subject,a.method,device)}[a.stage]()
if __name__=='__main__':main()
