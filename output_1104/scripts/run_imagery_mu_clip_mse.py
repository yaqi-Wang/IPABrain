#!/usr/bin/env python3
"""subj01: BrainVAE posterior mean to CLIP image MSE; previous anchor terms preserved."""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import run_imagery_w010_pilot as pilot
import run_k60_vision_imagery as previous
import sweep_subj01_retrieval_k as sweep
from check_scheme1_clip import load_imagery_truth
from src.data.scheme1 import file_sha256
from src.models.imagery_residual_comparison import ImageryResidualAdapter
from src.utils.clip_mapping import mapping_lock
from src.utils.imagery_r_kl import diagonal_gaussian_kl
from src.utils.imagery_w010_pilot import anclip_loss
from src.utils.staged_queue import atomic_json

OUT=ROOT/'output_1104/imagery_mu_clip_mse_v1'
K10=ROOT/'output_1104/k_sweep_subj01/K010'
OLD=ROOT/'output_1104/imagery_k10_weighted_r_v_subj01_v1'
NAT=ROOT/'outputs_new/imagery_w010_subj01_natural_v1/protocol/split.json'
GEO=ROOT/'outputs_new/imagery_w010_subj01_expansion_v1/geometry/protocol/split.json'
METHODS=('linear',)

def read(p): return json.loads(Path(p).read_text())

def load_inputs():
    cfg,source,identity=previous.source()
    natural,geometry=read(NAT),read(GEO)
    data=pilot.load_trials(source)
    records=read(ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs/imagery_mu_manifest.json')['rows']
    tokens,token_records,_=load_imagery_truth(ROOT/'data/processed/imagery/subj01/imagery_trials.npz',1)
    token_by_id={(r['stimulus_type'],r['stimulus_id']): tokens[i] for i,r in enumerate(token_records)}
    targets=[]; splits={}; classes=[]
    for label,row in enumerate(records):
        sid=row['stimulus_id']; kind=row['stimulus_type']; s=natural if kind=='complex' else geometry
        z=s['rows'][sid]; keys=('train','val','test')
        if any(len(z[k])!=n for k,n in zip(keys,(10,2,4))): raise ValueError('Unexpected trial counts')
        if any(set(z[a]) & set(z[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('Train, validation and test trial overlap')
        if z['test']!=row['test_rows']: raise ValueError('Held-out test trial identities changed')
        for phase in keys:
            for index in z[phase]:
                if str(data['stimulus_id'][index])!=sid or str(data['task_type'][index])!='imagery':
                    raise ValueError('Imagery trial target mismatch')
        targets.append(token_by_id[(kind,sid)])
        splits[sid]={key:z[key] for key in keys};classes.append(kind)
    if len(records)!=12 or classes!=['complex']*6+['simple']*6: raise ValueError('Wrong imagery targets')
    prior=read(OLD/'teacher_prior.json')
    prior_path=OLD/'teacher_prior.pt'
    if file_sha256(prior_path)!=prior['sha256']: raise ValueError('Teacher prior changed')
    if prior['identity']['brainvae_sha256']!=identity['brainvae_sha256'] or prior['identity']['w010_sha256']!=identity['w010_sha256']:
        raise ValueError('Anchor and model source differ')
    target_ids=[r['stimulus_id'] for r in records]
    prior_tensors=torch.load(prior_path,map_location='cpu',weights_only=False)
    if prior_tensors['targets']!=target_ids[:6]:raise ValueError('Anchor target ordering mismatch')
    cache_state, cache=pilot.validated_teacher_cache(K10)
    schedule=read(K10/'protocol/R_train_pairs.json')['batches']
    if len(schedule)!=18 or any(len(b)!=36 for b in schedule):raise ValueError('K10 schedule mismatch')
    for batch in schedule:
        for r in batch:
            if r['parent_row'] not in splits[target_ids[r['label']]]['train']:
                raise ValueError('Natural training schedule includes held-out imagery trial')
    return cfg,source,identity,data,records,torch.stack(targets),splits,prior_tensors,cache,schedule


def batches(data,records,splits,schedule,epoch):
    # Preserve prior K10 natural pairs; add balanced geometry imagery trials.
    rng=np.random.default_rng(42+epoch)
    for b,batch in enumerate(schedule):
        x=[]; labels=[]; tid=[]
        counts={i:0 for i in range(6)}
        for row in batch:
            label=int(row['label'])
            if counts[label]>=2: continue
            counts[label]+=1
            x.append(data['fmri_scaled'][row['parent_row']]);labels.append(label);tid.append(int(row['teacher_index']))
        if any(v!=2 for v in counts.values()):raise ValueError('Natural classes unbalanced')
        for label in range(6,12):
            pool=splits[records[label]['stimulus_id']]['train']
            for idx in rng.choice(pool,size=2,replace=False):
                x.append(data['fmri_scaled'][idx]);labels.append(label)
        yield torch.from_numpy(np.asarray(x,dtype=np.float32)), torch.tensor(labels), np.asarray(tid,dtype=np.int64)


def mu_to_tokens(mapping, mean):
    """Use the existing frozen w010 path for the unchanged contrastive term."""
    return pilot.map_tokens(mapping.mapper, mean)


def backwards(adapter,mapping,inputs,labels,teacher_ids,cache,targets,prior,scaler,micro=2):
    """MSE(mu, image CLIP) + unchanged w010 anchor contrast and latent KL."""
    states=[];predictions=[]
    with torch.no_grad():
        for start in range(0,len(inputs),micro):
            states.append((torch.get_rng_state(), torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None))
            predictions.append(mapping(adapter(inputs[start:start+micro])).float())
    end_cpu=torch.get_rng_state();end_cuda=torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
    prediction=torch.cat(predictions).detach().requires_grad_(True)
    nat=labels<6
    if int(nat.sum())!=len(teacher_ids) or not bool(torch.all(nat[:len(teacher_ids)])):
        raise ValueError('Teacher indices do not align with natural rows')
    teacher=torch.from_numpy(np.asarray(cache[teacher_ids],dtype=np.float32)).to(inputs.device)
    contrast=anclip_loss(prediction[nat],teacher,labels[nat],temperature=.05)
    effective=.01*.1*float(nat.float().mean())
    if not torch.isfinite(contrast):raise FloatingPointError('Nonfinite anchor contrast')
    gradient,=torch.autograd.grad(effective*contrast,prediction)
    del teacher,prediction
    mse_total=0.;kl_total=0.
    for ix,start in enumerate(range(0,len(inputs),micro)):
        torch.set_rng_state(states[ix][0])
        if inputs.is_cuda:torch.cuda.set_rng_state(states[ix][1],inputs.device)
        end=min(start+micro,len(inputs));part=inputs[start:end]
        adapted=adapter(part)
        def posterior_parameters(value):
            post=mapping.brain.encode(value)
            return post.mean,post.logvar
        mean,logvar=checkpoint(posterior_parameters,adapted,use_reentrant=False)
        if mean.shape != targets[labels[start:end]].shape:
            raise ValueError('BrainVAE posterior mean and image CLIP token shapes differ')
        mse=F.mse_loss(mean.float(),targets[labels[start:end]].float())
        ratio=(end-start)/len(inputs)
        # The replayed w010 output participates only in the unchanged anchor term.
        mapped=checkpoint(lambda value:mu_to_tokens(mapping,value),mean,use_reentrant=False)
        objective=ratio*mse+(mapped.float()*gradient[start:end]).sum()
        which=labels[start:end]<6
        if bool(which.any()):
            a=labels[start:end][which]
            kl=diagonal_gaussian_kl(mean[which],logvar[which],prior['mean'][a],prior['logvar'][a])
            fraction=int(which.sum())/len(inputs)
            objective=objective+.01*.01*fraction*kl
            kl_total+=fraction*float(kl.detach())
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite direct posterior loss')
        scaler.scale(objective).backward()
        mse_total+=ratio*float(mse.detach())
    torch.set_rng_state(end_cpu)
    if inputs.is_cuda:torch.cuda.set_rng_state(end_cuda,inputs.device)
    return {'mu_image_mse':mse_total,'anchor_contrast':float(contrast.detach()),'kl':kl_total,
            'g_natural':.01,'g_geometry':0.}


@torch.no_grad()
def validate(adapter,mapping,data,records,splits,targets,device):
    adapter.eval();losses=[]
    for label,row in enumerate(records):
        for idx in splits[row['stimulus_id']]['val']:
            x=torch.as_tensor(data['fmri_scaled'][idx],device=device,dtype=torch.float32).unsqueeze(0)
            out=mapping(adapter(x)).float()
            loss=F.mse_loss(out,targets[label:label+1])
            if not torch.isfinite(loss):raise FloatingPointError('Invalid validation loss')
            losses.append(float(loss))
    return float(np.mean(losses))


def train(method,device,epochs=30):
    cfg,source,identity,data,records,target,splits,prior,cache,schedule=load_inputs()
    target=target.to(device);prior={k:prior[k].to(device) for k in ('mean','logvar')}
    brain,mapper,_=pilot.load_frozen(cfg,source,device,sdpa=True)
    mapping=pilot.FrozenMapping(brain,mapper)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    adapter=ImageryResidualAdapter(data['fmri_scaled'].shape[-1],method).to(device)
    optim=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1.)
    folder=OUT/method/'runs';folder.mkdir(parents=True,exist_ok=True)
    signature={'method':method,'epochs':epochs,'source':identity,'natural_split_sha256':file_sha256(NAT),
               'geometry_split_sha256':file_sha256(GEO),'prior_sha256':file_sha256(OLD/'teacher_prior.pt'),
               'g_natural':.01,'g_geometry':0.,'anchor_weight':.1,'kl_beta':.01,'batch':24,'microbatch':2,
               'target':'brainvae_posterior_mean_vs_true_image_clip_tokens',
               'validation_selection':'w010_output_vs_true_image_clip_mse_preserved_for_comparison','seed':42,'only_seen_stimuli':True,
               'model_code_sha256':file_sha256(ROOT/'src/models/imagery_residual_comparison.py'),
               'script_sha256':file_sha256(Path(__file__))}
    last,best=folder/'last.pt',folder/'best.pt';start=0;bestval=float('inf')
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['signature']!=signature:raise ValueError('Checkpoint signature mismatch')
        adapter.load_state_dict(state['adapter']);optim.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start=state['epoch'];bestval=state['best_val'];torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    elif best.exists():raise ValueError('best.pt without resumable last.pt')
    for epoch in range(start,epochs):
        adapter.train();skip=0
        for batch_idx,(x,y,teacher) in enumerate(batches(data,records,splits,schedule,epoch)):
            x=x.to(device);y=y.to(device);optim.zero_grad(set_to_none=True)
            result=backwards(adapter,mapping,x,y,teacher,cache,target,prior,scaler)
            old=scaler.get_scale();scaler.step(optim);scaler.update();skip+=int(scaler.get_scale()<old)
            if skip:raise FloatingPointError('AMP skipped an update; restart from previous completed epoch')
            if batch_idx%6==0:print(json.dumps({'method':method,'epoch':epoch+1,'batch':batch_idx+1,**result}),flush=True)
        val=validate(adapter,mapping,data,records,splits,target,device)
        if val<bestval:
            bestval=val;pilot.atomic_checkpoint({'adapter':adapter.state_dict(),'signature':signature,'epoch':epoch+1,'val_mse':val},best)
        pilot.atomic_checkpoint({'adapter':adapter.state_dict(),'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),
            'signature':signature,'epoch':epoch+1,'best_val':bestval,'cpu_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all()},last)
        print(json.dumps({'method':method,'epoch':epoch+1,'val_mse':val,'best_val':bestval,'amp_skipped':skip}),flush=True)
        if skip:raise FloatingPointError('AMP skipped an update; inspect numeric stability')
    atomic_json({'status':'complete','best_sha256':file_sha256(best),'epochs':epochs,'signature':signature},folder/'training_complete.json')


def predict(method,device):
    cfg,source,identity,data,records,target,splits,prior,cache,schedule=load_inputs()
    ckpt=OUT/method/'runs/best.pt'; done=read(OUT/method/'runs/training_complete.json')
    if file_sha256(ckpt)!=done['best_sha256']:raise ValueError('Checkpoint checksum mismatch')
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    brain,mapper,_=pilot.load_frozen(cfg,source,device,sdpa=True);mapping=pilot.FrozenMapping(brain,mapper)
    adapter=ImageryResidualAdapter(data['fmri_scaled'].shape[-1],method).to(device).eval()
    adapter.load_state_dict(state['adapter'],strict=True)
    for task,(rows,values) in previous.task_values(source).items():
        with torch.inference_mode():
            output=torch.cat([mapping(adapter(x.to(device))).float().cpu() for x in torch.from_numpy(values.astype(np.float32)).split(1)])
        if tuple(output.shape)!=(12,256,1664) or not torch.isfinite(output).all():raise FloatingPointError('Invalid output')
        folder=OUT/method/'initial'/task;token=folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
        pilot.atomic_checkpoint({'mean':output,'records':rows,'representation':'mapped_raw_clip_tokens',
            'mapping_checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':done['best_sha256']},token)
        atomic_json({'protocol':'imagery_mu_clip_mse_v1','subject':1,'dataset':task,'method':method,'K':10,
            'aggregation':'mean_fmri','n_items':12,'num_samples':1,'num_steps':38,'seed':42,
            'checkpoint_sha256':done['best_sha256'],'w010_sha256':identity['w010_sha256'],
            'brainvae_sha256':identity['brainvae_sha256'],'data_fingerprint':identity['data_fingerprint'],
            'token_sha256':file_sha256(token),'partial_test':False,'debug_training':False,
            'target_usage':'all 12 training stimuli seen; held-out imagery trials'},folder/'metadata.json')
    atomic_json({'status':'complete','method':method},OUT/method/'prediction_complete.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preflight','smoke','train','predict','generate'],required=True)
    p.add_argument('--method',choices=METHODS)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=30)
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='preflight':
        *_,records,tokens,splits,prior,cache,schedule=load_inputs()
        print(json.dumps({'status':'ok','targets':len(records),'target_shape':list(tokens.shape),
                          'primary_mse':'BrainVAE posterior mean vs image CLIP',
                          'validation_selection':'unchanged w010 output vs image CLIP MSE',
                          'train_per_target':10,'validation_per_target':2,'test_per_target':4,
                          'natural_anchor_g':.01,'geometry_anchor_g':0.,'epochs':a.epochs}),flush=True);return
    if not a.method: p.error('--method required')
    device=torch.device(a.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if a.stage=='smoke':
        cfg,source,identity,data,records,target,splits,prior,cache,schedule=load_inputs()
        brain,mapper,_=pilot.load_frozen(cfg,source,device,sdpa=True)
        mapping=pilot.FrozenMapping(brain,mapper)
        adapter=ImageryResidualAdapter(data['fmri_scaled'].shape[-1],a.method).to(device).train()
        optim=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=1.)
        x,y,teacher=next(batches(data,records,splits,schedule,0))
        # Two natural classes and two geometry classes, exactly balanced.
        selected=torch.tensor([0,1,2,3,12,13,14,15]); x=x[selected].to(device);y=y[selected].to(device)
        prior={k:prior[k].to(device) for k in ('mean','logvar')}
        result=backwards(adapter,mapping,x,y,teacher[:4],cache,target.to(device),prior,scaler)
        scaler.unscale_(optim)
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()):raise FloatingPointError('Nonfinite gradient')
        if any(p.grad is not None for p in mapping.parameters()):raise ValueError('Frozen mapping has gradients')
        old=scaler.get_scale();scaler.step(optim);scaler.update()
        if scaler.get_scale()<old:raise FloatingPointError('AMP skipped smoke update')
        print(json.dumps({'status':'smoke_ok','method':a.method,'loss':result,
                          'gpu_peak_gib':torch.cuda.max_memory_allocated()/1024**3}),flush=True)
    elif a.stage=='train':train(a.method,device,a.epochs)
    elif a.stage=='predict':predict(a.method,device)
    else:
        sweep.experiment=lambda _k: OUT/a.method
        sweep.TASKS=('vision','imagery')
        sweep.generate(10,device)

if __name__=='__main__':main()
