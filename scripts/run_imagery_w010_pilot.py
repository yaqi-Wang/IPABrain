"""Reproducible subj01 natural-imagery trial-held-out W010 comparison."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from _bootstrap import add_project_root
ROOT=add_project_root()
from src.data.scheme1 import file_sha256, load_prepared
from src.utils.imagery_w010_pilot import (split_trials,trial_topk,teacher_image_split,
                                          pair_records,balanced_schedule,alignment_loss)
from src.utils.staged_queue import atomic_json


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/imagery_w010_pilot.yaml')
    p.add_argument('--stage',required=True,choices=('prepare','retrieval','cache_teachers','benchmark','train_R','train_V','train_R_MSE','predict','generate','evaluate','report'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args()


def configuration(args):
    cfg_path=ROOT/args.config
    cfg=yaml.safe_load(cfg_path.read_text())
    exp=ROOT/cfg['experiment_root']
    if cfg['subject'] != 1 or cfg['stimulus_type'] != 'complex':
        raise ValueError('This pilot fixes subj01 and six complex/natural targets')
    if (cfg['split_seed'],cfg['logical_batch_size'],cfg['examples_per_class'],cfg['knn_trials']) != (42,36,6,1000):
        raise ValueError('Unexpected pilot protocol parameters')
    if (cfg['anclip_weight'],cfg['temperature'],cfg['num_samples'],cfg['num_steps']) != (0.1,0.05,3,38):
        raise ValueError('Unexpected loss or generation protocol')
    exp.mkdir(parents=True,exist_ok=True)
    return cfg,exp,cfg_path


def run_source(cfg):
    selected=json.loads((ROOT/'outputs_new/w010_multisubject_v1/subj01/selection.json').read_text())['selected']
    checkpoint=ROOT/cfg['w010_checkpoint']
    config=ROOT/cfg['w010_config']
    if str(checkpoint)!=selected['checkpoint'] or str(config)!=selected['config']:
        raise ValueError('Pilot must use the recorded subj01 W010 checkpoint and config')
    digest=file_sha256(checkpoint)
    if digest!=selected['checkpoint_sha256'] or file_sha256(config)!=selected['config_sha256']:
        raise ValueError('W010 selected source hash mismatch')
    protocol=json.loads((checkpoint.parent.parent/'protocol.json').read_text())
    if protocol['subject']!=cfg['subject'] or protocol.get('debug_training'):
        raise ValueError('Incorrect W010 source')
    brain=Path(protocol['source_identity']['brainvae_checkpoint'])
    if file_sha256(brain)!=protocol['source_identity']['brainvae_checkpoint_sha256']:
        raise ValueError('Frozen BrainVAE source hash mismatch')
    prepared=Path(protocol['prepared_data'])
    manifest,split,normalizer=load_prepared(prepared)
    if manifest['subject']!=cfg['subject'] or normalizer.description!='beta_divisor_2000_no_zscore':
        raise ValueError('Unexpected source normalization or subject')
    if manifest.get('source_sample_check',{}).get('status')!='verified':
        raise ValueError('Source voxel order/scale has not been verified')
    return {'selected':selected,'protocol':protocol,'prepared':prepared,'manifest':manifest,
            'split':split,'normalizer':normalizer,'checkpoint':checkpoint,'config_path':config}


def load_trials(source):
    with np.load(source['prepared']/'imagery_trials.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def sha_config(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def require_identity(cfg,exp,source):
    path=exp/'protocol/identity.json'
    if not path.exists(): raise FileNotFoundError('Run prepare first')
    identity=json.loads(path.read_text())
    if (identity['config_sha256']!=sha_config(cfg) or
        identity['w010_sha256']!=source['selected']['checkpoint_sha256'] or
        identity['data_fingerprint']!=source['manifest']['data_fingerprint']):
        raise ValueError('Source/config changed; use a new experiment root')
    return identity


def prepare(cfg,exp,source):
    data=load_trials(source)
    split=split_trials(data,cfg['split_seed'],cfg['stimulus_type'])
    for target,row in split['rows'].items():
        row['trial_ids']={k:[int(data['global_trial_idx'][r]) for r in row[k]]
                          for k in ('train','val','test','vision_train','vision_val')}
        row['runs']={k:[int(data['run'][r]) for r in row[k]]
                     for k in ('train','val','test','vision_train','vision_val')}
    from prepare_nsd_perception_fmri import load_expdesign
    paths=yaml.safe_load((ROOT/'configs/paths.example.yaml').read_text())
    raw_root=Path(paths['raw_nsd_root'])
    design=load_expdesign(raw_root/'nsddata/experiments/nsd/nsd_expdesign.mat')
    shared=np.asarray(design['subjectim'][cfg['subject']-1,:1000],dtype=np.int64)-1
    arrays=source['split'];source_ids=arrays['source_image_ids'];idx=arrays['train_indices']
    candidates=source_ids[idx]
    for name,forbidden in [('NSD test',arrays['test_image_ids']),('shared1000',shared),('imagery targets',arrays['imagery_nsd_ids'])]:
        overlap=np.intersect1d(candidates,forbidden)
        if len(overlap): raise ValueError(f'Retrieval candidate overlaps {name}: {overlap.tolist()}')
    identity={'protocol':'subj01_imagery_w010_natural_trial_holdout_v1',
              'subject':cfg['subject'],'config':cfg,'config_sha256':sha_config(cfg),
              'w010_checkpoint':str(source['checkpoint']),'w010_sha256':source['selected']['checkpoint_sha256'],
              'brainvae_checkpoint':source['protocol']['source_identity']['brainvae_checkpoint'],
              'brainvae_sha256':source['protocol']['source_identity']['brainvae_checkpoint_sha256'],
              'data_fingerprint':source['manifest']['data_fingerprint'],
              'imagery_trials_sha256':file_sha256(source['prepared']/'imagery_trials.npz'),
              'normalization':source['normalizer'].description,'voxel_order':source['manifest']['voxel_order'],
              'source_training_images':len(candidates),'source_validation_images':len(arrays['val_indices']),
              'nsd_test_images':len(arrays['test_image_ids']), 'shared1000_excluded':True,
              'generalization':'held-out imagery trials for previously trained stimuli',
              'note':'subject-specific w010 epoch004; no target/teacher input at test'}
    folder=exp/'protocol';folder.mkdir(parents=True,exist_ok=True)
    for name,value in [('identity.json',identity),('split.json',split)]:
        path=folder/name
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError(f'Existing immutable protocol differs: {path}')
        atomic_json(value,path)
    print(json.dumps({'stage':'prepare','n_targets':len(split['targets']),
                      'imagery_train':60,'imagery_val':12,'imagery_test':24,
                      'vision_train':36,'vision_val':12,'source_training_images':len(candidates)},ensure_ascii=False),flush=True)


def clip_image_embedder(device,paths):
    sys.path.append(str(paths.generative_models_path))
    sys.path.append(str(paths.generative_models_path.parent))
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    from src.data.nsd_sdxl_features import OPENCLIP_ARCH
    return FrozenOpenCLIPImageEmbedder(arch=OPENCLIP_ARCH,version=str(paths.openclip_version),
                                     output_tokens=True,only_tokens=False).to(device).eval()


def retrieval(cfg,exp,source,device):
    from torch.utils.data import DataLoader,Subset
    from src.data.nsd_sdxl_features import NSDImageDataset,find_stimuli_hdf5
    from src.utils.clip_mapping import load_mapping_config
    from prepare_nsd_perception_fmri import load_expdesign,build_ppbe_split
    identity=require_identity(cfg,exp,source)
    split=json.loads((exp/'protocol/split.json').read_text())
    _,_,paths=load_mapping_config(source['config_path'],ROOT)
    hdf=find_stimuli_hdf5(paths.raw_nsd_root)
    target_root=paths.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages/setB'
    target_images=[]
    for name in split['targets']:
        with Image.open(target_root/(name+'.png')) as im:
            image=np.asarray(im.convert('RGB').resize((224,224),Image.BICUBIC),dtype=np.float32)/255.
        target_images.append(torch.from_numpy(image).permute(2,0,1))
    embedder=clip_image_embedder(device,paths)
    source_ids=source['split']['source_image_ids'];source_indices=source['split']['train_indices']
    image_ds=NSDImageDataset(hdf,source_ids)
    sample_indices=np.asarray(source_indices)[np.linspace(0,len(source_indices)-1,12,dtype=np.int64)]
    sample_loader=DataLoader(Subset(image_ds,sample_indices.tolist()),batch_size=4,num_workers=0)
    with torch.inference_mode():
        _,query=embedder(torch.stack(target_images).to(device))
        query=query.float().cpu().numpy()
        computed=[]
        for batch in sample_loader:
            _,out=embedder(batch.to(device));computed.append(out.float().cpu().numpy())
    computed=np.concatenate(computed)
    pool_file=ROOT/'data/processed/nsd/subj01/nsd_sdxl_clip_pool_train_sub1.npy'
    pool=np.load(pool_file,mmap_mode='r')
    if pool.shape != (len(source_ids),query.shape[1]):
        raise ValueError('Cached retrieval pool dimensions mismatch')
    existing=np.asarray(pool[sample_indices],dtype=np.float32)
    cos=(computed*existing).sum(1)/(np.linalg.norm(computed,axis=1)*np.linalg.norm(existing,axis=1))
    norm=np.linalg.norm(computed,axis=1)/np.linalg.norm(existing,axis=1)
    compatible=bool(np.isfinite(cos).all() and np.min(cos)>=0.9999 and np.max(np.abs(norm-1))<0.01)
    candidate_indices=np.asarray(source_indices,dtype=np.int64)
    if compatible:
        bank=np.asarray(pool[candidate_indices],dtype=np.float32)
    else:
        print('[检索] 旧 pooled 特征不匹配当前 CLIP，重新提取训练候选池',flush=True)
        bank_parts=[]
        with torch.inference_mode():
            loader=DataLoader(Subset(image_ds,candidate_indices.tolist()),batch_size=8,num_workers=0)
            for i,batch in enumerate(loader):
                _,out=embedder(batch.to(device));bank_parts.append(out.float().cpu().numpy())
                if i%50==0:print(f'[检索] 新特征 {min((i+1)*8,len(candidate_indices))}/{len(candidate_indices)}',flush=True)
        bank=np.concatenate(bank_parts)
        np.save(exp/'protocol/recomputed_nsd_pooled.npy',bank)
    del embedder;gc.collect();torch.cuda.empty_cache()
    similarities=(query/np.linalg.norm(query,axis=1,keepdims=True)) @ (bank/np.linalg.norm(bank,axis=1,keepdims=True)).T
    # Conservatively remove any image nearly identical to any imagery target.
    near=np.flatnonzero((similarities>=0.9999).any(axis=0))
    excluded=[{'image_id':int(source_ids[candidate_indices[i]]),'max_similarity':float(similarities[:,i].max())} for i in near]
    keep=np.ones(len(candidate_indices),dtype=bool);keep[near]=False
    candidate_indices=candidate_indices[keep];bank=bank[keep]
    target_ids=source_ids[candidate_indices]
    repeats=np.load(ROOT/'data/processed/nsd/subj01/nsd_train_fmri_scale_sub1.npy',mmap_mode='r').shape[1]
    groups=trial_topk(query,bank,target_ids,repeats,cfg['knn_trials'])
    for group in groups:
        for row in group:row['source_row']=int(candidate_indices[row['source_row']])
    groups=teacher_image_split(groups,cfg['seed'],cfg['teacher_val_fraction'])
    design=load_expdesign(paths.raw_nsd_root/'nsddata/experiments/nsd/nsd_expdesign.mat')
    _,_,trial_lookup,_=build_ppbe_split(design,cfg['subject'],40)
    for group in groups:
        for row in group:
            raw_trials=sorted(trial_lookup[row['image_id']])[:repeats]
            if len(raw_trials)!=repeats:raise ValueError('Missing real NSD trial identity')
            row['global_trial_idx']=int(raw_trials[row['repeat_index']])
    train_ids={r['image_id'] for g in groups for r in g if r['teacher_split']=='train'}
    val_ids={r['image_id'] for g in groups for r in g if r['teacher_split']=='val'}
    assert not train_ids.intersection(val_ids)
    keys=sorted({(r['source_row'],r['repeat_index']) for g in groups for r in g})
    teachers=[{'teacher_index':i,'kind':'nsd_trial','source_row':row,'repeat_index':rep,
               'image_id':int(source_ids[row]),'global_trial_idx':int(sorted(trial_lookup[int(source_ids[row])])[rep])}
              for i,(row,rep) in enumerate(keys)]
    key_to_index={(r['source_row'],r['repeat_index']):r['teacher_index'] for r in teachers}
    for group in groups:
        for row in group:row['teacher_index']=key_to_index[(row['source_row'],row['repeat_index'])]
    for target in split['targets']:
        for split_name in ('vision_train','vision_val'):
            for raw_row in split['rows'][target][split_name]:
                teachers.append({'teacher_index':len(teachers),'kind':'vision_trial','source_row':raw_row,
                                 'target':target,'teacher_split':'train' if split_name.endswith('train') else 'val'})
    overview={'protocol':'trial_level_clip_retrieval_v1','model':'OpenCLIP ViT-bigG-14',
              'openclip_checkpoint':str(paths.openclip_version),'openclip_sha256':file_sha256(paths.openclip_version),
              'pool_source':str(pool_file) if compatible else str(exp/'protocol/recomputed_nsd_pooled.npy'),
              'sample_cache_cos_min':float(cos.min()),'sample_cache_norm_ratio':norm.tolist(),
              'cache_matches_current_model':compatible,'identical_image_exclusions':excluded,
              'k_unit':'individual_trial','k_per_query':cfg['knn_trials'],
              'teacher_train_val_grouped_by_image':True,
              'counts':[{'target':name,'train':sum(r['teacher_split']=='train' for r in g),
                         'val':sum(r['teacher_split']=='val' for r in g)} for name,g in zip(split['targets'],groups)],
              'unique_retrieved_trials':len(keys),'total_unique_teachers':len(teachers),
              'overlapping_query_teacher_trials':sum(map(len,groups))-len(keys)}
    atomic_json({'overview':overview,'queries':dict(zip(split['targets'],groups)),'teachers':teachers},exp/'protocol/retrieval.json')
    print(json.dumps(overview,ensure_ascii=False),flush=True)


def load_frozen(cfg,source,device,sdpa=False):
    from src.models.clip_mapping import build_clip_mapper
    from src.utils.clip_mapping import MAPPING_PROTOCOL,frozen_brain,load_mapping_config
    from src.utils.clip_head_evaluation import load_trained_head_source
    from src.utils.ppbe_training import configure_ppbe_execution
    head_cfg,brain_cfg,paths=load_mapping_config(source['config_path'],ROOT)
    checkpoint,loaded,protocol=load_trained_head_source(source['checkpoint'],head_cfg,brain_cfg,ROOT,
                                                        cfg['subject'],MAPPING_PROTOCOL)
    brain=frozen_brain(loaded,device)
    if sdpa:configure_ppbe_execution(brain,{'attention_backend':'sdpa'})
    mapper=build_clip_mapper(checkpoint['config']).to(device).eval().requires_grad_(False)
    mapper.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint,loaded
    return brain,mapper,paths


def encode_mean(brain,x):
    if x.ndim==2:x=x.unsqueeze(1)
    return brain.pre_projector_mean(brain.encoder(x)).float()


def map_tokens(mapper,mean):
    from src.utils.mapping_precision import predict_mapping
    return predict_mapping(mapper,mean,'fp16')


def teacher_voxels(rows,nsd,data,normalizer):
    values=[]
    for row in rows:
        if row['kind']=='nsd_trial':
            value=normalizer.transform(nsd[row['source_row'],row['repeat_index']],'nsd')
        elif row['kind']=='vision_trial':
            value=normalizer.transform(data['fmri_scaled'][row['source_row']],'imagery_scaled')
        else:raise ValueError('Unknown teacher type')
        values.append(value)
    return torch.from_numpy(np.stack(values).astype(np.float32))


def cache_teachers(cfg,exp,source,device):
    retrieval_path=exp/'protocol/retrieval.json'
    retrieval_data=json.loads(retrieval_path.read_text());teachers=retrieval_data['teachers']
    nsd=np.load(ROOT/'data/processed/nsd/subj01/nsd_train_fmri_scale_sub1.npy',mmap_mode='r')
    data=load_trials(source);directory=exp/'teacher_cache';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'teacher_tokens.npy';state_path=directory/'state.json'
    spec={'protocol':'frozen_trial_teacher_tokens_v1','retrieval_sha256':file_sha256(retrieval_path),
          'w010_sha256':source['selected']['checkpoint_sha256'],
          'brainvae_sha256':source['protocol']['source_identity']['brainvae_checkpoint_sha256'],
          'shape':[len(teachers),256,1664],'storage_dtype':'float16','brainvae_precision':'fp32',
          'w010_precision':'fp16','brainvae_attention':'original_vanilla'}
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['spec']!=spec:raise ValueError('Teacher cache belongs to a different protocol')
        if state['next_index']==len(teachers) and path.is_file():
            cache=np.load(path,mmap_mode='r')
            if list(cache.shape)!=spec['shape'] or cache.dtype!=np.float16:raise ValueError('Invalid teacher cache array')
            digest=file_sha256(path)
            if state.get('cache_sha256') and state['cache_sha256']!=digest:raise ValueError('Teacher cache hash changed')
            state.update(complete=True,cache_sha256=digest);atomic_json(state,state_path)
            print('[教师缓存] 已完成，复用',flush=True);return
    else:
        if path.exists():raise ValueError('Teacher cache has no provenance')
        state={'spec':spec,'next_index':0,'max_fp16_roundtrip_error':0.0}
        atomic_json(state,state_path)
    if path.exists():
        cache=np.lib.format.open_memmap(path,mode='r+')
        if list(cache.shape)!=spec['shape'] or cache.dtype!=np.float16:raise ValueError('Invalid teacher cache array')
    else:cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=tuple(spec['shape']))
    brain,mapper,_=load_frozen(cfg,source,device,sdpa=False)
    batch_size=4
    with torch.inference_mode():
        for start in range(state['next_index'],len(teachers),batch_size):
            stop=min(start+batch_size,len(teachers))
            x=teacher_voxels(teachers[start:stop],nsd,data,source['normalizer']).to(device)
            mu=brain.encode(x).mode();out=map_tokens(mapper,mu)
            if out.shape != (stop-start,256,1664) or not torch.isfinite(out).all():raise FloatingPointError('Invalid frozen teacher output')
            converted=out.to(torch.float16).float()
            error=float((out-converted).abs().max())
            state['max_fp16_roundtrip_error']=max(state['max_fp16_roundtrip_error'],error)
            if error>1e-5:raise ValueError(f'FP16 teacher cache round-trip error {error}')
            cache[start:stop]=out.cpu().numpy().astype(np.float16)
            cache.flush();state['next_index']=stop;atomic_json(state,state_path)
            if start%100==0 or stop==len(teachers):print(f'[教师编码] {stop}/{len(teachers)}',flush=True)
    del brain,mapper,cache;gc.collect();torch.cuda.empty_cache()
    state['cache_sha256']=file_sha256(path);state['complete']=True;atomic_json(state,state_path)
    print(json.dumps({'teacher_cache_complete':len(teachers),'max_roundtrip_error':state['max_fp16_roundtrip_error'],
                      'cache_sha256':state['cache_sha256']}),flush=True)


def make_schedules(cfg,exp):
    split=json.loads((exp/'protocol/split.json').read_text())
    retrieval_data=json.loads((exp/'protocol/retrieval.json').read_text())
    teachers=retrieval_data['teachers'];targets=split['targets']
    for portion in ('train','val'):
        groups=[]
        for label,name in enumerate(targets):
            q=[r['teacher_index'] for r in retrieval_data['queries'][name] if r['teacher_split']==portion]
            groups.append(pair_records(split['rows'][name][portion],q,cfg['seed']+label+(0 if portion=='train' else 1000)))
        r_schedule=balanced_schedule(groups,cfg['examples_per_class'],cfg['seed']+(0 if portion=='train' else 2000))
        rng=np.random.default_rng(cfg['seed']+(5000 if portion=='train' else 6000))
        vision={}
        for label,name in enumerate(targets):
            vals=[x['teacher_index'] for x in teachers if x.get('kind')=='vision_trial' and x.get('target')==name and x.get('teacher_split')==portion]
            if len(vals)!=(6 if portion=='train' else 2):raise ValueError('Incorrect vision teacher split')
            vision[label]=list(map(int,rng.permutation(vals)))
        counts={i:0 for i in range(6)};v_schedule=[]
        for batch in r_schedule:
            output=[]
            for pair in batch:
                row=dict(pair);label=row['label'];choices=vision[label]
                row['teacher_index']=choices[counts[label]%len(choices)];counts[label]+=1
                output.append(row)
            v_schedule.append(output)
        atomic_json({'protocol':'deterministic_balanced_pairs_v1','portion':portion,'method':'R',
                     'note':'All retrieved trial pairs retained; short labels padded with recorded noisy copies for balanced full batches.',
                     'batches':r_schedule},exp/f'protocol/R_{portion}_pairs.json')
        atomic_json({'protocol':'deterministic_balanced_pairs_v1','portion':portion,'method':'V',
                     'note':'Same imagery parent/noise as R; six/two unique vision teachers reused.',
                     'batches':v_schedule},exp/f'protocol/V_{portion}_pairs.json')
        print(f'[配对] {portion}: {len(r_schedule)} batches ×36; V/R share identical imagery parents and noise',flush=True)


class FrozenMapping(torch.nn.Module):
    def __init__(self,brain,mapper):
        super().__init__();self.brain=brain;self.mapper=mapper
        self.eval().requires_grad_(False)

    def forward(self,x):
        if torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            mu=checkpoint(lambda value:encode_mean(self.brain,value),x,use_reentrant=False)
            return checkpoint(lambda value:map_tokens(self.mapper,value),mu,use_reentrant=False)
        return map_tokens(self.mapper,encode_mean(self.brain,x))


def make_adapter(cfg,voxels,device):
    from src.models.imagery_adapter import ImageryAlignmentAdapter
    return ImageryAlignmentAdapter(imagery_voxels=voxels,nsd_voxels=voxels,
                                   hidden_dim=cfg['adapter_hidden_dim'],
                                   dropout=cfg['adapter_dropout']).to(device)


def code_fingerprint():
    files=['scripts/run_imagery_w010_pilot.py','src/utils/imagery_w010_pilot.py',
           'src/models/imagery_adapter.py','src/models/brainvae_ppbe.py','src/models/ppbe_blocks.py',
           'src/models/clip_mapping.py','src/models/adatoken_refinement.py','src/s2n/sit.py',
           'src/utils/mapping_precision.py','scripts/decode_brainvae_scheme1.py',
           'scripts/decode_synthesised_fmri_unclip.py','scripts/eval_scheme1_reconstruction.py',
           'scripts/eval_unclip_image_semantics.py']
    return {name:file_sha256(ROOT/name) for name in files}


def benchmark(cfg,exp,source,device):
    import time
    from src.utils.imagery_w010_pilot import replay_backward
    split=json.loads((exp/'protocol/split.json').read_text());data=load_trials(source)
    rows=[r for name in split['targets'] for r in split['rows'][name]['train'][:6]]
    x=torch.from_numpy(np.asarray(data['fmri_scaled'][rows],dtype=np.float32)).to(device)
    labels=torch.arange(6,device=device).repeat_interleave(6)
    teacher_file=exp/'teacher_cache/teacher_tokens.npy'
    cache=np.load(teacher_file,mmap_mode='r');teacher=torch.from_numpy(np.array(cache[:36],dtype=np.float32)).to(device)
    brain,mapper,_=load_frozen(cfg,source,device,sdpa=False)
    model=FrozenMapping(brain,mapper)
    rows=[]
    for micro in (6,9,12,18,24):
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        adapter=make_adapter(cfg,x.shape[-1],device).train()
        opt=torch.optim.AdamW(adapter.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True)
        scaler=torch.amp.GradScaler('cuda',init_scale=256.)
        torch.cuda.reset_peak_memory_stats();result={'micro':micro}
        try:
            times=[]
            for step in range(2):
                torch.cuda.synchronize();started=time.monotonic();opt.zero_grad(set_to_none=True)
                loss=replay_backward(adapter,model,x,teacher,labels,micro,scaler,
                                     cfg['anclip_weight'],cfg['temperature'],check_replay=step==0)
                scaler.unscale_(opt)
                if not all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite adapter gradients')
                if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Frozen model got a gradient')
                scaler.step(opt);scaler.update()
                torch.cuda.synchronize();times.append(time.monotonic()-started)
            result.update(seconds=times[-1],peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                          loss=loss['total'],replay_max_abs=loss['replay_max_abs'],status='ok')
        except torch.cuda.OutOfMemoryError:
            result.update(status='out_of_memory');print(f'[测速] micro={micro} 显存不足',flush=True)
        rows.append(result);print('[测速] '+json.dumps(result),flush=True)
        opt.zero_grad(set_to_none=True);del opt,adapter,scaler;gc.collect();torch.cuda.empty_cache()
    valid=[r for r in rows if r['status']=='ok' and r['peak_allocated_gib']<20.5]
    if not valid:raise RuntimeError('No safe microbatch size on the current GPU')
    chosen=min(valid,key=lambda r:r['seconds'])
    report={'protocol':'frozen_imagery_mapping_gpu_benchmark_v1','logical_batch':36,
            'benchmark_batch':36,'candidates':rows,'chosen_microbatch':chosen['micro'],
            'note':'Synthetic optimizer updates on real train fMRI and frozen cached teacher tokens; no formal checkpoint written. Full36 contrast across microbatches in formal training.'}
    atomic_json({'files':code_fingerprint()},exp/'protocol/code_lock.json')
    atomic_json(report,exp/'protocol/gpu_benchmark.json')
    del model,brain,mapper;gc.collect();torch.cuda.empty_cache()
    print('[测速完成] '+json.dumps(report),flush=True)


def validated_teacher_cache(exp):
    state=json.loads((exp/'teacher_cache/state.json').read_text())
    path=exp/'teacher_cache/teacher_tokens.npy'
    if not state.get('complete') or state['next_index']!=state['spec']['shape'][0]:
        raise ValueError('Teacher cache is incomplete')
    if file_sha256(path)!=state.get('cache_sha256'):raise ValueError('Teacher cache checksum mismatch')
    data=np.load(path,mmap_mode='r')
    if list(data.shape)!=state['spec']['shape'] or data.dtype!=np.float16:raise ValueError('Teacher cache shape mismatch')
    return state,data


def augmented_inputs(data,split,schedule,scale):
    """Regenerate fixed parent/noise copies using training-only per-voxel sigma."""
    sigma={label:np.std(data['fmri_scaled'][split['rows'][name]['train']],axis=0,dtype=np.float64).astype(np.float32)
           for label,name in enumerate(split['targets'])}
    out=np.empty((len(schedule),36,data['fmri_scaled'].shape[-1]),dtype=np.float32)
    for i,batch in enumerate(schedule):
        for j,row in enumerate(batch):
            value=data['fmri_scaled'][row['parent_row']].astype(np.float32,copy=True)
            if row['augmented']:
                noise=np.random.default_rng(row['noise_seed']).standard_normal(len(value)).astype(np.float32)
                value+=scale*sigma[row['label']]*noise
            out[i,j]=value
    return out


def training_teacher_groups(retrieval_data,targets,method,portion):
    if method in ('R','R_MSE'):
        return [[r['teacher_index'] for r in retrieval_data['queries'][name] if r['teacher_split']==portion] for name in targets]
    return [[r['teacher_index'] for r in retrieval_data['teachers'] if r['kind']=='vision_trial'
             and r['target']==name and r['teacher_split']==portion] for name in targets]


def teacher_class_means(cache,groups,device):
    means=[]
    with torch.no_grad():
        for ids in groups:
            total=None
            for start in range(0,len(ids),12):
                t=torch.from_numpy(np.asarray(cache[ids[start:start+12]],dtype=np.float32)).to(device)
                flat=F.normalize(t.flatten(1),dim=1,eps=1e-12).sum(dim=0)
                total=flat if total is None else total+flat
            means.append(total/len(ids))
    return torch.stack(means)


@torch.no_grad()
def validate_adapter(adapter,mapping,data,split,schedule,inputs,cache,class_means,micro,cfg,device,weight):
    adapter.eval();losses=[];norms=[];teacher_norms=[]
    for index,batch in enumerate(schedule):
        x=torch.from_numpy(inputs[index]).to(device)
        teacher=torch.from_numpy(np.asarray(cache[[r['teacher_index'] for r in batch]],dtype=np.float32)).to(device)
        labels=torch.tensor([r['label'] for r in batch],device=device)
        pred=torch.cat([mapping(adapter(chunk)) for chunk in x.split(micro)])
        total,mse,con=alignment_loss(pred,teacher,labels,weight,cfg['temperature'])
        losses.append((float(total),float(mse),float(con)))
        norms.append(float(pred.float().flatten(1).norm(dim=1).mean()))
        teacher_norms.append(float(teacher.float().flatten(1).norm(dim=1).mean()))
    selected=[r for target in split['targets'] for r in split['rows'][target]['val']]
    x=torch.from_numpy(np.asarray(data['fmri_scaled'][selected],dtype=np.float32)).to(device)
    pred=torch.cat([mapping(adapter(chunk)) for chunk in x.split(micro)])
    values=F.normalize(pred.float().flatten(1),dim=1,eps=1e-12) @ class_means.T
    labels=torch.arange(6,device=device).repeat_interleave(2)
    same=values[torch.arange(len(labels),device=device),labels]
    other=(values.sum(1)-same)/5
    gap=same-other
    return {'loss':float(np.mean([x[0] for x in losses])), 'mse':float(np.mean([x[1] for x in losses])),
            'anclip':float(np.mean([x[2] for x in losses])), 'delta_val':float(gap.mean()),
            'same_cosine':float(same.mean()),'other_cosine':float(other.mean()),
            'token_norm_mean':float(np.mean(norms)), 'teacher_norm_mean':float(np.mean(teacher_norms)),
            'per_target_delta':{name:float(gap[2*i:2*i+2].mean()) for i,name in enumerate(split['targets'])},
            'original_imagery_val_trials':12,'fixed_pair_batches':len(schedule)}


def atomic_checkpoint(payload,path):
    from src.data.scheme1 import atomic_torch_save
    atomic_torch_save(payload,path)


def train_adapter(cfg,exp,source,device,method):
    import signal
    from src.utils.imagery_w010_pilot import replay_backward
    if method not in ('R','V','R_MSE'):raise ValueError(method)
    schedule_tag='R' if method in ('R','R_MSE') else 'V'
    split=json.loads((exp/'protocol/split.json').read_text())
    retrieval_data=json.loads((exp/'protocol/retrieval.json').read_text())
    schedules={k:json.loads((exp/f'protocol/{schedule_tag}_{k}_pairs.json').read_text())['batches'] for k in ('train','val')}
    data=load_trials(source);cache_state,cache_map=validated_teacher_cache(exp)
    print('[训练准备] 载入教师缓存到内存',flush=True)
    cache=np.array(cache_map,copy=True);del cache_map
    inputs={k:augmented_inputs(data,split,schedules[k],cfg['augmentation_noise_scale']) for k in schedules}
    benchmark_data=json.loads((exp/'protocol/gpu_benchmark.json').read_text())
    micro=int(benchmark_data['chosen_microbatch'])
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    adapter=make_adapter(cfg,data['fmri_scaled'].shape[-1],device)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True)
    scaler=torch.amp.GradScaler('cuda',init_scale=256.)
    brain,mapper,_=load_frozen(cfg,source,device,sdpa=False)
    mapping=FrozenMapping(brain,mapper)
    means=teacher_class_means(cache,training_teacher_groups(retrieval_data,split['targets'],method,'val'),device)
    out=exp/'runs'/method;out.mkdir(parents=True,exist_ok=True)
    signature={'method':method,'w010_sha256':source['selected']['checkpoint_sha256'],
               'brainvae_sha256':source['protocol']['source_identity']['brainvae_checkpoint_sha256'],
               'split_sha256':file_sha256(exp/'protocol/split.json'),
               'retrieval_sha256':file_sha256(exp/'protocol/retrieval.json'),
               'schedule_sha256':{k:file_sha256(exp/f'protocol/{schedule_tag}_{k}_pairs.json') for k in schedules},
               'teacher_cache_sha256':cache_state['cache_sha256'],'config_sha256':sha_config(cfg),
               'micro_batch':micro,'objective_weight':0.0 if method=='R_MSE' else cfg['anclip_weight'],
               'code_lock_sha256':file_sha256(exp/'protocol/code_lock.json')}
    completion=out/'training_complete.json'
    if completion.exists():
        record=json.loads(completion.read_text())
        if record['signature']!=signature or file_sha256(out/'best.pt')!=record['best_sha256']:
            raise ValueError('Completed adapter no longer matches its protocol')
        print(f'[复用训练] {method}',flush=True);return
    current={'epoch_index':0,'next_batch':0,'global_step':0,'best_score':None,'best_epoch':0,
             'stale_epochs':0,'history':[],'epoch_sums':{'total':0.,'mse':0.,'anclip':0.,'token_norm_mean':0.,'teacher_norm_mean':0.},'epoch_steps':0,
             'epoch_first_gradient':None,
             'amp_skipped_updates':0}
    last=out/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Adapter resume identity mismatch')
        adapter.load_state_dict(saved['adapter']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        current=saved['state'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],device)
        print(f'[恢复训练] {method}: epoch={current["epoch_index"]+1}, batch={current["next_batch"]}',flush=True)
    stop={'requested':False}
    def interrupt(signum,frame):stop['requested']=True
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    def save_last():
        atomic_checkpoint({'signature':signature,'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),
                           'scaler':scaler.state_dict(),'state':current,'torch_rng':torch.get_rng_state(),
                           'cuda_rng':torch.cuda.get_rng_state(device)},last)
    if not last.exists():save_last()
    weight=signature['objective_weight'];best_path=out/'best.pt'
    for epoch in range(current['epoch_index'],cfg['max_epochs']):
        if current['stale_epochs']>=cfg['patience']:break
        adapter.train();order=np.random.default_rng(cfg['seed']+epoch).permutation(len(schedules['train']))
        for offset in range(current['next_batch'],len(order)):
            batch_index=int(order[offset]);batch=schedules['train'][batch_index]
            x=torch.from_numpy(inputs['train'][batch_index]).to(device)
            teacher=torch.from_numpy(np.asarray(cache[[r['teacher_index'] for r in batch]],dtype=np.float32)).to(device)
            labels=torch.tensor([r['label'] for r in batch],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True)
            losses=replay_backward(adapter,mapping,x,teacher,labels,micro,scaler,weight,cfg['temperature'],
                                   check_replay=current['global_step']==0)
            before=scaler.get_scale();scaler.unscale_(optimizer)
            if current['epoch_steps']==0:
                grads=[p.grad for p in adapter.parameters() if p.grad is not None]
                finite=bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
                grad_norm=math.sqrt(sum(float(g.norm())**2 for g in grads)) if finite else None
                current['epoch_first_gradient']={'finite':finite,'l2_norm':grad_norm,'n_tensors':len(grads)}
                if current['global_step']==0 and (not finite or grad_norm==0):
                    raise FloatingPointError('First adapter gradients missing, zero or nonfinite')
                if any(p.grad is not None for p in mapping.parameters()):raise RuntimeError('Frozen source received gradients')
            scaler.step(optimizer);scaler.update()
            current['amp_skipped_updates']+=int(scaler.get_scale()<before)
            current['global_step']+=1;current['next_batch']=offset+1;current['epoch_steps']+=1
            for k in ('total','mse','anclip','token_norm_mean','teacher_norm_mean'):current['epoch_sums'][k]+=losses[k]
            if current['global_step']%50==0:
                print(f'[训练 {method}] epoch={epoch+1} batch={offset+1}/{len(order)} total={losses["total"]:.6f}',flush=True)
            if current['global_step']%200==0 or stop['requested']:
                save_last()
                if stop['requested']:raise SystemExit(143)
        val=validate_adapter(adapter,mapping,data,split,schedules['val'],inputs['val'],cache,means,micro,cfg,device,weight)
        score=val['delta_val'];improved=current['best_score'] is None or score>current['best_score']
        if improved:current.update(best_score=score,best_epoch=epoch+1,stale_epochs=0)
        else:current['stale_epochs']+=1
        history={'epoch':epoch+1,'train':{k:v/current['epoch_steps'] for k,v in current['epoch_sums'].items()},
                 'validation':val,'best_epoch':current['best_epoch'],'best_delta_val':current['best_score'],
                 'first_batch_adapter_gradient':current['epoch_first_gradient'],
                 'global_step':current['global_step'],'amp_skipped_updates':current['amp_skipped_updates']}
        current['history'].append(history)
        if improved:atomic_checkpoint({'signature':signature,'adapter':adapter.state_dict(),
                                       'epoch':epoch+1,'validation':val},best_path)
        current.update(epoch_index=epoch+1,next_batch=0,epoch_sums={'total':0.,'mse':0.,'anclip':0.,'token_norm_mean':0.,'teacher_norm_mean':0.},epoch_steps=0,epoch_first_gradient=None)
        save_last();atomic_json(current['history'],out/'history.json')
        print('[验证 '+method+'] '+json.dumps(history,ensure_ascii=False),flush=True)
        if current['stale_epochs']>=cfg['patience']:break
        if stop['requested']:raise SystemExit(143)
    if current['global_step']==current['amp_skipped_updates']:raise FloatingPointError('Every update skipped')
    atomic_json({'status':'complete','signature':signature,'best_epoch':current['best_epoch'],
                 'best_delta_val':current['best_score'],'epochs_completed':current['epoch_index'],
                 'global_step':current['global_step'],'amp_skipped_updates':current['amp_skipped_updates'],
                 'best_sha256':file_sha256(best_path)},completion)
    print(f'[训练完成] {method}',flush=True)
    del mapping,brain,mapper,adapter,optimizer,cache;gc.collect();torch.cuda.empty_cache()


def test_records(split,data):
    rows=[]
    for name in split['targets']:
        group=split['rows'][name]
        if len(group['test'])!=4 or set(group['test']) & (set(group['train'])|set(group['val'])):
            raise ValueError('Test trials are not held out')
        rows.append({'subject_id':1,'stimulus_id':name,'stimulus_type':'complex','task_type':'imagery',
                     'n_trials':4,'source_trial_ids':[int(data['global_trial_idx'][i]) for i in group['test']],
                     'runs':[int(data['run'][i]) for i in group['test']]})
    return rows


def predict(cfg,exp,source,device):
    split=json.loads((exp/'protocol/split.json').read_text());data=load_trials(source)
    records=test_records(split,data)
    values=np.stack([data['fmri_scaled'][split['rows'][name]['test']].mean(axis=0) for name in split['targets']])
    x=torch.from_numpy(values.astype(np.float32)).to(device)
    brain,mapper,paths=load_frozen(cfg,source,device,sdpa=False);mapping=FrozenMapping(brain,mapper)
    identity=json.loads((exp/'protocol/identity.json').read_text())
    split_sha=file_sha256(exp/'protocol/split.json')
    methods=('B0','R','V','R_MSE')
    for method in methods:
        folder=exp/'runs'/method/'evaluation'/'imagery';folder.mkdir(parents=True,exist_ok=True)
        tokens=folder/'tokens';tokens.mkdir(exist_ok=True)
        adapter_sha=None
        if method=='B0':
            adapter=None;checkpoint_sha=identity['w010_sha256']
        else:
            complete=json.loads((exp/'runs'/method/'training_complete.json').read_text())
            path=exp/'runs'/method/'best.pt';adapter_sha=file_sha256(path)
            if adapter_sha!=complete['best_sha256']:raise ValueError(f'{method} adapter changed')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload['signature']!=complete['signature']:raise ValueError('Adapter signature mismatch')
            adapter=make_adapter(cfg,x.shape[-1],device).eval().requires_grad_(False)
            adapter.load_state_dict(payload['adapter'],strict=True);checkpoint_sha=adapter_sha
        metadata={'protocol':'imagery_w010_trial_heldout_v1','subject':1,'dataset':'imagery',
                  'method':method,'aggregation':'mean_fmri','checkpoint_sha256':checkpoint_sha,
                  'adapter_sha256':adapter_sha,'w010_sha256':identity['w010_sha256'],
                  'brainvae_sha256':identity['brainvae_sha256'],'data_fingerprint':identity['data_fingerprint'],
                  'split_sha256':split_sha,'num_samples':cfg['num_samples'],'num_steps':cfg['num_steps'],
                  'seed':cfg['generation_seed'],'rgb_mode':cfg['rgb_mode'],'partial_test':False,
                  'debug_training':False,'test_used_for_selection':False,
                  'target_usage':'reference images are attached after model prediction and used only for evaluation',
                  'input':'mean of the four held-out imagery fMRI trials for each target'}
        with torch.inference_mode():
            adapted=x if adapter is None else adapter(x)
            output=torch.cat([mapping(chunk) for chunk in adapted.split(2)]).float().cpu()
        if not torch.isfinite(output).all():raise FloatingPointError(f'Nonfinite predicted tokens {method}')
        chunk=tokens/'chunk00000.pt'
        payload={'mean':output,'records':records,'representation':'mapped_raw_clip_tokens',
                 'mapping_checkpoint_sha256':identity['w010_sha256'],
                 'adapter_sha256':adapter_sha,'split_sha256':split_sha}
        if chunk.exists():
            previous=torch.load(chunk,map_location='cpu',weights_only=False)
            if (not torch.equal(previous['mean'],output) or previous['records']!=records or
                    any(previous.get(k)!=payload[k] for k in ('mapping_checkpoint_sha256','adapter_sha256','split_sha256'))):
                raise ValueError(f'Existing predictions differ; choose a new experiment: {method}')
        else:
            if any(folder.glob('sample*/images/*.png')):raise ValueError('Images exist without prediction provenance')
            atomic_checkpoint(payload,chunk)
        metadata['token_sha256']=file_sha256(chunk)
        old=folder/'metadata.json'
        if old.exists() and json.loads(old.read_text())!=metadata:raise ValueError('Existing generation metadata differs')
        atomic_json(metadata,folder/'metadata.json')
        print(f'[预测] {method}: six held-out trial means',flush=True)
        del adapter
    del mapping,brain,mapper;gc.collect();torch.cuda.empty_cache()
    atomic_json({'status':'complete','methods':list(methods),'n_targets':6,'split_sha256':split_sha},exp/'prediction_complete.json')


def generation_ok(cfg,exp):
    split=json.loads((exp/'protocol/split.json').read_text());targets=split['targets']
    for method in ('B0','R','V','R_MSE'):
        folder=exp/'runs'/method/'evaluation'/'imagery'
        metadata=json.loads((folder/'metadata.json').read_text())
        if file_sha256(folder/'tokens/chunk00000.pt')!=metadata['token_sha256']:return False
        for sample in range(cfg['num_samples']):
            s=folder/f'sample{sample:02d}';manifest=s/'manifest.json'
            if not manifest.exists():return False
            m=json.loads(manifest.read_text());items=m['items']
            if [r['stimulus_id'] for r in items]!=targets:return False
            for i,row in enumerate(items):
                if row['seed']!=cfg['generation_seed']+sample*100000+i:return False
                image=s/row['path']
                if row['n_trials']!=4 or not image.is_file() or not image.stat().st_size:return False
    return True


def generate(cfg,exp,source,device):
    from functools import partial
    from torchvision.utils import save_image
    from src.utils.clip_mapping import load_mapping_config
    from decode_brainvae_scheme1 import generate_samples
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    _,_,paths=load_mapping_config(source['config_path'],ROOT)
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute():unclip=ROOT/unclip
    engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,cfg['num_steps'])
    target_dir=paths.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages/setB'
    targets={p.stem:p for p in target_dir.glob('*.png')}
    for method in ('B0','R','V','R_MSE'):
        folder=exp/'runs'/method/'evaluation'/'imagery'
        metadata=json.loads((folder/'metadata.json').read_text())
        if file_sha256(folder/'tokens/chunk00000.pt')!=metadata['token_sha256']:
            raise ValueError('Prediction token chunk changed')
        generate_samples([folder/'tokens/chunk00000.pt'],folder,metadata,engine,suffix,offset,
                         partial(unclip_decode_batch,rgb_mode=cfg['rgb_mode']),save_image,
                         cfg['num_samples'],cfg['generation_seed'],targets,True)
    if not generation_ok(cfg,exp):raise RuntimeError('Incomplete generation artifacts')
    atomic_json({'status':'complete','methods':['B0','R','V','R_MSE'],
                 'images':6*cfg['num_samples']*4},exp/'generation_complete.json')
    del engine;gc.collect();torch.cuda.empty_cache()


def evaluate(cfg,exp,device):
    import subprocess
    for method in ('B0','R','V','R_MSE'):
        folder=exp/'runs'/method/'evaluation'/'imagery';log=exp/'logs'/f'eval_{method}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','scripts/eval_scheme1_reconstruction.py','--dataset-dir',str(folder),
                 '--models','Alex2,Alex5,Incep,CLIP,Eff,SwAV','--device',str(device),'--batch-size','8']
        with log.open('a') as f:
            f.write('\nCOMMAND '+json.dumps(command)+'\n');f.flush()
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        values=json.loads((folder/'image_metrics.json').read_text())
        required=('PixCorr','SSIM','Alex2_2way','Alex5_2way','Incep_2way','CLIP_2way','Eff_corr_distance','SwAV_corr_distance')
        if values['n_targets']!=6 or values['n_samples']!=3 or not all(math.isfinite(values['metrics'][x]) for x in required):
            raise ValueError(f'Invalid metrics: {method}')
        print(f'[评估完成] {method}',flush=True)
    atomic_json({'status':'complete','methods':['B0','R','V','R_MSE']},exp/'evaluation_complete.json')


def report(cfg,exp):
    import csv
    methods=('B0','R','V','R_MSE')
    keys=('PixCorr','SSIM','Alex2_2way','Alex5_2way','Incep_2way','CLIP_2way','Eff_corr_distance','SwAV_corr_distance')
    if not generation_ok(cfg,exp):raise ValueError('Generation not fully verified')
    rows=[]
    for method in methods:
        folder=exp/'runs'/method/'evaluation'/'imagery'
        metrics=json.loads((folder/'image_metrics.json').read_text())
        if metrics['n_targets']!=6 or metrics['n_samples']!=3:raise ValueError('Invalid evaluation scope')
        row={'method':method,**{k:metrics['metrics'][k] for k in keys}}
        if not all(math.isfinite(row[k]) for k in keys):raise ValueError('Nonfinite metrics')
        rows.append(row)
    with (exp/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method',*keys]);writer.writeheader();writer.writerows(rows)
    atomic_json({'status':'complete','subject':1,'targets':6,'samples_per_target':3,
                 'original_test_trials':24,'split_seed':42,'methods':rows,
                 'caveat':'Six trained stimulus identities with held-out trials; three diffusion samples and overlapping trials are not independent human subjects.'},exp/'summary.json')
    lines=['| 方法 | '+' | '.join(keys)+' |','|---|'+'---:|'*len(keys)]
    for row in rows:lines.append('| '+row['method']+' | '+' | '.join(f'{row[k]:.3f}' for k in keys)+' |')
    lines=['# subj01 自然图像想象试点：B0 / R / V / R-MSE', '',
           '固定 BrainVAE+w010；每目标 imagery 10/2/4，测试只使用四条留出 trial 的 fMRI 平均。六个目标，各生成三张，所有样本参与评分。',
           '这是训练见过的六个刺激的留出 trial 比较，不是未见刺激的零样本评价。六个目标、一个划分；不将生成种子当作独立人类样本。','',*lines,'',
           '每个方法的样本和完整评估文件位于 `runs/<方法>/evaluation/imagery/`；训练协议、试次划分和教师清单位于 `protocol/`。','']
    (exp/'summary.md').write_text('\n'.join(lines))
    print((exp/'summary.md').read_text(),flush=True)


def main():
    args=arguments();torch.set_num_threads(4)
    cfg,exp,_=configuration(args);source=run_source(cfg)
    if args.stage=='prepare':prepare(cfg,exp,source);return
    require_identity(cfg,exp,source)
    if args.stage not in ('retrieval','cache_teachers','benchmark'):
        lock=exp/'protocol/code_lock.json'
        if not lock.is_file() or json.loads(lock.read_text())['files']!=code_fingerprint():
            raise ValueError('Experiment source code changed after benchmarking; use a new run or explicitly audit the revision')
    if args.stage=='retrieval':retrieval(cfg,exp,source,torch.device(args.device));return
    if args.stage=='cache_teachers':cache_teachers(cfg,exp,source,torch.device(args.device));make_schedules(cfg,exp);return
    if args.stage=='benchmark':benchmark(cfg,exp,source,torch.device(args.device));return
    if args.stage.startswith('train_'):train_adapter(cfg,exp,source,torch.device(args.device),args.stage[6:]);return
    if args.stage=='predict':predict(cfg,exp,source,torch.device(args.device));return
    if args.stage=='generate':generate(cfg,exp,source,torch.device(args.device));return
    if args.stage=='evaluate':evaluate(cfg,exp,torch.device(args.device));return
    if args.stage=='report':report(cfg,exp);return
    raise NotImplementedError(args.stage)


if __name__=='__main__':main()
