#!/usr/bin/env python3
"""Subj01 nsdgeneral: reconstruct train/validation trial means of seen stimuli."""
import argparse
import csv
from contextlib import contextmanager
import gc
import json
import os
import signal
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
import run_k60_vision_imagery as prev
import run_imagery_w010_pilot as pilot
import sweep_subj01_retrieval_k as sweep
import refine_w010_mindeye2_img2img as refine_base
from multisubject_generate import generate_samples_verified
from multisubject_resume import archive_file
from src.data.scheme1 import file_sha256
from src.models.imagery_residual_comparison import ImageryResidualAdapter
from src.utils.clip_mapping import load_mapping_config, mapping_lock
from src.utils.staged_queue import StageQueue, atomic_json

OUT=ROOT/'output_1104/subj01_imagery_train_val_nsdgeneral_v1'
SPLITS={'complex':ROOT/'outputs_new/imagery_w010_subj01_natural_v1/protocol/split.json',
        'simple':ROOT/'outputs_new/imagery_w010_subj01_expansion_v1/geometry/protocol/split.json'}
ORIGINAL_NAT=ROOT/'outputs_new/imagery_w010_subj01_natural_v1/protocol/split.json'
METHODS=('B0','w010_mse','mu_mse')
GROUPS=('imagery_train','imagery_val','vision_train','vision_val')
RUNS={'w010_mse':ROOT/'output_1104/imagery_clipl_direct_v1/linear/runs',
      'mu_mse':ROOT/'output_1104/imagery_mu_clip_mse_v1/linear/runs'}
RIDGE=ROOT/'output_1104/nsd_subj01/ridge_three_text/ridge_mu_three_text_v3/model'
ACTIVE=None

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def reconcile_checkpoint_alias(source, expected):
    """Accept a moved checkpoint only when all provenance and file identity agree."""
    actual=source['identity']; historical=expected['brainvae_checkpoint']
    current=actual['brainvae_checkpoint']
    if not Path(historical).samefile(current):
        raise ValueError('BrainVAE checkpoint paths do not identify the same file')
    normalized={**actual,'brainvae_checkpoint':historical}
    if normalized!=expected:
        keys=[k for k in sorted(set(normalized)|set(expected)) if normalized.get(k)!=expected.get(k)]
        raise ValueError('BrainVAE identity differs beyond checkpoint path: '+str(keys))
    # The original head / Ridge equality checks remain active on all fields.
    source['identity']=normalized
    return source

@contextmanager
def head_checkpoint_alias(expected):
    import src.utils.clip_head_evaluation as head_eval
    original=head_eval.load_brain_source
    def load(*args,**kwargs):
        return reconcile_checkpoint_alias(original(*args,**kwargs),expected)
    head_eval.load_brain_source=load
    try: yield
    finally: head_eval.load_brain_source=original
def immutable(p, obj):
    p=Path(p)
    if p.exists():
        if read(p)!=obj: raise ValueError('Provenance mismatch: '+str(p))
    else: atomic_json(obj,p)

def inputs():
    cfg,source,ident=prev.source();data=pilot.load_trials(source)
    if file_sha256(source['prepared']/'imagery_trials.npz')!=ident['imagery_trials_sha256']:
        raise ValueError('Imagery trial source differs from adapter training')
    if data['fmri_scaled'].shape[1]!=15724: raise ValueError('Not subj01 nsdgeneral voxels')
    original=read(ORIGINAL_NAT);man=read(ROOT/'output_1104/imagery_ridge_refine_subj01_v1/inputs/imagery_mu_manifest.json')
    records=man['rows']; plans={key:[] for key in GROUPS}; values={key:[] for key in GROUPS}
    splits={kind:read(path) for kind,path in SPLITS.items()}
    if len(records)!=12: raise ValueError('Expected 12 stimuli')
    for record in records:
        sid,kind=record['stimulus_id'],record['stimulus_type']; split=splits[kind]; row=split['rows'][sid]
        if row['test']!=record['test_rows']:raise ValueError('Held-out imagery trial mismatch')
        for task in ('imagery','vision'):
            if task=='imagery':
                names,counts=('train','val','test'),(10,2,4)
            elif kind=='complex':
                # Use the natural pilot's original vision teacher split (6/2).
                names,counts=('vision_train','vision_val'),(6,2)
            else:
                names,counts=('vision_train','vision_val','vision_test'),(4,2,2)
            groups=[set(row[n]) for n in names]
            if [len(x) for x in groups]!=list(counts) or sum(map(len,groups))!=len(set.union(*groups)):
                raise ValueError('Invalid disjoint trial split: '+sid)
            for phase in ('train','val'):
                key=task+'_'+phase; idx=row[phase if task=='imagery' else key]
                if any(str(data['stimulus_id'][i])!=sid or str(data['stimulus_type'][i])!=kind or str(data['task_type'][i])!=task for i in idx):
                    raise ValueError('Trial identity mismatch: '+sid)
                x=np.asarray(data['fmri_scaled'][idx],dtype=np.float32).mean(0)
                if not np.isfinite(x).all(): raise ValueError('Nonfinite input')
                values[key].append(x)
                plans[key].append({'subject_id':1,'stimulus_id':sid,'stimulus_type':kind,'task_type':task,
                    'partition':phase,'trial_rows':list(map(int,idx)),'n_trials':len(idx),
                    'source_trial_ids':[int(data['global_trial_idx'][i]) for i in idx],
                    'runs':[int(data['run'][i]) for i in idx]})
    hashes={}
    for method,run in RUNS.items():
        done=read(run/'training_complete.json');best=run/'best.pt'
        if done.get('status')!='complete' or file_sha256(best)!=done['best_sha256']:raise ValueError('Adapter checkpoint mismatch '+method)
        src=done['signature']['source']
        if src['brainvae_sha256']!=ident['brainvae_sha256'] or src['w010_sha256']!=ident['w010_sha256']:raise ValueError('Adapter frozen model mismatch')
        if (done['signature']['natural_split_sha256']!=file_sha256(ORIGINAL_NAT) or
            done['signature']['geometry_split_sha256']!=file_sha256(SPLITS['simple'])):
            raise ValueError('Adapter training split changed: '+method)
        hashes[method]=done['best_sha256']
    ridge=read(RIDGE/'model_complete.json')
    if ridge.get('status')!='complete' or ridge['brain_source']['brainvae_checkpoint_sha256']!=ident['brainvae_sha256']:raise ValueError('Ridge model identity mismatch')
    for group in GROUPS:
        if len(plans[group])!=12:raise ValueError('Unexpected number of stimuli')
    identity={'protocol':'subj01_nsdgeneral_seen_stimuli_train_val_v1','subject':1,'methods':list(METHODS),'groups':list(GROUPS),
        'brainvae_sha256':ident['brainvae_sha256'],'w010_sha256':ident['w010_sha256'],'adapter_sha256':hashes,
        'data_fingerprint':ident['data_fingerprint'],'imagery_npz_sha256':ident['imagery_trials_sha256'],
        'split_sha256':{k:file_sha256(v) for k,v in SPLITS.items()},'original_natural_split_sha256':file_sha256(ORIGINAL_NAT),
        'ridge_model_metadata_sha256':file_sha256(RIDGE/'model_complete.json'),
        'code_sha256':file_sha256(Path(__file__)),'seed':'42+stimulus_index','samples':1,
        'note':'Original splits: imagery 10/2/4, natural vision 6/2, geometry vision 4/2/2. Vision splits are historical teacher roles and do not train these three adapters. All 12 stimuli seen during adapter training. Diagnostic only.'}
    return cfg,source,ident,plans,{k:np.stack(v) for k,v in values.items()},identity

def verify_json(path):
    return Path(path).is_file() and read(path).get('status')=='complete'

def prepare(device):
    cfg,source,ident,plans,values,identity=inputs();immutable(OUT/'identity.json',identity)
    with head_checkpoint_alias(source['protocol']['source_identity']):
        brain,mapper,_=pilot.load_frozen(cfg,source,device,sdpa=True)
    mapping=pilot.FrozenMapping(brain,mapper)
    adapters={}
    for method,run in RUNS.items():
        state=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        adapter=ImageryResidualAdapter(15724,'linear').to(device).eval().requires_grad_(False)
        adapter.load_state_dict(state['adapter'],strict=True);adapters[method]=adapter
    allmu=[]
    for group in GROUPS:
        rows=plans[group];x=torch.from_numpy(values[group]).to(device)
        results={m:[] for m in METHODS}; mus=[]
        with torch.inference_mode():
            for batch in x.split(1):
                mu=pilot.encode_mean(brain,batch);mus.append(mu.cpu())
                results['B0'].append(pilot.map_tokens(mapper,mu).float().cpu())
                for method,adapter in adapters.items():results[method].append(mapping(adapter(batch)).float().cpu())
        mu=torch.cat(mus);allmu.append(mu.numpy())
        for method in METHODS:
            tokens=torch.cat(results[method]);folder=OUT/method/'initial'/group;path=folder/'tokens/chunk00000.pt';path.parent.mkdir(parents=True,exist_ok=True)
            if tuple(tokens.shape)!=(12,256,1664) or not torch.isfinite(tokens).all():raise ValueError('Invalid tokens')
            payload={'mean':tokens,'records':rows,'representation':'mapped_raw_clip_tokens','method':method}
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=False)
                if old['records']!=rows or not torch.equal(old['mean'],tokens):raise ValueError('Existing token content differs')
            else:pilot.atomic_checkpoint(payload,path)
            meta={'protocol':identity['protocol'],'subject':1,'dataset':group,'method':method,'aggregation':'mean_fmri',
                  'n_items':12,'num_samples':1,'num_steps':38,'seed':42,'partial_test':False,'debug_training':False,
                  'checkpoint_sha256':identity['adapter_sha256'].get(method,ident['w010_sha256']),
                  'data_fingerprint':ident['data_fingerprint'],'brainvae_sha256':ident['brainvae_sha256'],
                  'w010_sha256':ident['w010_sha256'],'token_sha256':file_sha256(path),'identity_sha256':file_sha256(OUT/'identity.json')}
            immutable(folder/'metadata.json',meta)
    p=OUT/'inputs/all_mu.npy';p.parent.mkdir(parents=True,exist_ok=True)
    arr=np.concatenate(allmu)
    if p.exists():
        if not np.array_equal(np.load(p,mmap_mode='r'),arr):raise ValueError('Existing mu differs')
    else:
        tmp=p.with_suffix('.partial.npy');np.save(tmp,arr,allow_pickle=False);os.replace(tmp,p)
    atomic_json({'status':'complete','mu_sha256':file_sha256(p),'groups':list(GROUPS),'plans':plans},OUT/'inputs/complete.json')
    del mapping,brain,mapper,adapters;gc.collect();torch.cuda.empty_cache()

def prepared_ok():
    p=OUT/'inputs/complete.json';mu=OUT/'inputs/all_mu.npy'
    if not verify_json(p) or not mu.is_file() or file_sha256(mu)!=read(p)['mu_sha256']:return False
    for method in METHODS:
        for group in GROUPS:
            d=OUT/method/'initial'/group
            if not (d/'metadata.json').is_file() or not (d/'tokens/chunk00000.pt').is_file() or read(d/'metadata.json')['token_sha256']!=file_sha256(d/'tokens/chunk00000.pt'):return False
    return True

def ridge(device):
    if not prepared_ok():raise ValueError('Prepare required')
    import ridge_three_text as ridge_mod
    import predict_ridge_three_text as predictor
    folder=OUT/'ridge/predictions/train_val_all'
    if folder.exists() and not verify_json(folder/'prediction_complete.json'):
        backup=folder.with_name(folder.name+'.interrupted.'+str(time.time_ns()))
        folder.rename(backup)
        print('[preserve incomplete]',backup,flush=True)
    ridge_mod.FORMAL_ROOT=OUT/'ridge'
    sys.argv=['predict_ridge_three_text.py','--model-dir',str(RIDGE),'--mu-file',str(OUT/'inputs/all_mu.npy'),
              '--name','train_val_all','--device',str(device)]
    original=predictor.common.load_brain_context
    def load_context():
        context,paths=original()
        return reconcile_checkpoint_alias(context,read(RIDGE/'model_complete.json')['brain_source']),paths
    predictor.common.load_brain_context=load_context
    try: predictor.main()
    finally: predictor.common.load_brain_context=original

def ridge_ok():
    d=OUT/'ridge/predictions/train_val_all';p=d/'prediction_complete.json'
    if not verify_json(p):return False
    obj=read(p)
    if obj['input']['sha256']!=file_sha256(OUT/'inputs/all_mu.npy'):return False
    return all((d/('predicted_'+name+'.npy')).is_file() and file_sha256(d/('predicted_'+name+'.npy'))==sha for name,sha in obj['prediction_sha256'].items()) and set(obj['prediction_sha256'])==set(prev.SHAPES)

def targets(source,plans):
    _,_,paths=load_mapping_config(source['config_path'],ROOT); result={}
    for row in plans:
        path=paths.raw_nsd_root/'nsddata/experiments/nsdimagery/rawtargetimages'/('setB' if row['stimulus_type']=='complex' else 'setA')/(row['stimulus_id']+'.png')
        if not path.is_file():raise FileNotFoundError(path)
        result[row['stimulus_id']]=path
    return result

def generate(device):
    if not prepared_ok():raise ValueError('Prepare required')
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    cfg,source,ident=prev.source();_,_,paths=load_mapping_config(source['config_path'],ROOT)
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint']);unclip=unclip if unclip.is_absolute() else ROOT/unclip
    engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
    for method in METHODS:
        for group in GROUPS:
            d=OUT/method/'initial'/group;token=d/'tokens/chunk00000.pt';rows=torch.load(token,map_location='cpu',weights_only=False)['records']
            generate_samples_verified([token],d,read(d/'metadata.json'),engine,suffix,offset,partial(unclip_decode_batch,rgb_mode='standard'),save_image,1,42,targets(source,rows))
    del engine;gc.collect();torch.cuda.empty_cache()
    if not images_ok('initial'):raise ValueError('Incomplete initial images')
    atomic_json({'status':'complete'},OUT/'initial_complete.json')

def images_ok(phase):
    return all(sweep.checked_sample(OUT/m/phase/g,0) for m in METHODS for g in GROUPS)

def refine(device):
    if not images_ok('initial') or not ridge_ok():raise ValueError('Images and Ridge predictions required')
    if file_sha256(refine_base.ZAVY)!=refine_base.ZAVY_SHA:raise ValueError('Refinement checkpoint changed')
    ridge_dir=OUT/'ridge/predictions/train_val_all'
    arrays={name:np.load(ridge_dir/('predicted_'+name+'.npy'),mmap_mode='r') for name in prev.SHAPES}
    if any(a.shape[0]!=48 for a in arrays.values()):raise ValueError('Ridge prediction count mismatch')
    negcross,negpool,audit=refine_base.negative_condition(device);immutable(OUT/'negative_condition_audit.json',audit)
    engine,suffix=refine_base.load_engine(device,38,5.)
    for method in METHODS:
        for gi,group in enumerate(GROUPS):
            src=OUT/method/'initial'/group/'sample00';out=OUT/method/'refined'/group;sample=out/'sample00';(sample/'images').mkdir(parents=True,exist_ok=True)
            original=read(src/'manifest.json');items=[]
            for i,row in enumerate(original['items']):
                keep={k:row[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')}
                items.append({**keep,'item_index':i,'sample_index':0,'seed':42+i,'path':f'images/item{i:05d}.png'})
            meta={**read(OUT/method/'initial'/group/'metadata.json'),'generator_backend':'zavy_sdxl_img2img',
                  'text_condition':'nsd_ridge_three_branch_unadapted_same_fmri','text_oracle':False,'oracle':False,
                  'ridge_prediction_sha256':read(ridge_dir/'prediction_complete.json')['prediction_sha256'],
                  'source_manifest_sha256':file_sha256(src/'manifest.json'),'zavy_sha256':refine_base.ZAVY_SHA}
            immutable(out/'metadata.json',meta);immutable(sample/'manifest.json',{**meta,'level':'stimulus_level','sample_index':0,'n_images':12,'items':items})
            ledger=sample/'progress.jsonl';saved={}
            if ledger.exists():
                for line in ledger.read_text().splitlines():
                    entry=json.loads(line);i=int(entry['index'])
                    if not 0<=i<12 or entry['seed']!=42+i or entry['stimulus_id']!=items[i]['stimulus_id']:raise ValueError('Progress mismatch')
                    saved[i]=entry
            with ledger.open('a',encoding='utf-8') as f:
                for i,row in enumerate(original['items']):
                    path=sample/items[i]['path'];entry=saved.get(i)
                    if entry and path.is_file() and file_sha256(path)==entry['sha256']:continue
                    if path.exists():archive_file(path)
                    with Image.open(src/row['path']) as im:
                        x=pil_to_tensor(ImageOps.exif_transpose(im).convert('RGB')).float().div_(255).unsqueeze(0).to(device)
                    cond,uncond=refine_base.conditions(arrays,gi*12+i,suffix,negcross,negpool,device)
                    rgb=refine_base.img2img(engine,x,cond,uncond,42+i,38,13)
                    tmp=path.with_name(path.stem+'.tmp.png');save_image(rgb.cpu(),str(tmp));os.replace(tmp,path)
                    f.write(json.dumps({'index':i,'stimulus_id':row['stimulus_id'],'seed':42+i,'sha256':file_sha256(path)})+'\n');f.flush()
                    print('[refine]',method,group,i+1,flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()
    if not images_ok('refined'):raise ValueError('Incomplete refinement')
    atomic_json({'status':'complete'},OUT/'refinement_complete.json')

def evaluate(method,group,phase,device):
    sweep.experiment=lambda _k:OUT/method;sweep.SAMPLES=1
    sweep.evaluate(10,group,phase,device)

def evaluate_ok(method,group,phase):
    sweep.experiment=lambda _k:OUT/method;sweep.SAMPLES=1
    return sweep.evaluation_ok(10,group,phase)

def summarize():
    rows=[]
    for method in METHODS:
        for group in GROUPS:
            for phase in ('initial','refined'):
                if not evaluate_ok(method,group,phase):raise ValueError('Evaluation missing or changed')
                info=read(OUT/method/phase/group/'image_metrics.json')
                for label,prefix in (('all',''),('natural','complex_'),('geometry','simple_')):
                    rows.append({'method':method,'group':group,'phase':phase,'stimuli':label,
                                 **{k:float(info['metrics'][prefix+k]) for k in sweep.KEYS}})
    historical=ROOT/'output_1104/imagery_mu_clip_mse_v1/summary.json'
    old=read(historical)
    if old.get('status')!='complete':raise ValueError('Existing comparison summary is incomplete')
    aliases={'B0':'B0','w010_MSE_linear':'w010_mse','mu_MSE_linear':'mu_mse'}
    found=0
    for r in old['rows']:
        if r['method'] not in aliases or r['task'] not in ('vision','imagery') or r['phase'] not in ('initial','refined') or r['group'] not in ('all','natural','geometry'):continue
        group='imagery_test_existing' if r['task']=='imagery' else 'vision_all8_existing'
        rows.append({'method':aliases[r['method']],'group':group,'phase':r['phase'],'stimuli':r['group'],
                     **{k:float(r[k]) for k in sweep.KEYS}});found+=1
    if found!=36:raise ValueError('Existing comparison summary has unexpected rows')
    atomic_json({'status':'complete','rows':rows,'existing_summary':str(historical),
                 'existing_summary_sha256':file_sha256(historical),
                 'note':'Seen-stimulus diagnostic. Imagery test is the historical held-out 4 trials; historical vision used all 8 trials and overlaps train/val.'},OUT/'summary.json')
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','group','phase','stimuli',*sweep.KEYS]);writer.writeheader();writer.writerows(rows)

def stop(sig,frame):
    if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.send_signal(sig)
    raise SystemExit(128+sig)

def queue(device):
    global ACTIVE
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with mapping_lock(OUT):
        cfg,source,ident,plans,values,identity=inputs();immutable(OUT/'identity.json',identity)
        q=StageQueue(OUT,seconds=0);script=str(Path(__file__).resolve())
        def run(key,args,verify):
            def action():
                global ACTIVE
                log=OUT/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as f:
                    cmd=[sys.executable,'-u',script,'--stage',*args,'--device',str(device)];f.write('\n'+json.dumps(cmd)+'\n');f.flush()
                    ACTIVE=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);code=ACTIVE.wait();ACTIVE=None
                if code:raise RuntimeError(f'{key} failed ({code}); see {log}')
            q.run(key,action,verify,rest=False)
        run('prepare',['prepare'],prepared_ok)
        run('ridge',['ridge'],ridge_ok)
        run('generate',['generate'],lambda:verify_json(OUT/'initial_complete.json') and images_ok('initial'))
        run('refine',['refine'],lambda:verify_json(OUT/'refinement_complete.json') and images_ok('refined'))
        for method in METHODS:
            for group in GROUPS:
                for phase in ('initial','refined'):
                    run(f'eval_{method}_{group}_{phase}',['evaluate','--method',method,'--group',group,'--phase',phase],
                        lambda m=method,g=group,p=phase:evaluate_ok(m,g,p))
        summarize()
        q.state(status='complete',stage=None,error=None)
    print('[complete]',OUT,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('preflight','prepare','ridge','generate','refine','evaluate','queue'),default='queue')
    p.add_argument('--method',choices=METHODS);p.add_argument('--group',choices=GROUPS);p.add_argument('--phase',choices=('initial','refined'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='preflight':
        _,source,_,plans,_,identity=inputs()
        for group in GROUPS:targets(source,plans[group])
        for name in prev.SHAPES:
            if not (RIDGE/('dual_'+name+'.npy')).is_file():raise FileNotFoundError('Missing Ridge coefficients: '+name)
        if not refine_base.ZAVY.is_file():raise FileNotFoundError(refine_base.ZAVY)
        print(json.dumps({'status':'preflight_ok','counts':{k:[r['n_trials'] for r in v] for k,v in plans.items()},'identity':identity},ensure_ascii=False,indent=2));return
    device=torch.device(a.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if a.stage=='evaluate':
        if not all((a.method,a.group,a.phase)):p.error('evaluate requires method, group and phase')
        evaluate(a.method,a.group,a.phase,device)
    elif a.stage=='queue':queue(device)
    else:{'prepare':prepare,'ridge':ridge,'generate':generate,'refine':refine}[a.stage](device)

if __name__=='__main__':main()
