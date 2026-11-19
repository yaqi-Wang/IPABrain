#!/usr/bin/env python3
"""Controlled subj01 R-adapter sweep: K=5,10,20,50,100 NSD trials.

Normal invocation only runs when explicitly requested. --dry-run checks the
fixed grouped teacher split and pair schedules without GPU or output writes.
"""
from __future__ import annotations
import argparse
import copy
import csv
import gc
import json
import math
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
import yaml
from PIL import Image,ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
import run_imagery_w010_pilot as pilot
import run_k60_vision_imagery as prior
import refine_w010_mindeye2_img2img as base
from multisubject_generate import generate_samples_verified
from multisubject_resume import archive_file
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import load_mapping_config,mapping_lock
from src.utils.imagery_w010_pilot import balanced_schedule,pair_records
from src.utils.staged_queue import StageQueue,atomic_json

OUT=ROOT/'output_1104/k_sweep_subj01'
OLD=ROOT/'outputs_new/imagery_w010_subj01_natural_v1'
KS=(5,10,20,50,100)
TASKS=('vision','imagery')
KEYS=('PixCorr','SSIM','Alex2_2way','Alex5_2way','Incep_2way','CLIP_2way','Eff_corr_distance','SwAV_corr_distance')
EPOCHS=30
BATCHES=18
SAMPLES=1
REST=10


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path,value):
    path=Path(path)
    if path.exists() and read(path)!=value:raise ValueError('Existing identity differs: '+str(path))
    if not path.exists():atomic_json(value,path)


def make_plan(retrieval,split,k):
    if k not in KS:raise ValueError('Unexpected K')
    targets=split['targets'];queries=retrieval['queries']
    if len(targets)!=6:raise ValueError('Expected six training target stimuli')
    # Select a single image from each top-5, then use precisely its first two
    # trials as validation for every K. Its other repeats are excluded from
    # training, including when K grows, so validation cannot leak by image.
    val_ids={}
    for target in targets:
        first=queries[target][:5]
        if len(first)!=5 or len({x['image_id'] for x in first})!=2:
            raise ValueError('Top-5 no longer contains the expected two images')
        image=first[-1]['image_id']
        selected=[x for x in first if x['image_id']==image]
        if len(selected)!=2:raise ValueError('Expected two validation trials in top-5')
        val_ids[target]=[x['teacher_index'] for x in selected]
    validation_images={queries[target][4]['image_id'] for target in targets}
    if len(validation_images)!=6:raise ValueError('Validation image IDs overlap across targets; inspect grouping')
    groups=[copy.deepcopy(queries[target][:k]) for target in targets]
    if any(len(g)!=k for g in groups):raise ValueError('Insufficient ranked trials')
    old_ids=sorted({row['teacher_index'] for group in groups for row in group})
    remap={old:i for i,old in enumerate(old_ids)}
    teachers=[]
    for index in old_ids:
        teacher=copy.deepcopy(retrieval['teachers'][index])
        if teacher['kind']!='nsd_trial' or teacher['teacher_index']!=index:raise ValueError('Teacher provenance mismatch')
        teacher['original_teacher_index']=index;teacher['teacher_index']=remap[index];teachers.append(teacher)
    schedules={};grouped={};train_groups=[];validation_groups=[];counts=[]
    for label,(target,group) in enumerate(zip(targets,groups)):
        for row in group:
            old_id=row['teacher_index'];row['original_teacher_index']=old_id;row['teacher_index']=remap[old_id]
            row['teacher_split']=('val' if old_id in val_ids[target] else 'excluded_validation_image'
                                  if row['image_id'] in validation_images else 'train')
        train=[r['teacher_index'] for r in group if r['teacher_split']=='train']
        val=[r['teacher_index'] for r in group if r['teacher_split']=='val']
        if not train or len(val)!=2 or not {r['image_id'] for r in group if r['teacher_split']=='train'}.isdisjoint(validation_images):
            raise ValueError('Teacher image split leaks or has insufficient coverage')
        parents=list(map(int,split['rows'][target]['train']))
        if len(parents)!=10:raise ValueError('Expected ten original imagery training trials')
        rng_parent=np.random.default_rng(1000+label);parents=list(map(int,rng_parent.permutation(parents)))
        rng_teacher=np.random.default_rng(2000+label);teacher_order=list(map(int,rng_teacher.permutation(train)))
        rng_noise=np.random.default_rng(3000+label)
        pairs=[]
        for i in range(BATCHES*6):
            pairs.append({'parent_row':parents[i%len(parents)],'teacher_index':teacher_order[i%len(teacher_order)],
                          'augmented':i>=len(parents),'noise_seed':int(rng_noise.integers(1,2**31-1)),
                          'balance_repeat':i>=max(len(parents),len(train))})
        train_groups.append(pairs)
        validation_groups.append(pair_records(split['rows'][target]['val'],val,9000+label))
        grouped[target]=group
        counts.append({'target':target,'ranked_trials':len(group),'ranked_images':len({r['image_id'] for r in group}),
                       'train_trials':len(train),'validation_trials':len(val),
                       'excluded_trials':sum(r['teacher_split']=='excluded_validation_image' for r in group),
                       'train_images':len({r['image_id'] for r in group if r['teacher_split']=='train'}),
                       'validation_image':int(queries[target][4]['image_id'])})
    for portion,entries in (('train',train_groups),('val',validation_groups)):
        for attempt in range(512):
            try:
                batches=balanced_schedule(entries,6,42+attempt+(2000 if portion=='val' else 0));break
            except ValueError:continue
        else:raise ValueError('Cannot build collision-free paired schedule for '+portion)
        if portion=='train' and len(batches)!=BATCHES:raise ValueError('Unexpected number of optimizer batches')
        schedules[portion]={'protocol':'k_sweep_fixed_steps_v1','portion':portion,
                            'schedule_attempt':attempt,'batches':batches}
    plan={'overview':{'protocol':'fixed_val_image_trial_topk_v1','k_unit':'nsd_fmri_trial',
             'k_per_query':k,'ranked_candidates_include_validation':True,
             'validation_rule':'global held-out top5 second image; first two trials per target held for validation',
             'additional_repeats_of_heldout_images':'excluded from training and validation',
             'validation_images':sorted(map(int,validation_images)),
             'train_val_image_overlap':0,'target_query':'original target natural image; target-conditioned training',
             'counts':counts,'updates_per_epoch':BATCHES,'epochs':EPOCHS,'planned_updates':EPOCHS*BATCHES,
             'training_pairs_per_target_per_epoch':BATCHES*6,'sampling_note':'small K repeats teachers; first ten pairs cover all imagery training parents'},
          'queries':grouped,'teachers':teachers}
    return plan,schedules,old_ids


def plans():
    return read(OLD/'protocol/retrieval.json'),read(OLD/'protocol/split.json')


def make_cfg(k,directory):
    cfg=yaml.safe_load((ROOT/'configs/imagery_w010_pilot.yaml').read_text())
    cfg.update(knn_trials=k,experiment_root=str(directory.relative_to(ROOT)),max_epochs=EPOCHS,patience=EPOCHS)
    return cfg


def experiment(k):return OUT/f'K{k:03d}'


def prepare(k):
    retrieval,split=plans();plan,schedules,old_ids=make_plan(retrieval,split,k);directory=experiment(k)
    cfg=make_cfg(k,directory);source=pilot.run_source(cfg)
    original_cfg=yaml.safe_load((ROOT/'configs/imagery_w010_pilot.yaml').read_text())
    pilot.require_identity(original_cfg,OLD,source)
    if read(OLD/'protocol/code_lock.json')['files']!=pilot.code_fingerprint():raise ValueError('Original pilot code changed')
    protocol=directory/'protocol';protocol.mkdir(parents=True,exist_ok=True)
    plan['overview']['source_retrieval_sha256']=file_sha256(OLD/'protocol/retrieval.json')
    immutable(protocol/'split.json',split);immutable(protocol/'retrieval.json',plan)
    for part,value in schedules.items():immutable(protocol/f'R_{part}_pairs.json',value)
    benchmark=read(OLD/'protocol/gpu_benchmark.json')
    if benchmark['chosen_microbatch']!=18:raise ValueError('Original verified microbatch changed')
    immutable(protocol/'gpu_benchmark.json',benchmark);immutable(protocol/'config.json',cfg)
    immutable(protocol/'code_lock.json',{'files':pilot.code_fingerprint(),'sweep_script_sha256':file_sha256(Path(__file__))})
    source_state,cache=pilot.validated_teacher_cache(OLD)
    folder=directory/'teacher_cache';folder.mkdir(exist_ok=True)
    path=folder/'teacher_tokens.npy';state_path=folder/'state.json'
    spec={'protocol':'checked_subset_of_k1000_teacher_cache_v1','shape':[len(old_ids),256,1664],
          'storage_dtype':'float16','source_cache_sha256':source_state['cache_sha256'],
          'retrieval_sha256':file_sha256(protocol/'retrieval.json'),'original_teacher_indices':old_ids}
    if state_path.exists():
        state=read(state_path)
        if state['spec']!=spec or not state.get('complete') or file_sha256(path)!=state['cache_sha256']:
            raise ValueError('Existing teacher cache differs')
    else:
        if path.exists():raise ValueError('Unverified teacher cache exists; preserve and inspect')
        temp=path.with_name(path.name+'.partial')
        if temp.exists():
            archive_file(temp)
        array=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float16,shape=tuple(spec['shape']))
        for i,old_id in enumerate(old_ids):array[i]=cache[old_id]
        array.flush();del array;os.replace(temp,path)
        atomic_json({'spec':spec,'complete':True,'next_index':len(old_ids),'cache_sha256':file_sha256(path)},state_path)
    print('[prepare] '+json.dumps(plan['overview'],ensure_ascii=False),flush=True)
    return cfg,source


def train(k,device):
    cfg,source=prepare(k);pilot.train_adapter(cfg,experiment(k),source,device,'R')
    p=experiment(k)/'runs/R/training_complete.json';record=read(p)
    if record['global_step']!=EPOCHS*BATCHES or record['amp_skipped_updates']!=0:
        raise ValueError('Training did not produce exactly 540 successful updates')


def predict(k,device):
    directory=experiment(k);cfg=make_cfg(k,directory);source=pilot.run_source(cfg)
    identity=pilot.require_identity(yaml.safe_load((ROOT/'configs/imagery_w010_pilot.yaml').read_text()),OLD,source)
    complete=read(directory/'runs/R/training_complete.json');ckpt=directory/'runs/R/best.pt';digest=file_sha256(ckpt)
    if complete['best_sha256']!=digest or complete['global_step']!=EPOCHS*BATCHES or complete['amp_skipped_updates']:
        raise ValueError('R training is incomplete or differs')
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    if state['signature']!=complete['signature']:raise ValueError('R checkpoint signature changed')
    task_inputs=prior.task_values(source);brain,mapper,_=pilot.load_frozen(cfg,source,device,sdpa=False)
    mapping=pilot.FrozenMapping(brain,mapper);adapter=pilot.make_adapter(cfg,task_inputs['imagery'][1].shape[-1],device).eval().requires_grad_(False)
    adapter.load_state_dict(state['adapter'],strict=True)
    for task in TASKS:
        records,values=task_inputs[task]
        with torch.inference_mode():
            x=torch.from_numpy(values).to(device);out=torch.cat([mapping(adapter(chunk)) for chunk in x.split(2)]).float().cpu()
        if out.shape!=(12,256,1664) or not torch.isfinite(out).all():raise FloatingPointError('Invalid predicted tokens')
        folder=directory/'initial'/task;token=folder/'tokens/chunk00000.pt';token.parent.mkdir(parents=True,exist_ok=True)
        payload={'mean':out,'records':records,'representation':'mapped_raw_clip_tokens',
                 'mapping_checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':digest}
        if token.exists():
            existing=torch.load(token,map_location='cpu',weights_only=False)
            if existing['records']!=records or existing['adapter_sha256']!=digest or not torch.equal(existing['mean'],out):
                raise ValueError('Existing token predictions differ')
        else:pilot.atomic_checkpoint(payload,token)
        meta={'protocol':'subj01_k_sweep_r_initial_v1','subject':1,'dataset':task,'method':'R','K':k,
              'aggregation':'mean_fmri','n_items':12,'num_samples':SAMPLES,'num_steps':38,'seed':42,
              'checkpoint_sha256':digest,'w010_sha256':identity['w010_sha256'],'brainvae_sha256':identity['brainvae_sha256'],
              'data_fingerprint':identity['data_fingerprint'],'token_sha256':file_sha256(token),
              'partial_test':False,'debug_training':False,'target_usage':'evaluation only; natural training retrieval is target conditioned'}
        immutable(folder/'metadata.json',meta)
    atomic_json({'status':'complete','k':k,'checkpoint_sha256':digest,'tasks':list(TASKS)},directory/'prediction_complete.json')
    del mapping,brain,mapper,adapter;gc.collect();torch.cuda.empty_cache()


def generate(k,device):
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    cfg,source,_=prior.source();_,_,paths=load_mapping_config(source['config_path'],ROOT)
    unclip=Path(source['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute():unclip=ROOT/unclip
    engine,suffix,offset=load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
    for task in TASKS:
        folder=experiment(k)/'initial'/task;meta=read(folder/'metadata.json');token=folder/'tokens/chunk00000.pt'
        if file_sha256(token)!=meta['token_sha256']:raise ValueError('Predicted token hash mismatch')
        targets={row['stimulus_id']:Path(row['target_image']) for row in prior.original_rows(task,'B0')}
        generate_samples_verified([token],folder,meta,engine,suffix,offset,partial(unclip_decode_batch,rgb_mode='standard'),
                                  save_image,SAMPLES,42,targets)
    del engine;gc.collect();torch.cuda.empty_cache()
    if not initial_ok(k):raise ValueError('Initial generation did not verify')
    atomic_json({'status':'complete','k':k,'targets_per_task':12,'samples':SAMPLES},experiment(k)/'initial_complete.json')


def checked_sample(folder,sample):
    p=folder/f'sample{sample:02d}'/'manifest.json';ledger=p.parent/'progress.jsonl'
    if not p.is_file() or not ledger.is_file():return False
    m=read(p);items=m['items'];records={int(x['index']):x for x in map(json.loads,ledger.read_text().splitlines())}
    if len(items)!=12 or set(records)!=set(range(12)):return False
    for i,item in enumerate(items):
        rec=records[i];image=p.parent/item['path']
        if (item['seed']!=42+sample*100000+i or rec.get('seed')!=item['seed']
                or rec.get('stimulus_id')!=item['stimulus_id'] or not image.is_file()
                or file_sha256(image)!=rec.get('sha256')):return False
    return True


def initial_ok(k):
    for task in TASKS:
        folder=experiment(k)/'initial'/task;meta=folder/'metadata.json'
        if not meta.is_file() or read(meta).get('K')!=k or not (folder/'tokens/chunk00000.pt').is_file():return False
        if file_sha256(folder/'tokens/chunk00000.pt')!=read(meta)['token_sha256']:return False
        if not all(checked_sample(folder,s) for s in range(SAMPLES)):return False
    return True


def refine(k,device):
    if not initial_ok(k):raise ValueError('Initial images not complete')
    if file_sha256(base.ZAVY)!=base.ZAVY_SHA:raise ValueError('Zavy hash mismatch')
    arrays={task:prior.predictions(task) for task in TASKS};directory=experiment(k)
    negative_cross,negative_pool,audit=base.negative_condition(device);immutable(directory/'negative_condition_audit.json',audit)
    engine,suffix=base.load_engine(device,38,5.0)
    for task in TASKS:
        src=directory/'initial'/task;dest=directory/'refined'/task;dest.mkdir(parents=True,exist_ok=True)
        original_meta=read(src/'metadata.json')
        source_hashes={f'sample{s:02d}':{r['stimulus_id']:file_sha256(src/f'sample{s:02d}'/r['path'])
                        for r in read(src/f'sample{s:02d}/manifest.json')['items']} for s in range(SAMPLES)}
        pred_dir=prior.VISION_PRED if task=='vision' else prior.IMAGE_PRED
        provenance={'source_image_hashes':source_hashes,'ridge_prediction_sha256':{n:file_sha256(pred_dir/f'predicted_{n}.npy') for n in base.BRANCHES},
                    'zavy_sha256':base.ZAVY_SHA,'steps':38,'timepoint':13,'cfg_scale':5.0,
                    'seed_policy':'42 + sample*100000 + item_index','script_sha256':file_sha256(Path(__file__))}
        meta={**original_meta,'protocol':'subj01_k_sweep_r_zavy_refinement_v1',
              'generator_backend':'mindeye2_zavy_sdxl_img2img','text_condition':'ridge_predicted',
              'text_oracle':False,'oracle':False,'provenance':provenance}
        immutable(dest/'metadata.json',meta)
        for sample in range(SAMPLES):
            source=src/f'sample{sample:02d}';output=dest/f'sample{sample:02d}';(output/'images').mkdir(parents=True,exist_ok=True)
            original=read(source/'manifest.json')['items'];items=[]
            for i,row in enumerate(original):
                keep={key:row[key] for key in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')}
                items.append({**keep,'item_index':i,'sample_index':sample,'seed':42+sample*100000+i,'path':f'images/item{i:05d}.png'})
            manifest={**meta,'level':'stimulus_level','sample_index':sample,'n_images':12,'items':items}
            immutable(output/'manifest.json',manifest)
            ledger=output/'progress.jsonl';completed={}
            if ledger.exists():
                for line in ledger.read_text().splitlines():
                    record=json.loads(line);idx=int(record['index'])
                    if not 0<=idx<12 or record['seed']!=items[idx]['seed'] or record['stimulus_id']!=items[idx]['stimulus_id']:
                        raise ValueError('Refinement progress identity mismatch')
                    completed[idx]=record
            with ledger.open('a',encoding='utf-8') as handle:
                for i,row in enumerate(original):
                    path=output/items[i]['path'];old=completed.get(i)
                    if old and path.is_file() and file_sha256(path)==old['sha256']:continue
                    if path.exists():archive_file(path)
                    with Image.open(source/row['path']) as image:
                        x=pil_to_tensor(ImageOps.exif_transpose(image).convert('RGB')).float().div_(255).unsqueeze(0).to(device)
                    cond,uncond=base.conditions(arrays[task],i,suffix,negative_cross,negative_pool,device)
                    image=base.img2img(engine,x,cond,uncond,items[i]['seed'],38,13)
                    tmp=path.with_name(path.stem+'.tmp.png');save_image(image.cpu(),str(tmp));os.replace(tmp,path)
                    handle.write(json.dumps({'index':i,'stimulus_id':row['stimulus_id'],'seed':items[i]['seed'],'sha256':file_sha256(path)})+'\n');handle.flush()
                    print(f'[refine] K={k} {task} sample={sample} {i+1}/12',flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()
    if not refined_ok(k):raise ValueError('Refined images do not verify')
    atomic_json({'status':'complete','k':k,'samples':SAMPLES,'targets_per_task':12},directory/'refinement_complete.json')


def refined_ok(k):
    return all(checked_sample(experiment(k)/'refined'/task,s) for task in TASKS for s in range(SAMPLES))


def evaluate(k,task,stage,device):
    directory=experiment(k)/stage/task;metric=directory/'image_metrics.json';receipt=directory/'evaluation_complete.json'
    targets=read(directory/'sample00/manifest.json')['items']
    identity={'metadata_sha256':file_sha256(directory/'metadata.json'),
              'manifest_sha256':[file_sha256(directory/f'sample{s:02d}/manifest.json') for s in range(SAMPLES)],
              'progress_sha256':[file_sha256(directory/f'sample{s:02d}/progress.jsonl') for s in range(SAMPLES)],
              'target_sha256':{item['stimulus_id']:file_sha256(Path(item['target_image'])) for item in targets},
              'evaluator_sha256':file_sha256(ROOT/'scripts/eval_scheme1_reconstruction.py')}
    if receipt.exists():
        old=read(receipt)
        if old['identity']!=identity or file_sha256(metric)!=old['metrics_sha256']:raise ValueError('Existing evaluation changed')
        return
    if metric.exists():archive_file(metric)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_scheme1_reconstruction.py'),'--dataset-dir',str(directory),
         '--models','Alex2,Alex5,Incep,CLIP,Eff,SwAV','--device',str(device),'--batch-size','8']
    subprocess.run(cmd,cwd=ROOT,check=True)
    result=read(metric)
    if result['n_targets']!=12 or result['n_samples']!=SAMPLES or len(result['per_seed'])!=SAMPLES:
        raise ValueError('Image evaluation scope mismatch')
    for part in ('','complex_','simple_'):
        if not all(math.isfinite(seed[part+key]) for seed in result['per_seed'] for key in KEYS):raise ValueError('Invalid image score')
    atomic_json({'status':'complete','identity':identity,'metrics_sha256':file_sha256(metric)},receipt)


def evaluation_ok(k,task,stage):
    directory=experiment(k)/stage/task;receipt=directory/'evaluation_complete.json';metric=directory/'image_metrics.json'
    if not receipt.is_file() or not metric.is_file():return False
    info=read(receipt);identity=info['identity']
    return (info.get('status')=='complete' and file_sha256(metric)==info['metrics_sha256']
            and identity['metadata_sha256']==file_sha256(directory/'metadata.json')
            and identity['target_sha256']=={item['stimulus_id']:file_sha256(Path(item['target_image'])) for item in read(directory/'sample00/manifest.json')['items']}
            and identity['evaluator_sha256']==file_sha256(ROOT/'scripts/eval_scheme1_reconstruction.py')
            and all(identity['manifest_sha256'][s]==file_sha256(directory/f'sample{s:02d}/manifest.json')
                    and identity['progress_sha256'][s]==file_sha256(directory/f'sample{s:02d}/progress.jsonl')
                    and checked_sample(directory,s) for s in range(SAMPLES)))


def summarize():
    rows=[]
    for k in KS:
        completion=read(experiment(k)/'runs/R/training_complete.json')
        for task in TASKS:
            for stage in ('initial','refined'):
                if not evaluation_ok(k,task,stage):raise ValueError('Evaluation incomplete')
                metric=read(experiment(k)/stage/task/'image_metrics.json')
                for group,prefix in (('all',''),('natural','complex_'),('geometry','simple_')):
                    for sample,result in [(f'sample{i:02d}',v) for i,v in enumerate(metric['per_seed'])]:
                        rows.append({'K':k,'dataset':task,'stage':stage,'group':group,'sample':sample,
                                     'training_updates':completion['global_step'],'selected_epoch':completion['best_epoch'],
                                     **{key:float(result[prefix+key]) for key in KEYS}})
    old_summary=read(prior.OUT/'image_metrics_summary.json')
    if old_summary.get('status')!='complete':raise ValueError('Historical reference summary is incomplete')
    reference=[row for row in old_summary['rows'] if row['method'] in ('B0','V','R')]
    atomic_json({'status':'complete','K_values':list(KS),'samples':SAMPLES,'rows':rows,
                 'historical_reference_sample00':reference,
                 'reference_note':'Existing B0/V and historical K60 R are single-sample references; historical K60 training protocol differs from this fixed-update sweep.',
                 'note':'K counts ranked fMRI trials, including heldout teachers; R was separately trained for each K. '
                        'K=5 has one validation image/target. Geometry and vision are natural-adapter transfer. '
                        'Original target images condition retrieval during training; six stimuli per group.'},OUT/'summary.json')
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['K','dataset','stage','group','sample','training_updates','selected_epoch',*KEYS]);writer.writeheader();writer.writerows(rows)
    lines=['# subj01：R 的 K 值对照','','每个 K 使用相同留出教师、540 次更新；每个刺激只生成一次，表中为 sample00。',
           'K=5 的每目标验证教师仅 1 张图，几何与 vision 使用自然适配器迁移；原图参与训练教师检索。','',
           '| K | 任务 | 阶段 | 组别 | '+' | '.join(KEYS)+' |',
           '|---:|---|---|---|'+'---:|'*len(KEYS)]
    for r in rows:
        if r['sample']=='sample00':lines.append('| '+' | '.join([str(r['K']),r['dataset'],r['stage'],r['group'],*[f'{r[x]:.3f}' for x in KEYS]])+' |')
    lines+=['','每次采样及选中轮次见 summary.csv；原有 B0/V 单次结果见：',
            str((prior.OUT/'image_metrics_summary.md').resolve()),'']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf-8')


def train_ok(k):
    d=experiment(k)/'runs/R';p=d/'training_complete.json'
    if not p.is_file():return False
    r=read(p);signature=r['signature'];exp=experiment(k);cache=exp/'teacher_cache/teacher_tokens.npy'
    if (r.get('status')!='complete' or r.get('global_step')!=EPOCHS*BATCHES or r.get('amp_skipped_updates')!=0
            or not (d/'best.pt').is_file() or file_sha256(d/'best.pt')!=r['best_sha256']
            or not cache.is_file() or file_sha256(cache)!=signature['teacher_cache_sha256']
            or file_sha256(exp/'protocol/retrieval.json')!=signature['retrieval_sha256']
            or file_sha256(exp/'protocol/split.json')!=signature['split_sha256']
            or file_sha256(exp/'protocol/code_lock.json')!=signature['code_lock_sha256']):return False
    return all(file_sha256(exp/f'protocol/R_{part}_pairs.json')==digest
               for part,digest in signature['schedule_sha256'].items())


def predict_ok(k):
    if not (experiment(k)/'prediction_complete.json').is_file():return False
    return all((experiment(k)/'initial'/t/'metadata.json').is_file() and (experiment(k)/'initial'/t/'tokens/chunk00000.pt').is_file() and file_sha256(experiment(k)/'initial'/t/'tokens/chunk00000.pt')==read(experiment(k)/'initial'/t/'metadata.json')['token_sha256'] for t in TASKS)


def preflight():
    retrieval,split=plans();results=[]
    _,source,identity=prior.source()
    if file_sha256(source['prepared']/'imagery_trials.npz')!=identity['imagery_trials_sha256']:
        raise ValueError('Imagery trial data changed since the original experiment')
    prior.task_values(source)
    for task in TASKS:
        prior.predictions(task);prior.original_rows(task,'B0')
    for k in KS:
        plan,schedules,_=make_plan(retrieval,split,k)
        overview=plan['overview'];results.append({'K':k,'counts':overview['counts'],'train_batches':len(schedules['train']['batches']),
                                                 'validation_batches':len(schedules['val']['batches'])})
    if not all(r['train_batches']==BATCHES for r in results):raise ValueError('Optimizer update mismatch')
    print(json.dumps({'status':'preflight_ok','k_unit':'NSD fMRI trial','epochs':EPOCHS,'total_updates':EPOCHS*BATCHES,
                      'samples':SAMPLES,'fixed_validation_images':True,'results':results},indent=2,ensure_ascii=False),flush=True)


# BEGIN_PREDICT_CONFIG_RESUME_COMPATIBILITY
def pre_cooldown_source():
    """Recover and verify the exact pre-cooldown-change script for the old run."""
    import hashlib
    current=Path(__file__).read_text(encoding='utf-8')
    replacements=[
        ('REST=10\n','REST=120\n'),
        ('current='+'pre_cooldown_source()','current=Path(__file__).read_text(encoding=\'utf-8\')'),
        ("'samples':SAMPLES,"+"'cooldown_seconds':120,'dependencies':dependencies})",
         "'samples':SAMPLES,'cooldown_seconds':REST,'dependencies':dependencies})"),
        ("'repaired_script_sha256':"+"pre_cooldown_script_sha256(),'preserved_queue_identity':True,",
         "'repaired_script_sha256':file_sha256(Path(__file__)),'preserved_queue_identity':True,"),
        ("    q.ledger['rest_until']=min(q.ledger.get('rest_until',0),time.time()+REST)\n    atomic_json(q.ledger,q.path)\n",''),
    ]
    for new,old in replacements:
        if current.count(new)!=1:raise ValueError('Unexpected cooldown compatibility change')
        current=current.replace(new,old,1)
    # This helper is inside the compatibility block removed by resume_script_identity.
    return current


def pre_cooldown_script_sha256():
    import hashlib
    current=(pre_cooldown_source)()
    start=current.index('def pre_cooldown_source():\n')
    end=current.index('def resume_script_identity():\n',start)
    current=current[:start]+current[end:]
    digest=hashlib.sha256(current.encode('utf-8')).hexdigest()
    if digest!='7a33e1d79da3d4ae19672e932cbc4c34cc53be6fa0758317f5862e68b0ebe5c6':
        raise ValueError('Sweep changes exceed the verified cooldown update')
    return digest


def resume_script_identity():
    """Permit only the audited predict-config fix for the existing stopped run."""
    import hashlib
    current=pre_cooldown_source()
    path=OUT/'queue_identity.json'
    if not path.exists():return file_sha256(Path(__file__))
    recorded=read(path).get('dependencies',{}).get('script')
    legacy='547603a64ffcb9ad08e991697ff33154452fecce24be9591037793eee4f22152'
    if recorded!=legacy:return file_sha256(Path(__file__))
    snapshot=OUT/'sweep_before_predict_fix_547603a64ffc.py'
    if not snapshot.is_file() or file_sha256(snapshot)!=legacy:
        raise ValueError('Audited original sweep snapshot is missing or changed')
    start=current.index('# BEGIN_PREDICT_CONFIG_RESUME_COMPATIBILITY\n')
    end=current.index('# END_PREDICT_CONFIG_RESUME_COMPATIBILITY\n',start)+len('# END_PREDICT_CONFIG_RESUME_COMPATIBILITY\n')
    restored=current[:start]+current[end:]
    fixed="identity=pilot.require_identity(yaml.safe_load((ROOT/'configs/imagery_w010_pilot.yaml').read_text()),OLD,source)"
    old='identity=pilot.require_identity(cfg,OLD,source)'
    if restored.count(fixed)!=1 or restored.count("dependencies={'script':resume_script_identity(),")!=1:
        raise ValueError('Sweep changes exceed the verified predict-config fix')
    restored=restored.replace(fixed,old,1).replace("dependencies={'script':resume_script_identity(),",
                                                  "dependencies={'script':file_sha256(Path(__file__)),",1)
    restored=restored.replace('\n\n\n\n\ndef queue(device):','\n\n\ndef queue(device):',1)
    audit_block="""    if dependencies['script']!=file_sha256(Path(__file__)):
        immutable(OUT/'predict_config_fix_audit.json',{'original_script_sha256':dependencies['script'],
                  'repaired_script_sha256':file_sha256(Path(__file__)),'preserved_queue_identity':True,
                  'change':'prediction validates the original pilot configuration; training and sample protocol are unchanged'})
"""
    if restored.count(audit_block)!=1:raise ValueError('Sweep audit block differs')
    restored=restored.replace(audit_block,'',1)
    if hashlib.sha256(restored.encode('utf-8')).hexdigest()!=legacy:
        raise ValueError('Sweep changes exceed the verified predict-config fix')
    return legacy
# END_PREDICT_CONFIG_RESUME_COMPATIBILITY


def queue(device):
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies={'script':resume_script_identity(),'pilot':pilot.code_fingerprint(),
                  'source_retrieval':file_sha256(OLD/'protocol/retrieval.json'),'source_split':file_sha256(OLD/'protocol/split.json'),
                  'historical_reference':file_sha256(prior.OUT/'image_metrics_summary.json'),
                  'generation_helper':file_sha256(ROOT/'output_1104/scripts/multisubject_generate.py'),
                  'resume_helper':file_sha256(ROOT/'output_1104/scripts/multisubject_resume.py'),
                  'prior_script':file_sha256(ROOT/'output_1104/scripts/run_k60_vision_imagery.py'),
                  'refiner':file_sha256(ROOT/'output_1104/scripts/refine_w010_mindeye2_img2img.py')}
    immutable(OUT/'queue_identity.json',{'protocol':'subj01_retrieval_k_fixed_steps_v1','ks':list(KS),
               'steps_per_epoch':BATCHES,'epochs':EPOCHS,'samples':SAMPLES,'cooldown_seconds':120,'dependencies':dependencies})
    if dependencies['script']!=file_sha256(Path(__file__)):
        immutable(OUT/'predict_config_fix_audit.json',{'original_script_sha256':dependencies['script'],
                  'repaired_script_sha256':pre_cooldown_script_sha256(),'preserved_queue_identity':True,
                  'change':'prediction validates the original pilot configuration; training and sample protocol are unchanged'})
    q=StageQueue(OUT,REST);active={'process':None}
    q.ledger['rest_until']=min(q.ledger.get('rest_until',0),time.time()+REST)
    atomic_json(q.ledger,q.path)
    if q.ledger.get('inflight'):
        q.ledger['recovered_inflight']=q.ledger.pop('inflight');q.ledger['rest_until']=max(q.ledger.get('rest_until',0),time.time()+REST);atomic_json(q.ledger,q.path)
    def cooldown():
        q.ledger.pop('inflight',None);q.ledger['rest_until']=max(q.ledger.get('rest_until',0),time.time()+REST);atomic_json(q.ledger,q.path)
    def stop(signum,_frame):
        if active['process'] is not None:active['process'].send_signal(signum);active['process'].wait();cooldown()
        q.state(status='interrupted',signal=signum);raise SystemExit(128+signum)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def runstage(key,args,verify):
        def action():
            if verify():return
            if __import__('shutil').disk_usage(OUT).free<15*1024**3:raise RuntimeError('Less than 15 GiB free; no results removed')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),*map(str,args),'--device',str(device)]
            log=OUT/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
            q.ledger['inflight']={'stage':key,'started_epoch':time.time()};atomic_json(q.ledger,q.path)
            with log.open('a') as handle:
                handle.write('\nCOMMAND '+json.dumps(cmd)+'\n');handle.flush()
                active['process']=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                rc=active['process'].wait();active['process']=None
            cooldown()
            if rc:raise RuntimeError(f'{key} failed with exit {rc}; see {log}')
        q.run(key,action,verify)
    try:
        for k in KS:
            runstage(f'K{k:03d}_train',['--stage','train','--k',k],lambda n=k:train_ok(n))
            runstage(f'K{k:03d}_predict',['--stage','predict','--k',k],lambda n=k:predict_ok(n))
            runstage(f'K{k:03d}_generate',['--stage','generate','--k',k],lambda n=k:initial_ok(n) and (experiment(n)/'initial_complete.json').is_file())
            runstage(f'K{k:03d}_refine',['--stage','refine','--k',k],lambda n=k:refined_ok(n) and (experiment(n)/'refinement_complete.json').is_file())
            for task in TASKS:
                for stage in ('initial','refined'):
                    runstage(f'K{k:03d}_{task}_{stage}_eval',['--stage','evaluate','--k',k,'--task',task,'--image-stage',stage],
                             lambda n=k,t=task,s=stage:evaluation_ok(n,t,s))
        summarize();q.state(status='complete',stage=None,error=None)
    except BaseException as exc:
        if not isinstance(exc,SystemExit):q.state(status='failed',error=str(exc))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true',help='check all K groups without starting training')
    p.add_argument('--stage',choices=('train','predict','generate','refine','evaluate'))
    p.add_argument('--k',type=int,choices=KS);p.add_argument('--task',choices=TASKS);p.add_argument('--image-stage',choices=('initial','refined'))
    p.add_argument('--device',default='cuda:0');args=p.parse_args();torch.set_num_threads(4)
    if args.dry_run:preflight();return
    if not args.stage:
        preflight()
        with mapping_lock(OUT):queue(torch.device(args.device))
        return
    if args.k is None:p.error('--k is required for a stage')
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    if args.stage=='train':train(args.k,device)
    elif args.stage=='predict':predict(args.k,device)
    elif args.stage=='generate':generate(args.k,device)
    elif args.stage=='refine':refine(args.k,device)
    else:
        if not args.task or not args.image_stage:p.error('evaluation requires --task and --image-stage')
        evaluate(args.k,args.task,args.image_stage,device)


if __name__=='__main__':main()
