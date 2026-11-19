#!/usr/bin/env python3
"""K=60 R transfer, fMRI Ridge text, B0/R/V unCLIP and img2img evaluation."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'output_1104/scripts')]
import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image
import run_imagery_w010_pilot as pilot
import run_w010_vision_reference as vision
import train_sdxl_clipl_head as text_common
import refine_w010_mindeye2_img2img as base
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import atomic_json, frozen_brain, load_mapping_config

OUT = ROOT / 'output_1104/imagery_k60_ridge_subj01_v1'
PILOT = ROOT / 'outputs_new/imagery_w010_subj01_natural_v1'
TRANSFER = ROOT / 'outputs_new/imagery_w010_subj01_expansion_v1/transfer'
VISION = ROOT / 'outputs_new/w010_subj01_vision_reference_v1/evaluation/vision'
OLD_RUN = ROOT / 'output_1104/imagery_ridge_refine_subj01_v1'
RIDGE = ROOT / 'output_1104/nsd_subj01/ridge_three_text/predictions'
IMAGE_PRED = RIDGE / 'imagery_subj01_12stim_v1'
VISION_PRED = RIDGE / 'vision_subj01_k60_v1'
TEXT = ROOT / 'output_1104/imagery_caption_v1/sdxl_text/stimulus_user_short_git5_v1'
METHODS = ('B0', 'R', 'V')
TASKS = ('vision', 'imagery')
KEYS = ('PixCorr','SSIM','Alex2_2way','Alex5_2way','Incep_2way','CLIP_2way','Eff_corr_distance','SwAV_corr_distance')
SHAPES = {'clip_l': (77,768), 'bigg_tokens': (77,1280), 'bigg_pooled': (1280,)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Provenance differs; use a new experiment: ' + str(path))
    if not path.exists(): atomic_json(value, path)


def source():
    cfg = yaml.safe_load((ROOT / 'configs/imagery_w010_pilot.yaml').read_text())
    src = pilot.run_source(cfg)
    ident = pilot.require_identity(cfg, PILOT, src)
    if read(PILOT / 'protocol/code_lock.json')['files'] != pilot.code_fingerprint():
        raise ValueError('Original pilot code changed')
    return cfg, src, ident


def original_rows(task, method):
    if task == 'vision':
        folder = VISION if method == 'B0' else TRANSFER / 'vision_all12' / method / 'evaluation/vision'
        paths = [folder / 'sample00/manifest.json']
    else:
        paths = [PILOT / 'runs' / method / 'evaluation/imagery/sample00/manifest.json',
                 TRANSFER / 'imagery_geometry' / method / 'evaluation/imagery/sample00/manifest.json']
    rows = []
    for path in paths:
        manifest = read(path)
        if manifest['dataset'] != task or manifest['method'] != method:
            raise ValueError('Unexpected input manifest ' + str(path))
        for item in manifest['items']:
            image = (path.parent / item['path']).resolve()
            if not image.is_file(): raise FileNotFoundError(image)
            rows.append({**item, 'input_path': str(image), 'input_sha256': file_sha256(image)})
    if len(rows) != 12 or len({x['stimulus_id'] for x in rows}) != 12:
        raise ValueError('Expected exactly 12 different stimuli')
    return rows


def task_values(src):
    data = pilot.load_trials(src)
    split = read(PILOT / 'protocol/split.json')
    records, values = vision.vision_trials(data, split)
    outputs = {'vision': (records, values)}
    manifest = read(OLD_RUN / 'inputs/imagery_mu_manifest.json')
    img_records, img_values = [], []
    for row in manifest['rows']:
        ids = row['test_rows']
        if len(ids) != 4 or [int(data['global_trial_idx'][i]) for i in ids] != row['source_trial_ids']:
            raise ValueError('Imagery test trial mismatch')
        img_records.append({'subject_id':1, 'stimulus_id':row['stimulus_id'],
                            'stimulus_type':row['stimulus_type'], 'task_type':'imagery',
                            'n_trials':4, 'source_trial_ids':row['source_trial_ids'], 'runs':row['runs']})
        img_values.append(np.asarray(data['fmri_scaled'][ids], dtype=np.float32).mean(0))
    outputs['imagery'] = (img_records, np.stack(img_values))
    for task in TASKS:
        reference = original_rows(task,'B0')
        if [(r['stimulus_id'],r['source_trial_ids']) for r in outputs[task][0]] != [(r['stimulus_id'],r['source_trial_ids']) for r in reference]:
            raise ValueError('Source trial identities differ for ' + task)
    return outputs


def prepare_vision_mu(device):
    cfg, src, identity = source()
    records, values = task_values(src)['vision']
    out = OUT / 'inputs'; out.mkdir(parents=True, exist_ok=True)
    path, meta = out / 'vision_mu.npy', out / 'vision_mu_manifest.json'
    if meta.exists():
        info = read(meta)
        if info['mu_sha256'] != file_sha256(path) or info['records'] != records:
            raise ValueError('Existing vision mu changed')
        return
    if path.exists(): raise ValueError('Unverified vision mu exists')
    context, _ = text_common.load_brain_context()
    model = read(ROOT / 'output_1104/nsd_subj01/ridge_three_text/ridge_mu_three_text_v3/model/model_complete.json')
    if context['identity'] != model['brain_source'] or identity['brainvae_sha256'] != model['brain_source']['brainvae_checkpoint_sha256']:
        raise ValueError('Ridge and vision BrainVAE identities differ')
    brain = frozen_brain(context, device); del context['weights']
    parts = []
    with torch.inference_mode():
        for start in range(0,12,4):
            mu = brain.encode(torch.from_numpy(values[start:start+4]).to(device)).mode().float()
            if tuple(mu.shape[1:]) != (256,1664) or not torch.isfinite(mu).all(): raise ValueError('Invalid vision mu')
            parts.append(mu.cpu().numpy())
    tmp = path.with_suffix('.tmp.npy'); np.save(tmp, np.concatenate(parts)); tmp.replace(path)
    atomic_json({'status':'complete','dataset':'vision','records':records,'mu_sha256':file_sha256(path),
                 'brainvae_sha256':identity['brainvae_sha256'], 'data_fingerprint':identity['data_fingerprint'],
                 'note':'Mean of all 8 vision trials; V teacher trials overlap. No test caption input.'},meta)
    del brain; gc.collect(); torch.cuda.empty_cache()


def predictions(task):
    folder = VISION_PRED if task == 'vision' else IMAGE_PRED
    meta = read(folder / 'prediction_complete.json')
    mu = OUT / 'inputs/vision_mu.npy' if task == 'vision' else OLD_RUN / 'inputs/imagery_mu.npy'
    if meta['status'] != 'complete' or meta['test_text_used_as_input'] is not False or meta['input']['sha256'] != file_sha256(mu):
        raise ValueError('Ridge prediction provenance mismatch for ' + task)
    arrays = {}
    for key, shape in SHAPES.items():
        p = folder / ('predicted_' + key + '.npy'); x = np.load(p, mmap_mode='r')
        if x.shape != (12,*shape) or not np.isfinite(x).all() or file_sha256(p) != meta['prediction_sha256'][key]:
            raise ValueError('Invalid Ridge prediction: ' + key)
        arrays[key] = x
    return arrays


def predict_r(device):
    cfg, src, identity = source(); tasks = task_values(src)
    folder = OUT / 'r_training/runs/R'; complete = read(folder / 'training_complete.json')
    ckpt = folder / 'best.pt'; digest = file_sha256(ckpt)
    if complete['status'] != 'complete' or digest != complete['best_sha256'] or complete['signature']['w010_sha256'] != identity['w010_sha256']:
        raise ValueError('K60 training is incomplete or inconsistent')
    saved = torch.load(ckpt,map_location='cpu',weights_only=False)
    if saved['signature'] != complete['signature']: raise ValueError('K60 checkpoint signature differs')
    brain, mapper, _ = pilot.load_frozen(cfg,src,device,sdpa=False)
    mapping = pilot.FrozenMapping(brain,mapper)
    adapter = pilot.make_adapter(cfg,tasks['vision'][1].shape[-1],device).eval().requires_grad_(False)
    adapter.load_state_dict(saved['adapter'],strict=True)
    for task in TASKS:
        records, values = tasks[task]
        with torch.inference_mode():
            x = torch.from_numpy(values).to(device)
            pred = torch.cat([mapping(adapter(v)) for v in x.split(2)]).float().cpu()
        if tuple(pred.shape) != (12,256,1664) or not torch.isfinite(pred).all(): raise ValueError('Invalid R tokens')
        folder = OUT / 'initial/R' / task; token = folder / 'tokens/chunk00000.pt'; token.parent.mkdir(parents=True,exist_ok=True)
        payload = {'mean':pred,'records':records,'representation':'mapped_raw_clip_tokens',
                   'mapping_checkpoint_sha256':identity['w010_sha256'],'adapter_sha256':digest}
        if token.exists():
            old=torch.load(token,map_location='cpu',weights_only=False)
            if old['records'] != records or old['adapter_sha256'] != digest or not torch.equal(old['mean'],pred):
                raise ValueError('Existing R tokens differ')
        else: pilot.atomic_checkpoint(payload,token)
        immutable(folder / 'metadata.json',{'protocol':'k60_natural_R_transfer_v1','subject':1,'dataset':task,
                  'method':'R','aggregation':'mean_fmri','checkpoint_sha256':digest,'adapter_sha256':digest,
                  'w010_sha256':identity['w010_sha256'],'brainvae_sha256':identity['brainvae_sha256'],
                  'data_fingerprint':identity['data_fingerprint'],'num_samples':1,'num_steps':38,'seed':42,
                  'partial_test':False,'debug_training':False,'target_usage':'evaluation only after target-conditioned R training',
                  'n_items':12,'token_sha256':file_sha256(token),'transfer':'natural R applied to geometry and vision'})
    del adapter,mapping,brain,mapper; gc.collect(); torch.cuda.empty_cache()


def generate_r(device):
    from decode_brainvae_scheme1 import generate_samples
    from decode_synthesised_fmri_unclip import load_sdxl_unclip,unclip_decode_batch
    cfg,src,_ = source(); _,_,paths = load_mapping_config(src['config_path'],ROOT)
    unclip = Path(src['protocol']['config']['evaluation']['unclip_checkpoint'])
    if not unclip.is_absolute(): unclip = ROOT / unclip
    engine,suffix,offset = load_sdxl_unclip(paths.generative_models_path,unclip,paths.openclip_version,device,38)
    for task in TASKS:
        folder = OUT/'initial/R'/task; metadata=read(folder/'metadata.json')
        token=folder/'tokens/chunk00000.pt'
        if file_sha256(token) != metadata['token_sha256']: raise ValueError('R token checksum mismatch')
        target={row['stimulus_id']:Path(row['target_image']) for row in original_rows(task,'B0')}
        generate_samples([token],folder,metadata,engine,suffix,offset,
                         partial(unclip_decode_batch,rgb_mode='standard'),save_image,1,42,target,True)
    del engine; gc.collect(); torch.cuda.empty_cache()


def inputs(task, method):
    if method != 'R': rows = original_rows(task, method)
    else:
        folder=OUT/'initial/R'/task; manifest=folder/'sample00/manifest.json'
        rows=[]
        for item in read(manifest)['items']:
            image=(manifest.parent/item['path']).resolve()
            if not image.is_file(): raise FileNotFoundError(image)
            rows.append({**item,'input_path':str(image),'input_sha256':file_sha256(image)})
    reference=original_rows(task,'B0')
    if len(rows)!=12 or [(x['stimulus_id'],x['source_trial_ids']) for x in rows] != [(x['stimulus_id'],x['source_trial_ids']) for x in reference]:
        raise ValueError('Method and B0 inputs differ: '+task+'/'+method)
    return rows


def refine(device):
    if file_sha256(base.ZAVY) != base.ZAVY_SHA: raise ValueError('Zavy checkpoint mismatch')
    fingerprint=source()[2]['data_fingerprint']
    arrays={task:predictions(task) for task in TASKS}
    catalogs={(task,method):inputs(task,method) for task in TASKS for method in METHODS}
    negative_cross,negative_pool,audit=base.negative_condition(device)
    immutable(OUT/'negative_condition_audit.json',audit)
    engine,suffix=base.load_engine(device,38,5.0)
    for task in TASKS:
        for method in METHODS:
            rows=catalogs[task,method]; folder=OUT/'refined'/method/task; sample=folder/'sample00'
            (sample/'images').mkdir(parents=True,exist_ok=True)
            provenance={'ridge_prediction_sha256':{name:file_sha256((VISION_PRED if task=='vision' else IMAGE_PRED)/('predicted_'+name+'.npy')) for name in SHAPES},
                        'input_images':{r['stimulus_id']:r['input_sha256'] for r in rows},'zavy_sha256':base.ZAVY_SHA,
                        'script_sha256':file_sha256(Path(__file__))}
            meta={'protocol':'k60_w010_nsd_ridge_zavy_v1','subject':1,'dataset':task,'method':method,
                  'aggregation':'mean_fmri','n_items':12,'num_samples':1,'partial_test':False,'debug_training':False,
                  'checkpoint_sha256':base.digest_json(provenance),'data_fingerprint':fingerprint,
                  'generator_backend':'mindeye2_zavy_sdxl_img2img','text_condition':'nsd_ridge_from_'+task+'_mu',
                  'text_oracle':False,'oracle':False,'target_usage':'evaluation only after target-conditioned R training',
                  'vision_teacher_overlap':task=='vision' and method=='V','provenance':provenance}
            immutable(folder/'metadata.json',meta)
            items=[]
            for i,row in enumerate(rows):
                dest=sample/'images'/('item%05d.png'%i); receipt=sample/'images'/('item%05d.json'%i)
                if receipt.exists():
                    if not dest.is_file() or read(receipt)['sha256'] != file_sha256(dest): raise ValueError('Refined image changed')
                else:
                    if dest.exists(): raise ValueError('Unverified refined image exists')
                    with Image.open(row['input_path']) as img:
                        tensor=pil_to_tensor(ImageOps.exif_transpose(img).convert('RGB')).float().div_(255).unsqueeze(0).to(device)
                    cond,uncond=base.conditions(arrays[task],i,suffix,negative_cross,negative_pool,device)
                    rgb=base.img2img(engine,tensor,cond,uncond,42+i,38,13)
                    tmp=dest.with_name(dest.stem+'.tmp.png'); save_image(rgb.cpu(),str(tmp)); os.replace(str(tmp),str(dest))
                    atomic_json({'sha256':file_sha256(dest)},receipt)
                    print('[refine] {} {} {}/12 {}'.format(task,method,i+1,row['stimulus_id']),flush=True)
                items.append({**{k:row[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')},
                              'item_index':i,'sample_index':0,'seed':42+i,'path':'images/'+dest.name})
            immutable(sample/'manifest.json',dict(meta,level='stimulus_level',sample_index=0,n_images=12,items=items))
    del engine;gc.collect();torch.cuda.empty_cache()


def evaluate(device):
    rows=[]; fingerprint=source()[2]['data_fingerprint']
    for task in TASKS:
        for method in METHODS:
            baseline=OUT/'baseline'/method/task; sample=baseline/'sample00'; sample.mkdir(parents=True,exist_ok=True)
            initial=inputs(task,method)
            meta={'protocol':'k60_initial_matched_v1','subject':1,'dataset':task,'method':method,
                  'aggregation':'mean_fmri','n_items':12,'num_samples':1,'partial_test':False,'debug_training':False,
                  'checkpoint_sha256':base.digest_json({r['stimulus_id']:r['input_sha256'] for r in initial}),
                  'data_fingerprint':fingerprint,'target_usage':'evaluation only after target-conditioned R training',
                  'vision_teacher_overlap':task=='vision' and method=='V'}
            immutable(baseline/'metadata.json',meta)
            items=[{**{k:r[k] for k in ('subject_id','stimulus_id','stimulus_type','task_type','n_trials','source_trial_ids','target_image')},
                    'item_index':i,'sample_index':0,'path':r['input_path']} for i,r in enumerate(initial)]
            immutable(sample/'manifest.json',dict(meta,level='stimulus_level',sample_index=0,n_images=12,items=items))
            for stage,folder in [('initial',baseline),('refined',OUT/'refined'/method/task)]:
                receipt=folder/'evaluation_receipt.json'; metric=folder/'image_metrics.json'
                manifest=folder/'sample00/manifest.json'
                images=[(Path(r['path']) if Path(r['path']).is_absolute() else manifest.parent/r['path']) for r in read(manifest)['items']]
                identity={'manifest_sha256':file_sha256(manifest),'metadata_sha256':file_sha256(folder/'metadata.json'),
                          'image_sha256':[file_sha256(p) for p in images],
                          'targets_sha256':[file_sha256(Path(r['target_image'])) for r in read(manifest)['items']]}
                if receipt.exists():
                    if read(receipt)['identity'] != identity or read(receipt)['metrics_sha256'] != file_sha256(metric):
                        raise ValueError('Existing evaluation changed')
                else:
                    if metric.exists(): raise ValueError('Unverified metrics exist')
                    cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_scheme1_reconstruction.py'),'--dataset-dir',str(folder),
                         '--models','Alex2,Alex5,Incep,CLIP,Eff,SwAV','--device',str(device),'--batch-size','8']
                    subprocess.run(cmd,cwd=ROOT,check=True)
                    atomic_json({'identity':identity,'metrics_sha256':file_sha256(metric)},receipt)
                result=read(metric)
                if result['n_targets'] != 12 or result['n_samples'] != 1: raise ValueError('Metric scope differs')
                for group,prefix in [('all',''),('natural','complex_'),('geometry','simple_')]:
                    scores={k:float(result['metrics'][prefix+k]) for k in KEYS}
                    if not all(math.isfinite(v) for v in scores.values()): raise ValueError('Non-finite metrics')
                    rows.append({'dataset':task,'method':method,'stage':stage,'group':group,**scores})
    atomic_json({'status':'complete','rows':rows,'n_samples':1,
                 'note':'Natural R K60 transferred to geometry and vision; vision V includes teacher trials. R training uses original natural target image for retrieval.'},OUT/'image_metrics_summary.json')
    with (OUT/'image_metrics_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','method','stage','group',*KEYS]);writer.writeheader();writer.writerows(rows)
    lines=['# K60：vision 与 imagery 图像八指标','','每组单个被试、每刺激 sample00；几何 R/V 是自然适配器迁移；vision V 的视觉教师与评估试次重叠。','',
           '| 数据 | 方法 | 阶段 | 图像类型 | '+' | '.join(KEYS)+' |','| --- | --- | --- | --- | '+' | '.join(['---:']*len(KEYS))+' |']
    for row in rows: lines.append('| '+' | '.join([row['dataset'],row['method'],row['stage'],row['group'],*[f"{row[k]:.3f}" for k in KEYS]])+' |')
    (OUT/'image_metrics_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def compare_text():
    info=read(TEXT/'extraction_complete.json')
    if info['truncation_allowed'] or 'original test stimuli' not in info['input_role']:
        raise ValueError('Wrong oracle feature extraction')
    results={}
    for task in TASKS:
        pred=predictions(task); reference=original_rows(task,'B0'); results[task]={}
        for cap in ('qwen','git_beam5'):
            folder=TEXT/cap; manifests=[json.loads(s) for s in (folder/'captions.jsonl').read_text().splitlines()]
            if [r['stimulus_id'] for r in manifests] != [r['stimulus_id'] for r in reference]: raise ValueError('Reference order differs')
            results[task][cap]={}
            for key in SHAPES:
                p=folder/(key+'.npy'); rel=str(p.relative_to(TEXT))
                if file_sha256(p)!=info['output_files'][rel]['sha256']: raise ValueError('Text feature hash differs')
                y=np.asarray(np.load(p),dtype=np.float64).reshape(12,-1); x=np.asarray(pred[key],dtype=np.float64).reshape(12,-1)
                den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
                if np.any(den<=1e-12): raise ValueError('Zero text feature')
                results[task][cap][key]={g:{'mse':float(np.mean(((x-y)**2).mean(1)[ix])),
                                               'cosine':float(np.mean(((x*y).sum(1)/den)[ix]))}
                                         for g,ix in [('all',slice(None)),('natural',slice(0,6)),('geometry',slice(6,12))]}
    atomic_json({'status':'complete','reference_role':'user-edited or GIT original stimulus captions, oracle comparison only',
                 'reference_not_used_in_ridge_prediction_or_refinement':True,'scores':results},OUT/'text_prediction_vs_short_qwen_git.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=('vision_mu','predict_r','generate_r','refine','evaluate','compare_text','preflight'))
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);device=torch.device(a.device)
    if a.stage=='preflight':
        _,src,_=source(); task_values(src)
        for task in TASKS:
            for method in ('B0','V'): original_rows(task,method)
        print('[preflight] original B0/V and fMRI identities verified',flush=True);return
    if device.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    {'vision_mu':prepare_vision_mu,'predict_r':predict_r,'generate_r':generate_r,'refine':refine,'evaluate':evaluate}.get(a.stage,lambda _:compare_text())(device)


if __name__=='__main__':main()
