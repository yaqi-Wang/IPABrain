#!/usr/bin/env python3
"""Fail-closed subject-specific NSD then imagery experiment queue.

Each GPU stage rests 120 seconds after completion, persisted across restarts.
Partial Ridge fits are preserved and retried in a new numbered attempt.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import yaml
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import mapping_lock
from src.utils.staged_queue import StageQueue,atomic_json
import run_multisubject_imagery_natural as natural
import prepare_multisubject_b0 as b0
from src.utils.imagery_w010_pilot import split_trials
from multisubject_resume import archive_file

BASE=ROOT/'output_1104/multisubject_nsd_imagery_v1'
SCRIPT=ROOT/'output_1104/scripts'
COUNTS={1:1000,2:1000,3:614,4:515,5:1000,6:614,7:1000,8:515}
METHODS=('B0',)
KEYS=('PixCorr','SSIM','Alex2_2way','Alex5_2way','Incep_2way','CLIP_2way','Eff_corr_distance','SwAV_corr_distance')


def read(path):return json.loads(Path(path).read_text())


def matched(path,digest):return path.is_file() and file_sha256(path)==digest


def text_ok(root,subject):
    d=root/'coco_text_resumable';p=d/'extraction_complete.json'
    if not p.is_file():return False
    x=read(p)
    return (x.get('status')=='complete' and x.get('subject')==subject and not x.get('debug')
            and all(matched(d/name,item['sha256']) for name,item in x['files'].items()))


def ridge_ok(root,run,subject):
    d=root/'ridge'/run;m=d/'model/model_complete.json';p=d/'nsd_test/prediction_complete.json';s=d/'state.json'
    if not all(v.is_file() for v in (m,p,s)):return False
    meta,pred,state=read(m),read(p),read(s)
    if (state.get('status')!='complete' or meta.get('status')!='complete' or meta.get('debug')
            or meta.get('brain_source',{}).get('subject')!=subject or pred.get('images')!=COUNTS[subject]
            or pred.get('test_text_used_as_input') is not False):return False
    files={'input_mean.npy':'input_mean_sha256','target_stats.npz':'target_stats_sha256',
           'train_rows.npy':'train_rows_sha256','train_image_ids.npy':'train_ids_sha256'}
    return (all(matched(m.parent/name,meta[key]) for name,key in files.items())
            and all(matched(m.parent/('dual_'+k+'.npy'),v) for k,v in meta['dual_sha256'].items())
            and all(matched(p.parent/('predicted_'+k+'.npy'),v) for k,v in pred['predictions_sha256'].items()))


def generation_ok(folder,count):
    c=folder/'generation_complete.json'
    if not c.is_file():return False
    info=read(c)
    if info.get('status')!='complete' or info.get('images')!=count*3 or info.get('targets')!=count or info.get('samples')!=3:return False
    for sample in range(3):
        name=f'sample{sample:02d}';m=folder/name/'manifest.json';p=m.parent/'progress.jsonl'
        if not m.is_file() or not p.is_file():return False
        manifest=read(m);receipt=info['sample_receipts'][name]
        if (len(manifest.get('items',[]))!=count or not matched(m,receipt['manifest_sha256'])
                or not matched(p,receipt['progress_sha256'])):return False
        progress={int(x['index']):x for x in map(json.loads,p.read_text().splitlines())}
        if set(progress)!=set(range(count)):return False
        for i,item in enumerate(manifest['items']):
            rec=progress[i]
            if rec.get('seed')!=item.get('seed') or not matched(m.parent/item['path'],rec['image_sha256']):return False
    return True


def image_evaluation_ok(folder,count):
    p=folder/'evaluation_complete.json';m=folder/'image_metrics.json';g=folder/'generation_complete.json'
    manifests=[folder/f'sample{i:02d}'/'manifest.json' for i in range(3)]
    if not all(x.is_file() for x in (p,m,g,*manifests)):return False
    receipt,metrics=read(p),read(m)
    return (receipt.get('status')=='complete' and metrics.get('n_targets')==count and metrics.get('n_samples')==3
            and len(metrics.get('per_seed',[]))==3 and len(receipt.get('manifest_sha256',[]))==3
            and all(math.isfinite(metrics['metrics'][k]) for k in KEYS)
            and all(all(math.isfinite(row[k]) for k in KEYS) for row in metrics['per_seed'])
            and matched(m,receipt['metrics_sha256']) and all(matched(path,digest) for path,digest in zip(manifests,receipt['manifest_sha256']))
            and matched(g,receipt['generation_complete_sha256']))


def b0_initial_ok(root,stage):
    d=root/'imagery_b0_initial_3samples_v1'
    marker=d/('prediction_complete.json' if stage=='predict' else 'generation_complete.json')
    if not marker.is_file() or read(marker).get('status')!='complete':return False
    for method in METHODS:
        folder=d/'initial'/method/'imagery';meta=folder/'metadata.json'
        if not meta.is_file() or read(meta).get('num_samples')!=3:return False
        if stage=='predict':
            if not matched(folder/'tokens/chunk00000.pt',read(meta)['token_sha256']):return False
        if stage=='generate':
            for sample in range(3):
                manifest=folder/f'sample{sample:02d}'/'manifest.json'
                if not manifest.is_file():return False
                items=read(manifest).get('items',[])
                if len(items)!=12 or any(not (manifest.parent/item['path']).is_file() for item in items):return False
                if stage=='generate' and any(not matched(manifest.parent/item['path'],read(marker)['image_sha256'][method][sample][item['stimulus_id']]) for item in items):return False
    return True


def imagery_mu_ok(root,subject):
    d=root/'imagery_ridge_inputs_3samples_v1';p=d/'imagery_mu_manifest.json';f=d/'imagery_mu.npy'
    return p.is_file() and read(p).get('status')=='complete' and read(p).get('subject')==subject and read(p).get('shape')==[12,256,1664] and matched(f,read(p)['mu_sha256'])


def imagery_pred_ok(root):
    d=root/'ridge/predictions/imagery_12stim_3samples_v1';p=d/'prediction_complete.json'
    return p.is_file() and read(p).get('status')=='complete' and all(matched(d/('predicted_'+k+'.npy'),v) for k,v in read(p)['prediction_sha256'].items())


def imagery_refine_ok(root):
    d=root/'imagery_b0_refine_3samples_v1/refinement';p=d/'generation_complete.json'
    return p.is_file() and read(p).get('status')=='complete' and read(p).get('images')==36 and read(p).get('methods')==['B0'] and all(generation_ok(d/m/'imagery',12) for m in METHODS)


def imagery_eval_ok(root,subject):
    d=root/'imagery_b0_refine_3samples_v1';p=d/'image_metrics_summary.json'
    if not p.is_file():return False
    x=read(p)
    if x.get('status')!='complete' or x.get('subject')!=subject or len(x.get('rows',[]))!=2 or {r.get('method') for r in x['rows']}!={'B0'}:return False
    for name in ('baseline_sample00','refinement'):
        for method in METHODS:
            folder=d/name/method/'imagery';receipt=folder/'evaluation_complete.json'
            if not receipt.is_file():return False
            r=read(receipt);ident=r['identity']
            if (not matched(folder/'image_metrics.json',r['metrics_sha256'])
                    or not matched(folder/'metadata.json',ident['metadata_sha256'])
                    or len(ident['manifest_sha256'])!=3
                    or any(not matched(folder/f'sample{i:02d}/manifest.json',digest)
                           for i,digest in enumerate(ident['manifest_sha256']))):return False
    return True


def preflight(scope='all'):
    """Validate every subject's selected sources and 12-stimulus split before work starts."""
    for subject,count in COUNTS.items():
        root=BASE/f'subj{subject:02d}'
        selected=read(ROOT/f'outputs_new/w010_multisubject_v1/subj{subject:02d}/selection.json')['selected']
        checkpoint,config=Path(selected['checkpoint']),Path(selected['config'])
        if not matched(checkpoint,selected['checkpoint_sha256']) or not matched(config,selected['config_sha256']):
            raise ValueError(f'subj{subject:02d}: selected model/config hash mismatch')
        protocol=read(checkpoint.parent.parent/'protocol.json')
        if protocol.get('subject')!=subject or protocol.get('debug_training'):
            raise ValueError(f'subj{subject:02d}: wrong training protocol')
        prepared=Path(protocol['prepared_data'])
        with np.load(prepared/'split.npz',allow_pickle=False) as z:
            test_ids=np.asarray(z['test_image_ids']);source_ids=np.asarray(z['source_image_ids'])
            train=np.asarray(z['train_indices']);val=np.asarray(z['val_indices'])
        if len(test_ids)!=count or len(np.unique(test_ids))!=count or np.intersect1d(source_ids,test_ids).size or np.intersect1d(train,val).size:
            raise ValueError(f'subj{subject:02d}: invalid NSD image split')
        baseline=checkpoint.parent.parent/'evaluation/w010_selected_nsd_test_all_s42/nsd_test'
        meta=read(baseline/'metadata.json');manifest=read(baseline/'sample00/manifest.json')
        if (meta.get('subject')!=subject or meta.get('n_items')!=count or meta.get('partial_test')
                or meta.get('checkpoint_sha256')!=selected['checkpoint_sha256']
                or [x['image_id'] for x in manifest['items']]!=test_ids.tolist()):
            raise ValueError(f'subj{subject:02d}: NSD reconstruction identity/order mismatch')
        if subject==1:
            if scope=='all':b0.audit_existing_subj01()
            print('[preflight] subj01: existing NSD and imagery results require reuse checks',flush=True)
            continue
        if scope=='nsd':
            print(f'[preflight] subj{subject:02d}: NSD test={count}',flush=True)
            continue
        cfg=yaml.safe_load((root/'config_imagery_natural_k60_3samples.yaml').read_text())
        if (cfg['subject']!=subject or cfg['knn_trials']!=60 or cfg['num_samples']!=3
                or cfg['w010_checkpoint']!=selected['checkpoint'] or cfg['w010_config']!=selected['config']):
            raise ValueError(f'subj{subject:02d}: imagery configuration mismatch')
        with np.load(prepared/'imagery_trials.npz',allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        natural_split=split_trials(data,cfg['split_seed'],'complex')
        geometry_split=split_trials(data,cfg['split_seed'],'simple')
        if natural_split['subject']!=subject or geometry_split['subject']!=subject:
            raise ValueError(f'subj{subject:02d}: imagery trials belong to a different subject')
        print(f'[preflight] subj{subject:02d}: NSD test={count}, imagery=6 natural + 6 geometry',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('nsd','all'),default='all')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dry-run',action='store_true',help='validate all subject inputs without starting GPU tasks')
    args=parser.parse_args()
    preflight(args.scope)
    if args.dry_run:
        print('[dry-run] no queue or GPU work started',flush=True)
        return
    qdir=BASE/'queue_multisubject_nsd_imagery_b0_v2'
    scripts=['extract_multisubject_nsd_text_resumable.py','ridge_multisubject_three_text.py','refine_multisubject_nsd.py',
             'run_multisubject_imagery_natural.py','run_multisubject_imagery_geometry.py','prepare_multisubject_imagery_mu.py',
             'predict_multisubject_ridge.py','refine_multisubject_imagery.py','evaluate_multisubject_imagery.py',
             'prepare_subj01_imagery_three_samples.py','prepare_multisubject_b0.py']
    dependencies=['extract_sdxl_clip_l11.py','extract_sdxl_bigg_text.py','extract_imagery_stimulus_sdxl_text.py',
                  'train_sdxl_clipl_head.py','train_sdxl_three_text.py','refine_w010_mindeye2_img2img.py',
                  'run_k60_vision_imagery.py','prepare_imagery_r_k60.py']
    identity={'protocol':'subject_specific_nsd_then_imagery_b0_only_v2','subjects':list(COUNTS),'test_counts':COUNTS,'imagery_methods':['B0'],
              'rest_seconds':120,'minimum_free_bytes':12*1024**3,
              'queue_sha256':file_sha256(Path(__file__)),
              'scripts':{name:file_sha256(SCRIPT/name) for name in scripts},
              'dependencies':{name:file_sha256(SCRIPT/name) for name in dependencies},
              'resume_helper_sha256':file_sha256(SCRIPT/'multisubject_resume.py'),
              'generation_helper_sha256':file_sha256(SCRIPT/'multisubject_generate.py'),
              'natural_dependencies':natural.code_fingerprint(),
              'subject_configs':{str(s):file_sha256(BASE/f'subj{s:02d}/config_imagery_natural_k60_3samples.yaml') for s in COUNTS if s!=1},
              'subject_selections':{str(s):file_sha256(ROOT/f'outputs_new/w010_multisubject_v1/subj{s:02d}/selection.json') for s in COUNTS}}
    with mapping_lock(BASE/'queue_multisubject_nsd_imagery_v1'), mapping_lock(qdir):
        identity_path=qdir/'queue_identity.json'
        if identity_path.exists() and read(identity_path)!=json.loads(json.dumps(identity)):
            raise ValueError('Queue code/protocol changed; inspect before restarting')
        if not identity_path.exists():atomic_json(identity,identity_path)
        q=StageQueue(qdir,120)
        active={'process':None}
        legacy_ledger=BASE/'queue_multisubject_nsd_imagery_v1/stage_ledger.json'
        if not q.path.exists() and legacy_ledger.is_file() and read(legacy_ledger).get('inflight'):
            q.ledger['legacy_inflight']=read(legacy_ledger)['inflight']
            q.ledger['rest_until']=max(q.ledger.get('rest_until',0),time.time()+120)
            atomic_json(q.ledger,q.path)
        if q.ledger.get('inflight'):
            # The previous process may have died without recording its finish.
            q.ledger['recovered_inflight']=q.ledger.pop('inflight')
            q.ledger['rest_until']=max(q.ledger.get('rest_until',0),time.time()+120)
            atomic_json(q.ledger,q.path)
        def finish_stage_cooldown():
            q.ledger.pop('inflight',None)
            q.ledger['rest_until']=max(q.ledger.get('rest_until',0),time.time()+120)
            atomic_json(q.ledger,q.path)
        def stop(signum,_frame):
            if active['process'] is not None:
                active['process'].send_signal(signum);active['process'].wait()
                finish_stage_cooldown()
            q.state(status='interrupted',signal=signum)
            raise SystemExit(128+signum)
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        def command(key,arguments):
            # Reserve the expected next-stage footprint in addition to the
            # minimum; never reclaim or delete existing experiments implicitly.
            extra_gib = (7 if key.endswith('_ridge') and 'imagery' not in key else
                         4 if key.endswith('_text') else
                         3 if key.endswith('_nsd_refine') or 'cache_teachers' in key else 1)
            if shutil.disk_usage(BASE).free < (12+extra_gib)*1024**3:
                raise RuntimeError(f'Insufficient space for {key}: need {extra_gib} GiB plus 12 GiB reserve; no existing outputs were deleted')
            logfile=BASE/'logs'/(key+'.log');logfile.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u',*map(str,arguments)]
            print('[command] '+json.dumps(cmd),flush=True)
            q.ledger['inflight']={'stage':key,'started_epoch':time.time()}
            atomic_json(q.ledger,q.path)
            with logfile.open('a') as f:
                f.write('\nCOMMAND '+json.dumps(cmd)+'\n');f.flush()
                active['process']=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                result=active['process'].wait();active['process']=None
            finish_stage_cooldown()
            if result:
                raise RuntimeError(f'{key} exited {result}; see {logfile}')
        def stage(key,args,verify):
            q.run(key,lambda:None if verify() else command(key,args),verify)
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        try:
            chosen={}
            for subject in COUNTS:
                root=BASE/f'subj{subject:02d}';prefix=f'subj{subject:02d}'
                if subject==1:
                    name='ridge_mu_three_text_v3'
                else:
                    stage(prefix+'_text',[SCRIPT/scripts[0],'--subject',subject,'--device',args.device,'--batch-size',32,'--resume'],lambda r=root,s=subject:text_ok(r,s))
                    pointer=root/'ridge_selected_run.json'
                    name=read(pointer)['run_name'] if pointer.exists() else 'ridge_mu_three_text_v1'
                    if not ridge_ok(root,name,subject) and (root/'ridge'/name).exists():
                        index=2
                        while (root/'ridge'/f'ridge_mu_three_text_v{index}').exists():index+=1
                        name=f'ridge_mu_three_text_v{index}'
                    atomic_json({'subject':subject,'run_name':name},pointer)
                    stage(prefix+'_ridge',[SCRIPT/scripts[1],'--subject',subject,'--device',args.device,'--run-name',name],lambda r=root,n=name,s=subject:ridge_ok(r,n,s))
                chosen[subject]=name
                out=root/'runs/w010_ridge_nsd250_3samples_v1/nsd_test'
                stage(prefix+'_nsd_refine',[SCRIPT/scripts[2],'--subject',subject,'--ridge-run',name,'--device',args.device],lambda d=out:generation_ok(d,250))
                def evaluate_nsd(d=out,n=250,key=prefix+'_nsd_eval'):
                    if (d/'evaluation_complete.json').exists():
                        raise ValueError('Existing NSD evaluation receipt does not verify; preserve and inspect it')
                    if (d/'image_metrics.json').exists():
                        archived=archive_file(d/'image_metrics.json')
                        print('[resume] preserved unverified NSD metrics at '+str(archived),flush=True)
                    command(key,[ROOT/'scripts/eval_scheme1_reconstruction.py','--dataset-dir',d,'--models','Alex2,Alex5,Incep,CLIP,Eff,SwAV','--device',args.device,'--batch-size',8])
                    m=read(d/'image_metrics.json')
                    if m.get('n_targets')!=n or m.get('n_samples')!=3 or len(m.get('per_seed',[]))!=3 or not all(math.isfinite(row[k]) for row in m['per_seed'] for k in KEYS):raise ValueError('Invalid image metrics')
                    atomic_json({'status':'complete','metrics_sha256':file_sha256(d/'image_metrics.json'),
                                 'manifest_sha256':[file_sha256(d/f'sample{i:02d}/manifest.json') for i in range(3)],
                                 'generation_complete_sha256':file_sha256(d/'generation_complete.json')},d/'evaluation_complete.json')
                q.run(prefix+'_nsd_eval',lambda:None if image_evaluation_ok(out,250) else evaluate_nsd(),lambda d=out:image_evaluation_ok(d,250))
            if args.scope=='all':
                for subject in COUNTS:
                    root=BASE/f'subj{subject:02d}';prefix=f'subj{subject:02d}';name=chosen[subject]
                    if subject==1:
                        stage(prefix+'_imagery_b0_reuse',[SCRIPT/scripts[10],'--subject',subject,'--stage','reuse','--device',args.device],lambda r=root:b0_initial_ok(r,'generate'))
                        stage(prefix+'_imagery_b0_refine',[SCRIPT/scripts[7],'--subject',subject,'--b0-only','--device',args.device],lambda r=root:imagery_refine_ok(r))
                        stage(prefix+'_imagery_b0_eval',[SCRIPT/scripts[8],'--subject',subject,'--b0-only','--device',args.device],lambda r=root,s=subject:imagery_eval_ok(r,s))
                        continue
                    config=root/'config_imagery_natural_k60_3samples.yaml';exp=root/'imagery_natural_k60_3samples_v1'
                    stage(prefix+'_natural_prepare',[SCRIPT/scripts[3],'--config',config,'--stage','prepare','--device',args.device],
                          lambda e=exp:(e/'protocol/identity.json').is_file() and (e/'protocol/split.json').is_file())
                    stage(prefix+'_imagery_mu',[SCRIPT/scripts[5],'--subject',subject,'--ridge-run',name,'--device',args.device,'--resume'],lambda r=root,s=subject:imagery_mu_ok(r,s))
                    stage(prefix+'_imagery_ridge',[SCRIPT/scripts[6],'--subject',subject,'--model-dir',root/'ridge'/name/'model','--mu-file',root/'imagery_ridge_inputs_3samples_v1/imagery_mu.npy','--name','imagery_12stim_3samples_v1','--device',args.device,'--resume'],lambda r=root:imagery_pred_ok(r))
                    for phase in ('predict','generate'):
                        stage(prefix+'_imagery_b0_'+phase,[SCRIPT/scripts[10],'--subject',subject,'--stage',phase,'--device',args.device],lambda r=root,p=phase:b0_initial_ok(r,p))
                    stage(prefix+'_imagery_b0_refine',[SCRIPT/scripts[7],'--subject',subject,'--b0-only','--device',args.device],lambda r=root:imagery_refine_ok(r))
                    stage(prefix+'_imagery_b0_eval',[SCRIPT/scripts[8],'--subject',subject,'--b0-only','--device',args.device],lambda r=root,s=subject:imagery_eval_ok(r,s))
            q.state(status='complete',stage=None,scope=args.scope,error=None)
        except BaseException as e:
            if not isinstance(e,SystemExit):q.state(status='failed',error=str(e))
            raise


if __name__=='__main__':main()
