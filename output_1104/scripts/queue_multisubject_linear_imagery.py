#!/usr/bin/env python3
"""Eight subjects, two MSE linear adapters, vision/imagery, three samples."""
from __future__ import annotations
import argparse,csv,json,math,os,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import torch
import run_multisubject_linear_imagery as work
import prepare_multisubject_vision_ridge_inputs as vision
import run_k60_vision_imagery as previous
import sweep_subj01_retrieval_k as sweep
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import mapping_lock
from src.utils.staged_queue import StageQueue,atomic_json

OUT=work.OUT;SCRIPT=ROOT/'output_1104/scripts';SUBJECTS=tuple(range(1,9));METHODS=work.METHODS;TASKS=work.TASKS
ACTIVE=None;LOCKED=False

def read(p):return json.loads(Path(p).read_text())
def matches(path,digest):
    path=Path(path)
    return path.is_file() and file_sha256(path)==digest

def immutable(p,v):
    p=Path(p)
    if p.exists() and read(p)!=v:raise ValueError(f'Existing queue identity changed: {p}')
    if not p.exists():atomic_json(v,p)

def model_dir(s):
    root=work.OLD/f'subj{s:02d}';name=read(root/'ridge_selected_run.json')['run_name'];return root/'ridge'/name/'model'

def text_paths(s,task):
    if s==1:
        return ((previous.VISION_PRED,previous.OUT/'inputs/vision_mu_manifest.json') if task=='vision' else
                (previous.IMAGE_PRED,previous.OLD_RUN/'inputs/imagery_mu_manifest.json'))
    root=work.OLD/f'subj{s:02d}'
    return ((root/'ridge/predictions/vision_multisubject_linear_v1',work.subject_root(s)/'vision_ridge_inputs/imagery_mu_manifest.json') if task=='vision'
           else (root/'ridge/predictions/imagery_12stim_3samples_v1',root/'imagery_ridge_inputs_3samples_v1/imagery_mu_manifest.json'))

def text_ok(s,task):
    pred,manifest=text_paths(s,task);p=pred/'prediction_complete.json'
    if not p.is_file() or not manifest.is_file():return False
    a=read(p);b=read(manifest)
    if a.get('status')!='complete' or a.get('test_text_used_as_input') is not False or a['input']['sha256']!=b['mu_sha256'] or not matches(a['input']['path'],b['mu_sha256']):return False
    if a['input'].get('manifest_sha256') and a['input']['manifest_sha256']!=file_sha256(manifest):return False
    if s!=1 and (b.get('subject')!=s or b.get('dataset')!=task or a.get('model_metadata_sha256')!=file_sha256(model_dir(s)/'model_complete.json')):return False
    for key,shape in previous.SHAPES.items():
        import numpy as np
        path=pred/('predicted_'+key+'.npy')
        if not matches(path,a['prediction_sha256'][key]) or np.load(path,mmap_mode='r').shape!=(12,*shape):return False
    return True

def preflight(subjects):
    import shutil
    print('[preflight] disk_free_gib={:.2f}'.format(shutil.disk_usage(ROOT).free/2**30),flush=True)
    if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('Need at least 10 GiB free before starting')
    info=[]
    for s in subjects:
        cfg,source,ident,data,rows,splits,natural=work.data_parts(s)
        tokens,token_rows,_=work.load_imagery_truth(ROOT/f'data/processed/imagery/subj{s:02d}/imagery_trials.npz',s)
        expected={(r['stimulus_type'],r['stimulus_id']) for r in rows}
        if tuple(tokens.shape)!=(12,256,1664) or {(r['stimulus_type'],r['stimulus_id']) for r in token_rows}!=expected:
            raise ValueError(f'subj{s:02d}: image CLIP targets do not match stimuli')
        del tokens,token_rows
        if s==1:
            for task in TASKS:
                if not text_ok(s,task):raise ValueError(f'Subj01 {task} Ridge unavailable')
            for m in METHODS:
                old=work.old_run(m);done=read(old/'linear/runs/training_complete.json')
                if not matches(old/'linear/runs/best.pt',done['best_sha256']):raise ValueError('Subj01 adapter checkpoint changed')
                for task in TASKS:
                    sweep.SAMPLES=1;sweep.experiment=lambda _k,r=old:r/'linear'
                    if not sweep.evaluation_ok(10,task,'initial'):raise ValueError('Subj01 initial sample00 cannot be reused')
                    ref=ROOT/('output_1104/imagery_three_text_direct_v1' if m=='w010_mse' else 'output_1104/imagery_mu_clip_mse_v1')
                    sweep.experiment=lambda _k,r=ref:r/'linear'
                    if not sweep.evaluation_ok(10,task,'refined'):raise ValueError('Subj01 refined sample00 cannot be reused')
        else:
            vision.source_inputs(s)
            if not text_ok(s,'imagery'):raise ValueError(f'subj{s:02d} imagery text unavailable')
            if not (model_dir(s)/'model_complete.json').is_file():raise ValueError('Missing Ridge model')
        info.append({'subject':s,'voxels':int(data['fmri_scaled'].shape[-1]),'targets':len(rows),
                     'brainvae_sha256':ident['brainvae_sha256'],'w010_sha256':ident['w010_sha256'],
                     'image_target_cache_sha256':file_sha256(ROOT/f'data/processed/imagery/subj{s:02d}/imagery_trials.npz')})
        print('[preflight]',json.dumps(info[-1],ensure_ascii=False),flush=True)
        del data
    sweep.SAMPLES=3
    return info

def old_model_ok(s,m):
    folder=(work.old_run(m)/'linear/runs') if s==1 else (work.method_root(s,m)/'runs')
    p=folder/'training_complete.json'
    if not p.is_file() or read(p).get('status')!='complete' or not matches(folder/'best.pt',read(p)['best_sha256']):return False
    if s!=1 and ((folder/'last.pt').exists() or read(p).get('subject')!=s or read(p).get('method')!=m):return False
    return True

def prepare_ok(s):
    if s==1:return True
    folder=work.subject_root(s)/'retrieval60/protocol'
    return (folder/'identity.json').is_file() and (folder/'split.json').is_file() and read(folder/'identity.json')['subject']==s

def retrieval_ok(s):
    if s==1:return True
    root=work.subject_root(s)/'k10/protocol'
    return all((root/x).is_file() for x in ('split.json','retrieval.json','R_train_pairs.json','R_val_pairs.json','source.json')) and read(root/'source.json')['subject']==s

def teacher_ok(s):
    if s==1:return True
    root=work.subject_root(s)/'k10/teacher_cache';p=root/'state.json'
    return p.is_file() and read(p).get('complete') and matches(root/'teacher_tokens.npy',read(p)['cache_sha256'])

def prior_ok(s):
    if s==1:return True
    root=work.subject_root(s)/'k10';p=root/'teacher_prior.json'
    return p.is_file() and read(p).get('status')=='complete' and matches(root/'teacher_prior.pt',read(p)['sha256'])

def vision_mu_ok(s):
    if s==1:return text_ok(1,'vision')
    root=work.subject_root(s)/'vision_ridge_inputs';p=root/'imagery_mu_manifest.json'
    return p.is_file() and read(p).get('subject')==s and read(p).get('dataset')=='vision' and matches(root/'vision_mu.npy',read(p)['mu_sha256'])

def prediction_ok(s,m):
    root=work.method_root(s,m)
    if not (root/'prediction_complete.json').is_file():return False
    for task in TASKS:
        folder=root/'initial'/task;p=folder/'metadata.json'
        if not p.is_file() or read(p).get('subject')!=s or not matches(folder/'tokens/chunk00000.pt',read(p)['token_sha256']):return False
    return True

def generation_ok(s,m):
    root=work.method_root(s,m)
    return (root/'generation_complete.json').is_file() and all(sweep.checked_sample(root/'initial'/task,i) for task in TASKS for i in range(3))

def refine_ok(s,m,task):
    root=work.method_root(s,m)/'refined'/task;p=root/'refinement_complete.json'
    return p.is_file() and read(p).get('status')=='complete' and read(p).get('images')==36 and all(sweep.checked_sample(root,i) for i in range(3))

def eval_ok(s,m,task,phase):
    sweep.SAMPLES=3;sweep.experiment=lambda _k:work.method_root(s,m)
    return sweep.evaluation_ok(10,task,phase)

def evaluate(s,m,task,phase,device):
    sweep.SAMPLES=3;sweep.experiment=lambda _k:work.method_root(s,m)
    sweep.evaluate(10,task,phase,device)

def summarize(subjects):
    rows=[]
    for s in subjects:
        for m in METHODS:
            for task in TASKS:
                for phase in ('initial','refined'):
                    if not eval_ok(s,m,task,phase):raise ValueError('Incomplete evaluation')
                    info=read(work.method_root(s,m)/phase/task/'image_metrics.json')
                    for group,prefix in [('all',''),('natural','complex_'),('geometry','simple_')]:
                        for sample,values in [(f'sample{i:02d}',v) for i,v in enumerate(info['per_seed'])]+[('mean',info['metrics'])]:
                            vals={k:float(values[prefix+k]) for k in sweep.KEYS}
                            if not all(math.isfinite(v) for v in vals.values()):raise ValueError('Nonfinite metric')
                            rows.append({'subject':s,'method':m,'task':task,'phase':phase,'group':group,'sample':sample,**vals})
    names=['subject','method','task','phase','group','sample',*sweep.KEYS]
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
    atomic_json({'status':'complete','subjects':list(subjects),'rows':rows,'samples':3,
                 'note':'All stimuli seen in adapter training; imagery test uses held-out trials; vision uses 8 vision trials. Per-condition Ridge uses unadapted fMRI. Mean is over three generated samples.'},OUT/'summary.json')
    print('[summary]',OUT/'summary.csv',flush=True)

def child(key,cmd):
    global ACTIVE
    log=OUT/'logs'/f'{key}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        f.write('\nCOMMAND '+json.dumps(cmd)+'\n');f.flush()
        ACTIVE=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        rc=ACTIVE.wait();ACTIVE=None
    if rc:raise RuntimeError(f'{key} failed ({rc}); see {log}')

def handle_signal(signum,_):
    if ACTIVE is not None and ACTIVE.poll() is None:ACTIVE.send_signal(signum)
    raise SystemExit(128+signum)

def queue(subjects,poll,device):
    global LOCKED
    OUT.mkdir(parents=True,exist_ok=True);signal.signal(signal.SIGTERM,handle_signal);signal.signal(signal.SIGINT,handle_signal)
    with mapping_lock(OUT):
        LOCKED=True;q=StageQueue(OUT,seconds=120);info=preflight(subjects)
        dependencies=['run_multisubject_imagery_natural.py','run_imagery_clipl_direct.py','run_imagery_mu_clip_mse.py',
                      'predict_multisubject_ridge.py','multisubject_generate.py','sweep_subj01_retrieval_k.py',
                      'refine_w010_mindeye2_img2img.py']
        project_dependencies=['src/models/imagery_residual_comparison.py','src/utils/imagery_w010_pilot.py',
                              'src/utils/imagery_r_kl.py','src/utils/perception_anchor_prior.py',
                              'scripts/eval_scheme1_reconstruction.py']
        identity={'protocol':'subj01_08_w010_mu_mse_linear_3samples_v1','subjects':list(subjects),'methods':list(METHODS),
                  'tasks':list(TASKS),'samples':3,'seed_policy':'42+sample*100000+stimulus_index',
                  'source':info,'device':device,'worker_sha256':file_sha256(SCRIPT/'run_multisubject_linear_imagery.py'),
                  'vision_sha256':file_sha256(SCRIPT/'prepare_multisubject_vision_ridge_inputs.py'),
                  'refiner_sha256':file_sha256(SCRIPT/'refine_multisubject_vision_imagery_three.py'),
                  'queue_sha256':file_sha256(Path(__file__)),'cooldown_seconds':120,
                  'dependencies':{name:file_sha256(SCRIPT/name) for name in dependencies},
                  'project_dependencies':{name:file_sha256(ROOT/name) for name in project_dependencies}}
        immutable(OUT/'queue_identity.json',identity)
        if poll:
            from types import SimpleNamespace
            from wait_gpu_clip_to_synbrain import wait_for_idle
            q.state(status='waiting_for_gpu',stage=None,error=None)
            uuid=wait_for_idle(SimpleNamespace(gpu_index=0,max_used_mib=1024,max_utilization=5,poll_seconds=poll,idle_confirm_seconds=30,confirm_poll_seconds=10))
            os.environ['CUDA_VISIBLE_DEVICES']=uuid
        py=sys.executable
        def stage(key,script,args,verify):
            def action():
                if verify():print('[复用已核验产物]',key,flush=True)
                else:
                    import shutil
                    if shutil.disk_usage(ROOT).free<3*2**30:raise OSError('Less than 3 GiB free; preserve outputs and free disk before resuming')
                    child(key,[py,'-u',str(script),*map(str,args),'--device',device])
            q.run(key,action,verify)
        for s in subjects:
            label=f'subj{s:02d}';worker=SCRIPT/'run_multisubject_linear_imagery.py'
            if s!=1:
                for part,check in [('prepare',prepare_ok),('retrieval',retrieval_ok),('teachers',teacher_ok),('prior',prior_ok)]:
                    stage(f'{label}_{part}',worker,['--subject',s,'--stage',part],lambda s=s,c=check:c(s))
                stage(f'{label}_vision_mu',SCRIPT/'prepare_multisubject_vision_ridge_inputs.py',['--subject',s],lambda s=s:vision_mu_ok(s))
                pred,manifest=text_paths(s,'vision')
                stage(f'{label}_vision_ridge',SCRIPT/'predict_multisubject_ridge.py',['--subject',s,'--model-dir',model_dir(s),
                      '--mu-file',manifest.parent/'vision_mu.npy','--name','vision_multisubject_linear_v1','--resume'],lambda s=s:text_ok(s,'vision'))
            for method in METHODS:
                for part,check in [('train',old_model_ok),('predict',prediction_ok),('generate',generation_ok)]:
                    if s==1 and part=='train':
                        if not old_model_ok(s,method):raise ValueError('Subj01 old adapter missing')
                        continue
                    stage(f'{label}_{method}_{part}',worker,['--subject',s,'--method',method,'--stage',part],lambda s=s,m=method,c=check:c(s,m))
                for task in TASKS:
                    stage(f'{label}_{method}_initial_{task}_eval',Path(__file__),['--stage','evaluate','--subject',s,'--method',method,'--task',task,'--phase','initial'],lambda s=s,m=method,t=task:eval_ok(s,m,t,'initial'))
                    pred,manifest=text_paths(s,task);output=work.method_root(s,method)/'refined'/task
                    args=['--subject',s,'--task',task,'--source',work.method_root(s,method)/'initial'/task,'--prediction',pred,'--mu-manifest',manifest,'--output',output]
                    if s==1:
                        ref=ROOT/('output_1104/imagery_three_text_direct_v1' if method=='w010_mse' else 'output_1104/imagery_mu_clip_mse_v1')
                        args+=['--reuse-dir',ref/'linear/refined'/task,'--reuse-identity',ref/'identity.json']
                    stage(f'{label}_{method}_{task}_refine',SCRIPT/'refine_multisubject_vision_imagery_three.py',args,lambda s=s,m=method,t=task:refine_ok(s,m,t))
                    stage(f'{label}_{method}_refined_{task}_eval',Path(__file__),['--stage','evaluate','--subject',s,'--method',method,'--task',task,'--phase','refined'],lambda s=s,m=method,t=task:eval_ok(s,m,t,'refined'))
        summarize(subjects);q.state(status='complete',stage=None,error=None)
    print('[队列完成]',OUT,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('queue','preflight','evaluate','summary'),default='queue')
    p.add_argument('--subjects',type=int,nargs='+',default=list(SUBJECTS));p.add_argument('--subject',type=int,choices=SUBJECTS)
    p.add_argument('--method',choices=METHODS);p.add_argument('--task',choices=TASKS);p.add_argument('--phase',choices=('initial','refined'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-poll-seconds',type=int,default=1200);a=p.parse_args();torch.set_num_threads(4)
    if not a.subjects or len(set(a.subjects))!=len(a.subjects) or any(s not in SUBJECTS for s in a.subjects):p.error('Invalid subjects')
    if a.gpu_poll_seconds<0:p.error('Invalid poll interval')
    if a.gpu_poll_seconds and a.device!='cuda:0':p.error('GPU-idle wait selects physical GPU 0; use --device cuda:0')
    if a.stage=='preflight':preflight(a.subjects)
    elif a.stage=='evaluate':
        if not all((a.subject,a.method,a.task,a.phase)):p.error('Missing evaluation arguments')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        evaluate(a.subject,a.method,a.task,a.phase,torch.device(a.device))
    elif a.stage=='summary':summarize(a.subjects)
    else:
        try:queue(a.subjects,a.gpu_poll_seconds,a.device)
        except BaseException as error:
            if LOCKED and OUT.is_dir():StageQueue(OUT,seconds=120).state(status='interrupted' if isinstance(error,SystemExit) else 'failed',error=str(error))
            raise
if __name__=='__main__':main()
