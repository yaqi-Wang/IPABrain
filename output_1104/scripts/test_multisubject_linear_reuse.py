#!/usr/bin/env python3
"""Verify subj01 existing sample00 reuse and text/trial identity on CPU."""
import sys,tempfile
from pathlib import Path
import torch
from torchvision.utils import save_image
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import run_multisubject_linear_imagery as w
import queue_multisubject_linear_imagery as q
import refine_multisubject_vision_imagery_three as refine
from multisubject_generate import generate_samples_verified

def fake_decode(tokens,engine,suffix,offset,seed):return torch.zeros((1,3,4,4))
with tempfile.TemporaryDirectory(prefix='multisubject_reuse_') as tmp:
    w.OUT=Path(tmp)
    for method in w.METHODS:
        w.predict(1,method,torch.device('cpu'))
        for task in w.TASKS:
            source=w.method_root(1,method)/'initial'/task
            meta=w.read(source/'metadata.json');token=source/'tokens/chunk00000.pt'
            payload=torch.load(token,map_location='cpu',weights_only=False);records=payload['records']
            old=w.old_run(method)/'linear/initial'/task/'sample00/manifest.json'
            items=w.read(old)['items'];targets={x['stimulus_id']:Path(x['target_image']) for x in items}
            reused=w.sample00_reuse(1,method,task,records)
            assert len(reused)==12
            generate_samples_verified([token],source,meta,None,None,0,fake_decode,save_image,3,42,targets,reuse=reused)
            pred,mu_manifest=q.text_paths(1,task)
            _,_,inputs,ident=refine.checked(source,pred,mu_manifest,1,task)
            ref=ROOT/('output_1104/imagery_three_text_direct_v1' if method=='w010_mse' else 'output_1104/imagery_mu_clip_mse_v1')
            paths,provenance=refine.reuse_sample00(ref/'identity.json',ref/'linear/refined'/task,ident,inputs,task)
            assert len(paths)==12 and len(provenance['image_sha256'])==12
            print('verified',method,task,'first initial and refinement sample reused',flush=True)
