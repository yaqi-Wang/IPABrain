#!/usr/bin/env python3
"""Extract one matched COCO caption's SDXL text features per NSD image.

This is a new target extractor, not a training or reconstruction queue.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'output_1104/scripts')]
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
import open_clip
import extract_sdxl_clip_l11 as common
import extract_sdxl_bigg_text as bigg
import extract_imagery_stimulus_sdxl_text as encoder
from src.data.scheme1 import file_sha256
from src.utils.clip_mapping import atomic_json, mapping_lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,required=True,choices=range(1,9))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--limit',type=int,default=0,help='first N images per split, for smoke checks only')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--output-dir',type=Path,help='new, isolated output directory')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.batch_size<1 or a.limit<0:p.error('invalid batch size or limit')
    ids,paths=common.read_ids(a.subject,['train','test'],a.limit)
    sel=json.loads((ROOT/f'outputs_new/w010_multisubject_v1/subj{a.subject:02d}/selection.json').read_text())['selected']
    protocol=json.loads((Path(sel['checkpoint']).parent.parent/'protocol.json').read_text())
    with np.load(Path(protocol['prepared_data'])/'split.npz',allow_pickle=False) as split:
        if (not np.array_equal(ids['train'],split['source_image_ids'][:len(ids['train'])]) or
            not np.array_equal(ids['test'],split['test_image_ids'][:len(ids['test'])])):
            raise ValueError('NSD IDs differ from selected BrainVAE/w010 prepared data')
    grouped=common.read_captions(common.sources(common.NSD_ROOT),np.concatenate(list(ids.values())),'first')
    records={k:common.build_records(v,grouped)[0] for k,v in ids.items()}
    for split in records:
        if len(records[split])!=len(ids[split]) or any(int(r['image_id'])!=int(i) for r,i in zip(records[split],ids[split])):
            raise ValueError('Caption/NSD ID order mismatch')
    print('[aligned] subj{:02d} train={} test={} first COCO caption per image'.format(a.subject,len(ids['train']),len(ids['test'])),flush=True)
    if a.dry_run:return
    device=torch.device(a.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    base=ROOT/'output_1104/multisubject_nsd_imagery_v1'
    out=a.output_dir or (base/'debug' if a.limit else base)/f'subj{a.subject:02d}'/'coco_text_resumable'
    out=out.resolve()
    with mapping_lock(out):
        provenance=encoder.weight_provenance()
        identity={'protocol':'multisubject_sdxl_first_caption_resumable_v1',
                  'subject':a.subject,'limit':a.limit,'counts':{k:len(v) for k,v in ids.items()},
                  'ids_sha256':{k:hashlib.sha256(np.asarray(v,dtype=np.int64).tobytes()).hexdigest() for k,v in ids.items()},
                  'records_sha256':hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                  'weights':provenance,'selected_w010_checkpoint_sha256':sel['checkpoint_sha256'],
                  'script_sha256':file_sha256(Path(__file__)),'dtype':'float16'}
        state_path=out/'extraction_progress.json'
        complete_path=out/'extraction_complete.json'
        if complete_path.exists():
            done=json.loads(complete_path.read_text())
            if done.get('identity')!=identity or any(not (out/name).is_file() or file_sha256(out/name)!=entry['sha256'] for name,entry in done['files'].items()):
                raise ValueError('Completed extraction identity or output hashes differ')
            print('[verified complete] '+str(out),flush=True)
            return
        existing=[x for x in out.iterdir() if x.name!='.mapping.lock']
        if existing and not a.resume:
            raise FileExistsError('Output exists; pass --resume for a matching resumable run')
        if existing and not state_path.is_file():
            raise RuntimeError('Incomplete output without a progress record: preserve and inspect it')
        state=json.loads(state_path.read_text()) if state_path.exists() else None
        if state is not None and state.get('identity')!=identity:
            raise ValueError('Resume identity differs; create a new output directory')
        shapes={'clip_l':(77,768),'bigg_tokens':(77,1280),'bigg_pooled':(1280,)}
        if state is None:
            needed=sum(len(v)*sum(int(np.prod(s))*2 for s in shapes.values()) for v in ids.values())
            if shutil.disk_usage(out).free < needed+5*1024**3:
                raise RuntimeError('Insufficient disk space for text arrays plus 5 GiB reserve')
            state={'identity':identity,'next_index':{'clip_l':0,'bigg':0},'initialized':False}
            atomic_json(state,state_path)
        elif shutil.disk_usage(out).free < 2*1024**3:
            raise RuntimeError('Less than 2 GiB disk space remains; resume after freeing space')
        tok=CLIPTokenizer.from_pretrained(str(common.CLIP_DIR),local_files_only=True)
        all_records=records['train']+records['test']; texts=[x['text'] for x in all_records]
        ids_l=tok(texts,padding='max_length',truncation=True,max_length=77,return_tensors='pt')['input_ids']
        ids_g=open_clip.tokenize(texts,context_length=77)
        arrays={}
        for split in ('train','test'):
            if not state['initialized']:
                np.save(out/(split+'_image_ids.npy'),ids[split])
                with (out/(split+'_captions.jsonl')).open('w',encoding='utf-8') as f:
                    for row in records[split]:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
            elif not np.array_equal(np.load(out/(split+'_image_ids.npy'),allow_pickle=False),ids[split]):
                raise ValueError('Saved image IDs differ from source')
            for name,shape in shapes.items():
                path=out/(split+'_'+name+'.npy')
                mode='r+' if state['initialized'] else 'w+'
                arrays[split,name]=np.lib.format.open_memmap(path,mode=mode,dtype=np.float16,shape=(len(ids[split]),*shape))
                if arrays[split,name].shape!=(len(ids[split]),*shape) or arrays[split,name].dtype!=np.float16:
                    raise ValueError('Saved array shape or dtype differs')
        if not state['initialized']:
            for array in arrays.values():array.flush()
            state['initialized']=True
            atomic_json(state,state_path)
        # Each frozen tower is loaded and verified once, then reused for every batch.
        for name in ('clip_l','bigg'):
            if state['next_index'][name]==len(all_records):
                print('[resume] {} already complete'.format(name),flush=True)
                continue
            if name=='clip_l':
                model=CLIPTextModel.from_pretrained(str(common.CLIP_DIR),local_files_only=True,use_safetensors=True,
                                                    torch_dtype=torch.float16,attn_implementation='sdpa').to(device).eval().requires_grad_(False)
                verifier=common.FastClipL11(model).to(device).eval().requires_grad_(False)
                common.verify_fast(model,verifier,ids_l[:min(4,len(ids_l))].to(device),'float16')
                del verifier
            else:
                model=bigg.load_model(bigg.CHECKPOINT,device,'float16')
                bigg.verify_against_openclip(model,ids_g[:min(2,len(ids_g))].to(device),'float16')
            for start in range(state['next_index'][name],len(all_records),a.batch_size):
                end=min(start+a.batch_size,len(all_records))
                with torch.inference_mode():
                    if name=='clip_l':
                        value=model(input_ids=ids_l[start:end].to(device),output_hidden_states=True,return_dict=True).hidden_states[11]
                        outputs={'clip_l':value.float().cpu().numpy()}
                    else:
                        tokens,pooled=bigg.encode(model,ids_g[start:end].to(device))
                        outputs={'bigg_tokens':tokens.float().cpu().numpy(),'bigg_pooled':pooled.float().cpu().numpy()}
                if any(not np.isfinite(value).all() for value in outputs.values()):raise FloatingPointError('Nonfinite text features')
                boundary=len(ids['train'])
                for split,lo,hi in [('train',start,min(end,boundary)),('test',max(start,boundary),end)]:
                    if lo>=hi:continue
                    dest=lo if split=='train' else lo-boundary
                    for branch,value in outputs.items():arrays[split,branch][dest:dest+hi-lo]=value[lo-start:hi-start].astype(np.float16)
                for split in ('train','test'):
                    for branch in outputs:arrays[split,branch].flush()
                state['next_index'][name]=end
                atomic_json(state,state_path)
                print('[extract] {} {}/{}'.format(name,end,len(all_records)),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        for array in arrays.values():array.flush()
        del arrays
        files={str(path.name):{'sha256':file_sha256(path)} for path in out.iterdir() if path.suffix in ('.npy','.jsonl')}
        atomic_json({'status':'complete','subject':a.subject,'debug':bool(a.limit),'caption_policy':'first_nonempty_coco_caption','identity':identity,
                     'counts':{k:len(v) for k,v in ids.items()},'weights':provenance,'files':files,
                     'selected_w010_checkpoint_sha256':sel['checkpoint_sha256'],
                     'note':'NSD test captions are saved for reference; forbidden as Ridge inference or model-selection inputs.'},out/'extraction_complete.json')
        print('[complete] '+str(out),flush=True)


if __name__=='__main__':main()
