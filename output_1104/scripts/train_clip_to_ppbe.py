#!/usr/bin/env python
"""Compare CLIP-Vision -> frozen PPBE mean mappers on a subject.

Train/validation split is the source BrainVAE's image-level NSD train split;
NSD test is only accessed after selection. No test images train the mapper.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.scheme1 import atomic_torch_save, check_source_files, file_sha256
from src.models.clip_to_brain_mappers import build_mapper
from src.utils.clip_mapping import (atomic_json, frozen_brain, load_brain_source,
                                    load_head_source_config, mapping_lock, source_nsd_dataset)
from src.utils.mapping_precision import mapping_autocast, mapping_scaler
from src.utils.ppbe_training import configure_ppbe_execution
from src.utils.timed_cooldown import TimedCooldown


def setup(args):
    cfg, brain_cfg, paths = load_head_source_config(args.source_config, ROOT)
    if brain_cfg['model'].get('architecture') != 'ppbe':
        raise ValueError('A PPBE source configuration is required')
    source = load_brain_source(cfg, brain_cfg, ROOT, args.subject, args.brainvae_checkpoint)
    if source['identity'].get('source_debug_training'):
        raise ValueError('Refusing a debug-trained BrainVAE checkpoint')
    check_source_files(paths.nsd_root, args.subject, source['manifest'],
                       ('train_fmri', 'train_ids', 'train_clip', 'test_fmri', 'test_ids', 'test_clip'))
    split = source['split']
    data = {}
    for name in ('train', 'val'):
        data[name] = source_nsd_dataset(source, paths.nsd_root, args.subject, 'train',
            split[name + '_indices'], average_repeats=True, expected_ids=split['source_image_ids'])
    data['test'] = source_nsd_dataset(source, paths.nsd_root, args.subject, 'test',
        np.arange(min(args.test_count, len(split['test_image_ids']))), average_repeats=True,
        expected_ids=split['test_image_ids'])
    shape = (brain_cfg['model']['latent_tokens'], brain_cfg['model']['embed_dim'])
    if any(tuple(x.clip.shape[1:]) != shape for x in data.values()):
        raise ValueError('CLIP token shape differs from BrainVAE latent shape')
    if args.test_count > len(split['test_image_ids']):
        raise ValueError('test-count exceeds available NSD test images')
    return source, data, shape


def loader(ds, batch, seed=None):
    return DataLoader(ds, batch_size=batch, shuffle=seed is not None, num_workers=0,
                      generator=torch.Generator().manual_seed(seed or 0))


def metrics_add(totals, pred, target, prefix):
    a, b = pred.float().flatten(1), target.float().flatten(1)
    totals[prefix+'_mse'] += float((a-b).square().mean(1).sum())
    totals[prefix+'_cosine'] += float(F.cosine_similarity(a,b,dim=1).sum())
    a = a-a.mean(1,keepdim=True); b = b-b.mean(1,keepdim=True)
    totals[prefix+'_pearson'] += float(F.cosine_similarity(a,b,dim=1).sum())


@torch.no_grad()
def evaluate(mapper, brain, ds, args, device, save=None):
    mapper.eval()
    totals = {p+'_'+m: 0. for p in ('latent','fmri') for m in ('mse','cosine','pearson')}
    n = 0
    predictions, truths, ids = [], [], []
    for batch in loader(ds, args.batch_size):
        fmri = batch['fmri'].to(device, dtype=torch.float32)
        clip = batch['clip_vision'].to(device, dtype=torch.float32)
        with mapping_autocast(device, args.precision):
            target = brain.encode(fmri).mode()
            pred = mapper(clip)
            recon = brain.decode(pred.float(), fmri.shape[-1]).squeeze(1)
        if not all(torch.isfinite(x).all() for x in (target, pred, recon)):
            raise FloatingPointError('Non-finite predictions or targets')
        metrics_add(totals, pred, target, 'latent')
        metrics_add(totals, recon, fmri, 'fmri')
        n += len(fmri)
        if save is not None:
            predictions.append(recon.float().cpu()); truths.append(fmri.cpu()); ids.append(batch['image_id'].cpu())
    if n == 0:
        raise ValueError('Empty evaluation set')
    result = {key: val/n for key,val in totals.items()}
    result['n_images'] = n
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        atomic_torch_save({'synthetic_fmri':torch.cat(predictions), 'true_fmri':torch.cat(truths),
                           'image_ids':torch.cat(ids), 'metrics':result}, save/'test_fmri.pt')
        atomic_json(result, save/'test_metrics.json')
    return result


def train(kind, source, data, shape, args, out, device, identity, warmstart=None, partition_outputs=False):
    folder = out/kind
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(args.seed)
    mapper = build_mapper(kind, tokens=shape[0], dim=shape[1], mlp_hidden=args.mlp_hidden).to(device)
    brain = frozen_brain(source, device)
    configure_ppbe_execution(brain, {'attention_backend':'sdpa'})
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = mapping_scaler(device, args.precision)
    if args.precision == 'fp16': scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    last, best = folder/'last.pt', folder/'best.pt'
    start, best_loss = 0, float('inf')
    if last.exists():
        ckpt = torch.load(last, map_location='cpu', weights_only=False)
        if ckpt['identity'] != identity or ckpt['kind'] != kind:
            raise ValueError('Existing checkpoint belongs to a different experiment')
        mapper.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler']); start=ckpt['epoch']; best_loss=ckpt['best_loss']
        torch.set_rng_state(ckpt['rng_cpu'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
        print(f'{kind}: resuming from epoch {start}', flush=True)
    elif best.exists():
        raise RuntimeError(f'{best} exists without last.pt; preserve it and use a new output directory')
    elif warmstart is not None:
        base = torch.load(warmstart, map_location='cpu', weights_only=False)
        if base['kind'] != kind or base['identity'] != identity['base_identity']:
            raise ValueError('Fine-tuning initialization does not match the NSD base model')
        mapper.load_state_dict(base['model'], strict=True)
        del base
        print(f'{kind}: initialized from {warmstart}', flush=True)
    print(json.dumps({'method':kind,'parameters':sum(x.numel() for x in mapper.parameters()),
                      'device':str(device),'train':len(data['train']),'val':len(data['val'])}), flush=True)
    for epoch in range(start, args.epochs):
        mapper.train(); total=0.; count=0; skipped=0
        for batch in loader(data['train'], args.batch_size, args.seed + epoch + 1):
            fmri=batch['fmri'].to(device, dtype=torch.float32)
            clip=batch['clip_vision'].to(device, dtype=torch.float32)
            with torch.no_grad(), mapping_autocast(device, args.precision):
                target=brain.encode(fmri).mode().float()
            optimizer.zero_grad(set_to_none=True)
            with mapping_autocast(device, args.precision): pred=mapper(clip)
            loss=F.mse_loss(pred.float(), target)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            scaler.scale(loss).backward()
            old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            total += float(loss.detach())*len(fmri); count += len(fmri)
        # Selection uses only latent MSE, avoiding repeated expensive voxel decoding.
        mapper.eval(); val_loss=0.; vn=0
        with torch.no_grad():
            for batch in loader(data['val'], args.batch_size):
                fmri=batch['fmri'].to(device, dtype=torch.float32)
                clip=batch['clip_vision'].to(device, dtype=torch.float32)
                with mapping_autocast(device, args.precision):
                    target=brain.encode(fmri).mode(); pred=mapper(clip)
                loss=F.mse_loss(pred.float(),target.float())
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite validation loss')
                val_loss+=float(loss)*len(fmri); vn+=len(fmri)
        if not count or not vn: raise ValueError('Empty training or validation set')
        val_loss/=vn
        improved=val_loss < best_loss
        if improved:
            best_loss=val_loss
            atomic_torch_save({'model':mapper.state_dict(), 'epoch':epoch+1, 'val_loss':val_loss,
                               'identity':identity, 'kind':kind}, best)
        payload={'model':mapper.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                 'epoch':epoch+1,'best_loss':best_loss,'identity':identity,'kind':kind,
                 'rng_cpu':torch.get_rng_state(),
                 'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}
        atomic_torch_save(payload,last)
        print(json.dumps({'method':kind,'epoch':epoch+1,'train_mse':total/count,'val_mse':val_loss,
                          'best_val_mse':best_loss,'amp_skipped_steps':skipped}), flush=True)
    if not best.is_file(): raise RuntimeError('Missing best model checkpoint')
    state=torch.load(best,map_location='cpu',weights_only=False)
    if state['identity'] != identity: raise ValueError('Best checkpoint identity mismatch')
    mapper.load_state_dict(state['model'])
    if partition_outputs:
        results={}
        for partition in ('train','val','test'):
            dataset=data[partition]
            target=folder/partition
            results[partition]=evaluate(mapper,brain,dataset,args,device,target)
            # Subset indices retain the source index for unambiguous pairing.
            records=[dataset.dataset.records[int(i)] for i in dataset.indices]
            atomic_json(records,target/'records.json')
            print(json.dumps({'method':kind,'partition':partition,'metrics':results[partition]}),flush=True)
    else:
        results=evaluate(mapper,brain,data['test'],args,device,folder)
        print(json.dumps({'method':kind,'selected_epoch':state['epoch'],'test':results}),flush=True)
    del brain,mapper,optimizer,scaler
    if device.type=='cuda': torch.cuda.empty_cache()
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',type=int,default=1,choices=range(1,9))
    p.add_argument('--source-config',default='configs/clip_mapping_ppbe_local_adatoken.yaml')
    p.add_argument('--brainvae-checkpoint',default='')
    p.add_argument('--output',default='output_1104/clip_to_ppbe_conditional_v1')
    p.add_argument('--methods',nargs='+',choices=['s2n','linear','mlp'],default=['s2n','linear','mlp'])
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--conditional-epochs',type=int,default=30)
    p.add_argument('--conditional-lr',type=float,default=1e-5)
    p.add_argument('--mlp-hidden',type=int,default=2048); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--test-count',type=int,default=250); p.add_argument('--device',default='cuda')
    p.add_argument('--precision',choices=['no','fp16','bf16'],default='fp16')
    p.add_argument('--skip-imagery',action='store_true',help='Skip imagery evaluation; required for subjects other than 01')
    p.add_argument('--preflight',action='store_true',help='Read-only data and provenance checks; no CUDA')
    args=p.parse_args()
    if min(args.epochs,args.conditional_epochs,args.batch_size,args.mlp_hidden,args.test_count)<=0 or not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.conditional_lr) or args.conditional_lr<=0 or not math.isfinite(args.weight_decay) or args.weight_decay<0:
        p.error('Invalid training arguments')
    if len(set(args.methods))!=len(args.methods): p.error('Duplicate methods')
    torch.set_num_threads(4)
    source,data,shape=setup(args)
    imagery=None; imagery_identity=None; image_splits=None; split_info=None
    if not args.skip_imagery:
        from clip_to_ppbe_imagery_data import imagery_pairs, stratified_image_split
        _, _, paths = load_head_source_config(args.source_config, ROOT)
        imagery,imagery_identity=imagery_pairs(source,paths,args.subject)
        image_splits,split_info=stratified_image_split(imagery,args.seed)
    settings={'subject':args.subject,'source':source['identity'], 'shape':list(shape),
              'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'weight_decay':args.weight_decay,
              'seed':args.seed,'mlp_hidden':args.mlp_hidden,'test_count':args.test_count,'precision':args.precision,
              'train_and_validation':'mean_fmri_per_image', 'selection':'validation_latent_mse',
              'test':'first_N_images_mean_fmri', 'imagery_data':imagery_identity,
              'imagery_split':split_info, 'conditional_epochs':args.conditional_epochs,
              'conditional_lr':args.conditional_lr,
              'imagery_inference':'separate vision and imagery mappers fine-tuned from same NSD base', 's2n':{'depth':8,'heads':13,'mlp_ratio':4.}}
    print(json.dumps({'status':'preflight_ok','source':settings,'images':{k:len(v) for k,v in data.items()},
                      'imagery_images':{task:{part:len(ds) for part,ds in splits.items()}
                                         for task,splits in image_splits.items()} if image_splits else {},
                      'test_used_for_selection':False},indent=2),flush=True)
    if args.preflight: return
    device=torch.device(args.device)
    if device.type != 'cuda' and args.precision != 'no': raise ValueError('Mixed precision requires CUDA')
    out=Path(args.output); out=(out if out.is_absolute() else ROOT/out)/f'subj{args.subject:02d}'
    with mapping_lock(out):
        protocol=out/'protocol.json'
        if protocol.exists() and json.loads(protocol.read_text()) != settings:
            raise ValueError('Output already belongs to different settings; choose a new --output')
        if not protocol.exists(): atomic_json(settings,protocol)
        base_identity={'stage':'nsd_base','settings':settings}
        results={'nsd_test':{},'conditional':{}}
        cooldown=TimedCooldown(out, threshold=3600, seconds=120)
        for method in args.methods:
            results['nsd_test'][method]=cooldown.run(
                'nsd_'+method, lambda: train(method,source,data,shape,args,out/'nsd',device,base_identity))
        if image_splits is not None:
            conditional_args=copy.copy(args)
            conditional_args.epochs=args.conditional_epochs
            conditional_args.lr=args.conditional_lr
            for method in args.methods:
                base_path=out/'nsd'/method/'best.pt'
                base_sha=file_sha256(base_path)
                results['conditional'][method]={}
                for task in ('vision','imagery'):
                    stage_identity={'stage':'conditional_'+task,'base_identity':base_identity,
                                    'base_checkpoint_sha256':base_sha}
                    results['conditional'][method][task]=cooldown.run(
                        'conditional_'+task+'_'+method,
                        lambda: train(method,source,image_splits[task],shape,conditional_args,
                                      out/'conditional'/task,device,stage_identity,
                                      warmstart=base_path,partition_outputs=True))
        atomic_json(results,out/'summary.json')
    print('[complete]',out,flush=True)

if __name__=='__main__': main()
