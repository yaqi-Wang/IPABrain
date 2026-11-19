"""Split, pair and loss contracts for the frozen-w010 imagery pilot."""
from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def split_trials(data, seed=42, stimulus_type='complex'):
    """Split original trials within each run, before augmentation."""
    rng = np.random.default_rng(seed)
    task, kind = data['task_type'].astype(str), data['stimulus_type'].astype(str)
    ids = data['stimulus_id'].astype(str)
    targets = sorted(set(ids[(task == 'imagery') & (kind == stimulus_type)].tolist()))
    if len(targets) != 6:
        raise ValueError(f'Expected six targets, found {len(targets)}')
    rows = {}
    for label, target in enumerate(targets):
        img = np.flatnonzero((task == 'imagery') & (kind == stimulus_type) & (ids == target))
        vision = np.flatnonzero((task == 'vision') & (kind == stimulus_type) & (ids == target))
        runs = sorted(set(map(int, data['run'][img])))
        if len(img) != 16 or len(vision) != 8 or len(runs) != 2:
            raise ValueError(f'Unexpected imagery/vision structure for {target}')
        if np.any(data['condition_index'][vision] != 0):
            raise ValueError(f'Vision records include nonmatching conditions: {target}')
        splits = {'train': [], 'val': [], 'test': []}
        for run in runs:
            group = img[data['run'][img] == run]
            if len(group) != 8:
                raise ValueError(f'Expected 8 trials per imagery run: {target}/{run}')
            group = rng.permutation(group)
            splits['train'].extend(map(int, group[:5]))
            splits['val'].extend(map(int, group[5:6]))
            splits['test'].extend(map(int, group[6:]))
        vision = rng.permutation(vision)
        rows[target] = {'label': label, **splits, 'vision_train': list(map(int,vision[:6])),
                        'vision_val': list(map(int,vision[6:]))}
    for target, r in rows.items():
        assert (len(r['train']),len(r['val']),len(r['test'])) == (10,2,4)
        assert len(set(r['train']+r['val']+r['test'])) == 16
        assert not set(r['vision_train']).intersection(r['vision_val'])
    return {'protocol':'imagery_run_stratified_10_2_4_v1','seed':seed,
            'subject':int(data['subject_id']), 'stimulus_type':stimulus_type,
            'targets':targets,'rows':rows}


def trial_topk(query, bank, image_ids, repeats, k):
    """Cosine rank per image, then expand to distinct actual trial repeats."""
    query, bank = np.asarray(query,dtype=np.float32), np.asarray(bank,dtype=np.float32)
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError('CLIP retrieval embeddings have incompatible shapes')
    if len(bank) != len(image_ids) or len(np.unique(image_ids)) != len(image_ids):
        raise ValueError('Retrieval candidates must have unique aligned image IDs')
    qnorm = np.linalg.norm(query,axis=1,keepdims=True)
    bnorm = np.linalg.norm(bank,axis=1,keepdims=True)
    if np.any(qnorm == 0) or np.any(bnorm == 0):
        raise ValueError('Zero CLIP retrieval key')
    sims = (query/qnorm) @ (bank/bnorm).T
    rows = np.repeat(np.arange(len(image_ids),dtype=np.int64), repeats)
    ids = np.repeat(np.asarray(image_ids,dtype=np.int64),repeats)
    reps = np.tile(np.arange(repeats,dtype=np.int64),len(image_ids))
    if not 0 < k <= len(rows):
        raise ValueError('Invalid trial top-k')
    result=[]
    for scores in sims:
        score = np.repeat(scores,repeats)
        order=np.lexsort((reps,ids,-score))[:k]
        result.append([{'source_row':int(rows[x]),'image_id':int(ids[x]),
                        'repeat_index':int(reps[x]),'similarity':float(score[x])} for x in order])
    return result


def teacher_image_split(retrieved, seed=42, fraction=0.1):
    ids=sorted({r['image_id'] for group in retrieved for r in group})
    rng=np.random.default_rng(seed)
    n=max(1,min(len(ids)-1,round(len(ids)*fraction)))
    val=set(map(int,rng.permutation(ids)[:n]))
    return [[{**r,'teacher_split':'val' if r['image_id'] in val else 'train'} for r in group]
            for group in retrieved]


def pair_records(parent_rows, teacher_ids, seed, *, training=True):
    """One canonical pair per teacher; all parents appear before noise copies."""
    rng=np.random.default_rng(seed)
    teachers=list(map(int,rng.permutation(teacher_ids)))
    parents=list(map(int,rng.permutation(parent_rows)))
    if not parents or not teachers:
        raise ValueError('Need imagery parents and teacher trials')
    pairs=[]
    for i, teacher in enumerate(teachers):
        pairs.append({'parent_row':parents[i % len(parents)], 'teacher_index':teacher,
                      'augmented':i >= len(parents), 'noise_seed':int(rng.integers(1,2**31-1)),
                      'balance_repeat':False})
    return pairs


def balanced_schedule(groups, repeats_per_class=6, seed=42):
    """Visit every canonical pair; pad short groups for equal-class full batches."""
    rng=np.random.default_rng(seed)
    if len(groups) != 6 or any(not x for x in groups):
        raise ValueError('Balanced batches require six nonempty target groups')
    count=math.ceil(max(map(len,groups))/repeats_per_class)*repeats_per_class
    queues=[]
    for label, group in enumerate(groups):
        values=copy.deepcopy(group)
        for _ in range(count-len(values)):
            row=copy.deepcopy(group[int(rng.integers(len(group)))])
            row['balance_repeat']=True
            row['augmented']=True
            row['noise_seed']=int(rng.integers(1,2**31-1))
            values.append(row)
        rng.shuffle(values)
        for row in values: row['label']=label
        queues.append(values)
    batches=[]
    for start in range(0,count,repeats_per_class):
        batch=[];used=set()
        for label in range(6):
            selected=[]
            for i in range(repeats_per_class):
                # Only conflicts with another label matter; within-label repeat is allowed for V.
                pos=next((j for j in range(start+i,count)
                          if queues[label][j]['teacher_index'] not in used),None)
                if pos is None:
                    # Reassign this label's earlier slot only if doing so also
                    # leaves that earlier batch free of cross-label duplicates.
                    conflicting=queues[label][start+i]
                    for earlier in range(start):
                        candidate=queues[label][earlier]
                        if candidate['teacher_index'] in used:continue
                        previous=batches[earlier//repeats_per_class]
                        if any(r['label']!=label and r['teacher_index']==conflicting['teacher_index'] for r in previous):continue
                        queues[label][start+i],queues[label][earlier]=candidate,conflicting
                        previous[label*repeats_per_class+earlier%repeats_per_class]=conflicting
                        pos=start+i
                        break
                if pos is None: raise ValueError('Cannot avoid cross-label duplicate teachers in batch')
                queues[label][start+i],queues[label][pos]=queues[label][pos],queues[label][start+i]
                selected.append(queues[label][start+i])
            used.update(row['teacher_index'] for row in selected)
            batch.extend(selected)
        batches.append(batch)
    assert all(len(x)==36 and Counter(r['label'] for r in x)==Counter({i:6 for i in range(6)}) for x in batches)
    return batches


def anclip_loss(prediction, teacher, labels, temperature=0.05):
    """Single-direction, full-logical-batch, multi-positive contrast."""
    if len(prediction)!=len(teacher) or len(labels)!=len(prediction):
        raise ValueError('Mismatched contrastive batch')
    p=F.normalize(prediction.float().flatten(1),dim=1,eps=1e-12)
    t=F.normalize(teacher.float().flatten(1),dim=1,eps=1e-12)
    positive=labels[:,None].eq(labels[None,:])
    if not positive.any(1).all() or positive.all():
        raise ValueError('Each anchor needs positives and the batch needs negatives')
    logp=F.log_softmax((p @ t.T)/temperature,dim=1)
    return -(logp*positive).sum(dim=1).div(positive.sum(dim=1)).mean()


def alignment_loss(prediction, teacher, labels, weight=0.1, temperature=0.05):
    mse=F.mse_loss(prediction.float(),teacher.float())
    contrast=anclip_loss(prediction,teacher,labels,temperature)
    return mse+weight*contrast, mse, contrast


def replay_backward(adapter, frozen_mapping, inputs, teacher, labels, micro, scaler,
                    weight=0.1, temperature=0.05, check_replay=False):
    """Exact full-batch output loss with replayed dropout and frozen input gradients."""
    if micro <= 0 or len(inputs)!=len(teacher):
        raise ValueError('Invalid replay batch')
    states=[];predictions=[]
    with torch.no_grad():
        for start in range(0,len(inputs),micro):
            cpu=torch.get_rng_state()
            cuda=torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
            states.append((cpu,cuda))
            predictions.append(frozen_mapping(adapter(inputs[start:start+micro])).float())
    final_cpu=torch.get_rng_state()
    final_cuda=torch.cuda.get_rng_state(inputs.device) if inputs.is_cuda else None
    prediction=torch.cat(predictions).detach().requires_grad_(True)
    total,mse,contrast=alignment_loss(prediction,teacher,labels,weight,temperature)
    gradient,=torch.autograd.grad(total,prediction)
    maximum=0.0
    for index,start in enumerate(range(0,len(inputs),micro)):
        torch.set_rng_state(states[index][0])
        if inputs.is_cuda:torch.cuda.set_rng_state(states[index][1],inputs.device)
        replay=frozen_mapping(adapter(inputs[start:start+micro])).float()
        if check_replay:
            maximum=max(maximum,float((replay.detach()-predictions[index]).abs().max()))
        scaler.scale((replay*gradient[start:start+micro]).sum()).backward()
    torch.set_rng_state(final_cpu)
    if inputs.is_cuda:torch.cuda.set_rng_state(final_cuda,inputs.device)
    if check_replay and maximum>1e-4:
        raise RuntimeError(f'Dropout replay produced different outputs: {maximum}')
    return {'total':float(total.detach()),'mse':float(mse.detach()),
            'anclip':float(contrast.detach()),'replay_max_abs':maximum,
            'token_norm_mean':float(prediction.detach().flatten(1).norm(dim=1).mean()),
            'teacher_norm_mean':float(teacher.detach().float().flatten(1).norm(dim=1).mean())}
