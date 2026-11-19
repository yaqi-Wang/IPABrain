"""Image-by-image resumable UnCLIP generation with immutable input identity."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from src.data.scheme1 import file_sha256
from src.utils.staged_queue import atomic_json
from multisubject_resume import archive_file


def generate_samples_verified(chunks,directory,metadata,engine,vector_suffix,offset_noise,decode_fn,save_fn,
                              num_samples,base_seed,target_index,reuse=None):
    directory=Path(directory);reuse=reuse or {}
    records=[]
    for chunk in chunks:
        chunk=Path(chunk);payload=torch.load(chunk,map_location='cpu',weights_only=False)
        if len(payload['records'])!=len(payload['mean']):raise ValueError('Token/record count mismatch')
        for offset,row in enumerate(payload['records']):records.append((chunk,offset,row,payload['mean'][offset:offset+1]))
    for sample_index in range(num_samples):
        sample_dir=directory/f'sample{sample_index:02d}';images=sample_dir/'images';images.mkdir(parents=True,exist_ok=True)
        manifest_items=[]
        for index,(chunk,offset,row,_token) in enumerate(records):
            seed=base_seed+sample_index*100000+index
            image=images/f'item{index:05d}_{row["stimulus_id"]}.png'
            item={**row,'item_index':index,'sample_index':sample_index,'seed':seed,
                  'path':str(image.relative_to(sample_dir)),'token_chunk':str(chunk),'token_row':offset}
            if row['stimulus_id'] in target_index:item['target_image']=str(target_index[row['stimulus_id']])
            manifest_items.append(item)
        level={'mean_fmri':'stimulus_level','image_level':'image_level'}.get(metadata['aggregation'],'trial_level')
        manifest={**metadata,'level':level,'sample_index':sample_index,'n_images':len(records),'items':manifest_items}
        manifest_path=sample_dir/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Existing manifest differs')
        if not manifest_path.exists():atomic_json(manifest,manifest_path)
        ledger=sample_dir/'progress.jsonl';saved={}
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                row=json.loads(line);index=int(row['index'])
                if not 0<=index<len(records) or row['seed']!=manifest_items[index]['seed'] or row['stimulus_id']!=records[index][2]['stimulus_id']:
                    raise ValueError('Image progress belongs to a different run')
                saved[index]=row
        with ledger.open('a',encoding='utf-8') as progress:
            for index,(_chunk,_offset,row,token) in enumerate(records):
                item=manifest_items[index];image=sample_dir/item['path'];record=saved.get(index)
                if record and image.is_file() and file_sha256(image)==record['sha256']:continue
                if image.exists():
                    # A file without a matching receipt is preserved, never silently reused.
                    archive_file(image)
                key=(sample_index,index)
                if key in reuse:
                    import shutil
                    source,expected=reuse[key]
                    if file_sha256(source)!=expected:raise ValueError('Reused image checksum mismatch')
                    temp=image.with_name(image.stem+'.tmp.png');shutil.copyfile(source,temp);temp.replace(image)
                else:
                    print(f'[generate] sample={sample_index} item={index+1}/{len(records)} seed={item["seed"]}',flush=True)
                    generated=decode_fn(token,engine,vector_suffix,offset_noise,seed=item['seed'])
                    temp=image.with_name(image.stem+'.tmp.png');save_fn(generated[0].cpu(),temp);temp.replace(image)
                result={'index':index,'stimulus_id':row['stimulus_id'],'seed':item['seed'],'sha256':file_sha256(image)}
                progress.write(json.dumps(result)+'\n');progress.flush()
