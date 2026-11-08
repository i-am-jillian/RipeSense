"""Prepare licensed avocado stages and tomato quality without mixing source groups."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import urllib.request
import zipfile

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'datasets/camera_extras'
AVOCADO='https://data.mendeley.com/datasets/3xd9n945v8/1'
TOMATO='https://data.mendeley.com/datasets/vy5y8p2kdb/1'


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=45) as response:return response.read()


def prepare_avocado():
    table=pd.read_excel(ROOT/'datasets/raw/Avocado_Ripening.xlsx')
    table=table[table['Ripening Index Classification'].isin([1,2,3,4,5])].copy()
    cache=ROOT/'datasets/raw/hass_file_index.json'
    if cache.exists():files=json.loads(cache.read_text())
    else:
        def page(start):
            url=('https://data.mendeley.com/public-api/datasets/3xd9n945v8/files?'
                 f'folder_id=9685434a-9bdd-4088-8e71-f025b4c91bff&version=1&$start={start}&$limit=1000')
            return json.loads(fetch(url))
        with ThreadPoolExecutor(max_workers=4) as executor:pages=list(executor.map(page,range(0,15000,1000)))
        files=[r for p in pages for r in p]
        cache.write_text(json.dumps(files))
    by_name={Path(r['filename']).stem:r for r in files}
    labels={1:'unripe',2:'breaking',3:'ripe_first_stage',4:'ripe_second_stage',5:'overripe'}
    rows=table.to_dict('records');random.Random(42).shuffle(rows)
    selected=[];counts=Counter();missing=0
    for row in rows:
        name=str(row['File Name']).strip()
        file=by_name.get(name)
        if not file:missing+=1;continue
        group=f"{row['Storage Group']}_{int(row['Sample'])}"
        value=int(hashlib.sha256(('42/'+group).encode()).hexdigest()[:8],16)/2**32
        split='test' if value<.15 else 'val' if value<.30 else 'train'
        label=labels[int(row['Ripening Index Classification'])]
        cap=200 if split=='train' else 50
        if counts[split,label]>=cap:continue
        selected.append((file,split,label,group))
        counts[split,label]+=1
    print('Avocado reconciliation:',len(files),'files;',len(table),'labeled rows;',missing,'missing file joins',flush=True)
    print('Avocado selected:',dict(counts),flush=True)
    def download(record):
        file,split,label,group=record
        dest=DATA/'avocado'/split/label/file['filename']
        dest.parent.mkdir(parents=True,exist_ok=True)
        raw=fetch(file['content_details']['download_url'])
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=file['content_details']['sha256_hash']:
            raise ValueError('Avocado source checksum mismatch')
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        h,w=image.shape[:2];scale=min(1.,384/max(h,w))
        image=cv2.resize(image,(round(w*scale),round(h*scale)))
        cv2.imwrite(str(dest),image)
        return {'path':str(dest.relative_to(ROOT)),'split':split,'label':label,'group':group,
                'source':AVOCADO,'license':'CC BY 4.0','filename':file['filename'],
                'source_sha256':digest,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    manifest=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i,row in enumerate(executor.map(download,selected),1):
            manifest.append(row)
            if i%100==0:print('Avocado downloads',i,'/',len(selected),flush=True)
    return manifest


def prepare_tomato():
    manifest=[];seen={};source_groups={}
    with zipfile.ZipFile(ROOT/'datasets/raw/Tomato_Dataset.v5.zip') as archive:
        yml=next((n for n in archive.namelist() if n.endswith('data.yaml')),None)
        if yml:
            classes=yaml.safe_load(archive.read(yml))['names']
            if isinstance(classes,dict):classes=[classes[k] for k in sorted(classes)]
            mapping={i:('fresh' if 'fresh' in name.lower() else 'spoiled' if 'rot' in name.lower() else None)
                     for i,name in enumerate(classes)}
            assert all(mapping.values()),classes
        else:
            votes={}
            for name in archive.namelist():
                if '/labels/' not in name or not name.endswith('.txt'):continue
                label='fresh' if 'fresh' in name.lower() else 'spoiled' if 'rot' in name.lower() else None
                if label is None:continue
                for row in archive.read(name).decode().splitlines():
                    if row.strip():votes.setdefault(int(row.split()[0]),Counter())[label]+=1
            mapping={}
            for cid,counts in votes.items():
                label,n=counts.most_common(1)[0]
                assert n/sum(counts.values())>.95,counts
                mapping[cid]=label
            classes=[mapping[k] for k in sorted(mapping)]
            print('Tomato filename/annotation mapping:',votes,flush=True)
        print('Tomato annotation classes:',classes,mapping,flush=True)
        images={str(Path(n).with_suffix('')):n for n in archive.namelist() if n.endswith('.jpg')}
        labels=[n for n in archive.namelist() if '/labels/' in n and n.endswith('.txt')]
        labels.sort(key=lambda n:0 if '/test/' in n else 1 if '/valid/' in n else 2)
        for name in labels:
            ip=images.get(str(Path(name.replace('/labels/','/images/')).with_suffix('')))
            if not ip:continue
            split='test' if '/test/' in name else 'val' if '/valid/' in name else 'train'
            raw=archive.read(ip);digest=hashlib.sha256(raw).hexdigest()
            group=Path(ip).name.split('.rf.')[0]
            if digest in seen or (group in source_groups and source_groups[group]!=split):continue
            seen[digest]=split;source_groups[group]=split
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            h,w=image.shape[:2]
            for i,line in enumerate(archive.read(name).decode().splitlines()):
                parts=line.split()
                if len(parts)!=5:continue
                cid=int(parts[0]);cx,cy,bw,bh=map(float,parts[1:]);bw*=1.24;bh*=1.24
                x0,x1=max(0,int((cx-bw/2)*w)),min(w,int((cx+bw/2)*w))
                y0,y1=max(0,int((cy-bh/2)*h)),min(h,int((cy+bh/2)*h))
                if min(x1-x0,y1-y0)<20:continue
                crop=image[y0:y1,x0:x1];scale=min(1.,384/max(crop.shape[:2]))
                crop=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
                label=mapping[cid];dest=DATA/'tomato'/split/label/f'{Path(ip).stem}_{i}.jpg'
                dest.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(dest),crop)
                manifest.append({'path':str(dest.relative_to(ROOT)),'split':split,'label':label,'group':group,
                                 'source':TOMATO,'license':'CC BY 4.0','source_sha256':digest,
                                 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    print('Tomato selected:',dict(Counter((r['split'],r['label']) for r in manifest)),flush=True)
    return manifest


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    tomato=prepare_tomato()
    (DATA/'tomato_manifest.json').write_text(json.dumps(tomato,indent=1))
    avocado=prepare_avocado()
    (DATA/'avocado_manifest.json').write_text(json.dumps(avocado,indent=1))
    print('Extra datasets ready',flush=True)


if __name__=='__main__':main()
