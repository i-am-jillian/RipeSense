"""Standalone DeepHS feature workflow; embedded verbatim in the Colab notebook."""
from pathlib import Path
import hashlib, io, json, re, tempfile, zipfile, time
import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip
from scipy import ndimage
from PIL import Image

DEEPHS_BASE = 'https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/'
ANNOTATION_NAME = 'annotations-upd-2024-01-09.zip'
SOURCE_COMMIT = '5f47812a3fb5bd8f0ed5367cedf9b464f83076c1'
FRUITS = {'Avocado':'avocado', 'Kiwi':'kiwi', 'Kaki':'persimmon', 'Papaya':'papaya', 'Mango':'mango'}
STAGES = ['unripe', 'ripe', 'over-ripe']
CENTERS = np.array([415,445,480,515,555,590,630,680], dtype=float)
FWHM = np.array([26,30,36,39,39,40,50,52], dtype=float)
FEATURES = [f'F{i+1}_{int(w)}nm_mean_reflectance' for i,w in enumerate(CENTERS)]
ALGORITHM = 'deephs-vis-gaussian-fruitmask-v1'


def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)); tmp.replace(path)


def get_annotations(cache):
    """Numeric record_id is local to each JSON; join before concatenating splits."""
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    path=cache/ANNOTATION_NAME
    if not path.exists():
        r=requests.get(DEEPHS_BASE+ANNOTATION_NAME,timeout=90); r.raise_for_status()
        if len(r.content)>5_000_000: raise ValueError('Unexpected annotation download size')
        zipfile.ZipFile(io.BytesIO(r.content)).testzip()
        path.write_bytes(r.content)
    rows=[]; wavelengths=None
    with zipfile.ZipFile(path) as z:
        for split in ['train_only_labeled_v2','val_v2','test_v2']:
            d=json.loads(z.read('annotations/'+split+'.json'))
            wave=next(c['wavelengths'] for c in d['cameras'] if c['id']=='VIS')
            if wavelengths is not None: assert wave==wavelengths
            wavelengths=wave
            annotations={a['record_id']:a for a in d['annotations']}
            assert len(annotations)==len(d['annotations'])
            for r in d['records']:
                if r['camera_type']!='VIS' or r['fruit'] not in FRUITS or r['id'] not in annotations: continue
                a=annotations[r['id']]
                stage={'unripe':'unripe','perfect':'ripe','overripe':'over-ripe'}.get(a['ripeness_state'])
                if stage is None: raise ValueError('Unexpected coarse ripeness label')
                day=r['day']; match=re.search(r'_(\d+)_(front|back)\.bin$',r['files']['data_file'])
                if not match: raise ValueError('Unverified specimen filename: '+str(r))
                series=next((f'm{i}' for i in (2,3,4) if f'm{i}' in day),'m1')
                assert match[2]==r['side']
                specimen=f"{FRUITS[r['fruit']]}/{series}/{match[1]}"
                key=hashlib.sha256(r['files']['data_file'].encode()).hexdigest()[:24]
                rows.append(dict(key=key,fruit=FRUITS[r['fruit']],archive=r['fruit']+'.zip',
                    specimen=specimen,series=series,day=day,side=r['side'],stage=stage,
                    source_stage=a['ripeness_state'],fine_stage=a.get('ripeness_state_fine'),
                    header=r['files']['header_file'],data=r['files']['data_file'],official_split=split))
    df=pd.DataFrame(rows)
    assert df.key.is_unique and df.data.is_unique
    assert (df.groupby('specimen').stage.nunique()==1).all(), 'Conflicting specimen labels; audit first'
    wavelengths=np.asarray(wavelengths,dtype=float)
    assert len(wavelengths)==224 and np.all(np.diff(wavelengths)>0)
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    return df,wavelengths,source_hash


def make_plan(records, mode='smoke', seed=42, smoke_fruits=('mango',)):
    """Custom specimen-stratified split; never use a recording-level official split."""
    assert mode in ('smoke','full')
    if mode=='smoke':
        assert set(smoke_fruits) <= set(FRUITS.values()) and len(smoke_fruits)
        records=records[records.fruit.isin(smoke_fruits)]
    selected=[]; assignments={}
    for (fruit,stage),g in records.groupby(['fruit','stage'],sort=True):
        specimens=sorted(g.specimen.unique(),key=lambda s:hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
        if mode=='smoke': specimens=specimens[:3]
        # One locked split per mode. Smoke is an intentionally tiny pipeline check.
        ntest=max(1,int(np.ceil(len(specimens)*0.25))) if len(specimens)>=2 else 0
        for i,s in enumerate(specimens): assignments[s]='test' if i<ntest else 'train'
        selected.extend(specimens)
    plan=records[records.specimen.isin(selected)].copy()
    if mode=='smoke':
        # A single view per specimen saves bandwidth; omitted views are never used in another split.
        plan=plan.sort_values(['specimen','day','side']).drop_duplicates('specimen')
    plan['split']=plan.specimen.map(assignments)
    assert (plan.groupby('specimen').split.nunique()==1).all()
    return plan.sort_values(['fruit','split','stage','specimen','day','side']).reset_index(drop=True)


def channel_weights(wavelengths):
    w=np.asarray(wavelengths,dtype=float)
    assert w.min() <= 400 and w.max() >= 710, 'Insufficient visible band coverage'
    # Gaussian approximation; wavelength-bin widths account for irregular sampling.
    weights=np.exp(-0.5*((w[:,None]-CENTERS)/(FWHM/2.35482))**2)*np.gradient(w)[:,None]
    weights /= weights.sum(axis=0,keepdims=True)
    return weights


def read_envi(header_bytes, data_bytes):
    """Read documented ENVI BIP/float32 layout, fail on an unexpected encoding."""
    meta=dict(re.findall(r'^\s*([^=\n]+?)\s*=\s*([^\n]+)',header_bytes.decode(),flags=re.M))
    meta={k.strip().lower():v.strip() for k,v in meta.items()}
    assert meta['interleave'].lower()=='bip' and int(meta['data type'])==4
    shape=tuple(int(meta[k]) for k in ('lines','samples','bands'))
    offset=int(meta.get('header offset',0)); endian='<' if int(meta['byte order'])==0 else '>'
    assert len(data_bytes)-offset==np.prod(shape)*4, 'Truncated/mismatched cube'
    return np.frombuffer(data_bytes,dtype=endian+'f4',offset=offset).reshape(shape)


def extract_features(cube,wavelengths):
    assert cube.ndim==3 and cube.shape[2]==len(wavelengths)
    # Source cubes are already referenced. No second dark/white normalization.
    valid=np.isfinite(cube).all(axis=2)
    brightness=np.mean(cube,axis=2)
    candidate=valid & (brightness>0.05) & (brightness<1.5)
    candidate=ndimage.binary_fill_holes(candidate)
    components,n=ndimage.label(candidate)
    if not n: raise ValueError('No fruit foreground above referenced mean 0.05')
    sizes=np.bincount(components.ravel()); sizes[0]=0
    mask=components==sizes.argmax()
    # Erode one pixel to reduce mixed boundary spectra; use all image sizes without fixed-column cuts.
    inner=ndimage.binary_erosion(mask,iterations=1)
    if inner.sum()>=25: mask=inner
    mask &= valid
    fraction=float(mask.mean())
    if mask.sum()<25 or not 0.02 <= fraction <= 0.98:
        raise ValueError(f'Implausible fruit region: {fraction:.3f}; inspect segmentation')
    mean_spectrum=np.mean(cube[mask],axis=0,dtype=np.float64)
    vector=mean_spectrum@channel_weights(wavelengths)
    if not np.isfinite(vector).all(): raise ValueError('Nonfinite feature')
    rgb=cube[:,:,[np.argmin(abs(wavelengths-x)) for x in (630,555,480)]]
    preview=np.clip(np.nan_to_num(rgb),0,1)**(1/2.2)
    preview=(preview*255).astype(np.uint8)
    edge=mask ^ ndimage.binary_erosion(mask)
    preview[edge]=[255,0,255]
    qc=dict(mask_fraction=fraction,mask_pixels=int(mask.sum()),shape=list(cube.shape),
            reflectance_min=float(np.min(cube[mask])),reflectance_max=float(np.max(cube[mask])))
    return vector,preview,qc


def feature_signature(wavelengths,annotation_hash):
    spec=dict(algorithm=ALGORITHM,annotation_sha256=annotation_hash,source_commit=SOURCE_COMMIT,
              wavelengths_nm=list(map(float,wavelengths)),centers_nm=CENTERS.tolist(),fwhm_nm=FWHM.tolist(),
              feature_order=FEATURES,mask='finite; 0.05 < mean(all224bands) < 1.5; largest component; erode1',
              calibration='already-referenced DeepHS cubes; no second correction',sensor_ready=False)
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    return digest,spec


def process_batch(plan,wavelengths,cache,limit=15,max_member_mb=500):
    """Commit after each recording. No full archive downloads; missing rows retry next time."""
    cache=Path(cache); (cache/'rows').mkdir(parents=True,exist_ok=True); (cache/'previews').mkdir(exist_ok=True)
    pending=plan[~plan.key.map(lambda k:(cache/'rows'/f'{k}.json').exists())].head(limit)
    completed=0; failures=[]
    for archive,g in pending.groupby('archive',sort=False):
        try:
            with RemoteZip(DEEPHS_BASE+archive,timeout=180) as z:
                for r in g.to_dict('records'):
                    try:
                        hi,di=z.getinfo(r['header']),z.getinfo(r['data'])
                        if max(hi.file_size,di.file_size)>max_member_mb*1_000_000:
                            raise ValueError('Member exceeds RAM guard; increase only after checking Colab memory')
                        # RemoteZip validates the member CRC during read.
                        h=z.read(r['header']); b=z.read(r['data'])
                        cube=read_envi(h,b)
                        vector,preview,qc=extract_features(cube,wavelengths)
                        Image.fromarray(preview).save(cache/'previews'/f"{r['key']}.png")
                        row={**r,**dict(zip(FEATURES,map(float,vector))),**qc,
                             'data_sha256':hashlib.sha256(b).hexdigest(),
                             'zip_crc':di.CRC,'compressed_bytes':di.compress_size}
                        atomic_json(cache/'rows'/f"{r['key']}.json",row)
                        del cube,b,preview
                        completed+=1; print(f"Saved {r['fruit']} {r['specimen']} {r['side']} ({completed}/{len(pending)})",flush=True)
                    except Exception as e:
                        failures.append(dict(key=r['key'],data=r['data'],error=repr(e)))
                        print('FAILED',r['data'],repr(e),flush=True)
        except Exception as e:
            failures.append(dict(archive=archive,error=repr(e)))
            print('ARCHIVE FAILED',archive,repr(e),flush=True)
    atomic_json(cache/'last_batch.json',dict(attempted=len(pending),completed=completed,failures=failures))
    return completed


def load_features(plan,cache):
    rows=[]
    for r in plan.to_dict('records'):
        p=Path(cache)/'rows'/f"{r['key']}.json"
        if p.exists():
            row=json.loads(p.read_text())
            # Split belongs to this plan, not a cached row from another mode.
            assert row['specimen']==r['specimen'] and row['stage']==r['stage']
            row['split']=r['split']; rows.append(row)
    return pd.DataFrame(rows)


def train_and_report(plan,features,output,mode):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
    import joblib
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if features.empty or set(plan.key)!=set(features.key):
        raise ValueError('Finish every planned recording before fitting/reporting. Rerun batches; failures are not silently dropped.')
    assert (features.groupby('specimen').split.nunique()==1).all()
    assert set(features[features.split=='train'].specimen).isdisjoint(features[features.split=='test'].specimen)
    assert np.isfinite(features[FEATURES].to_numpy()).all()
    models={}; summaries=[]; stage_rows=[]; predictions=[]
    for fruit in FRUITS.values():
        f=features[features.fruit==fruit]; tr=f[f.split=='train']; te=f[f.split=='test']
        train_counts=tr.groupby('stage').specimen.nunique(); test_counts=te.groupby('stage').specimen.nunique()
        if tr.stage.nunique()<2 or te.empty:
            summaries.append(dict(fruit=fruit,status='not selected in this plan' if f.empty else 'insufficient training classes or held-out specimens'))
            for stage in STAGES:
                stage_rows.append(dict(fruit=fruit,stage=stage,train_specimens=int(train_counts.get(stage,0)),
                    test_specimens=int(test_counts.get(stage,0)),status='not evaluable in this plan',recall=None))
            continue
        model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))
        # Each physical specimen gets total training weight 1, irrespective of views/days.
        weights=1/tr.specimen.map(tr.groupby('specimen').size()).to_numpy()
        model.fit(tr[FEATURES],tr.stage,logisticregression__sample_weight=weights)
        models[fruit]=model
        prob=model.predict_proba(te[FEATURES]); classes=model.classes_
        rec=te[['key','fruit','specimen','day','side','stage']].copy()
        rec['predicted']=classes[prob.argmax(axis=1)]
        rec['max_probability']=prob.max(axis=1)
        for j,label in enumerate(classes): rec['p_'+label]=prob[:,j]
        predictions.append(rec)
        # Aggregate all held-out views to one prediction per physical fruit.
        specimen_prob=rec.groupby('specimen')[['p_'+s for s in classes]].mean()
        true=rec.groupby('specimen').stage.first().loc[specimen_prob.index]
        pred=classes[specimen_prob.to_numpy().argmax(axis=1)]
        specimen_out=pd.DataFrame(dict(specimen=specimen_prob.index,stage=true.to_numpy(),predicted=pred))
        specimen_out.to_csv(output/f'{fruit}_heldout_specimens.csv',index=False)
        for stage in STAGES:
            nt,nv=int(train_counts.get(stage,0)),int(test_counts.get(stage,0))
            enough=nt>=2 and nv>=2
            status='evaluated (small sample)' if enough else 'too few independent specimens (need >=2 train and >=2 test)'
            selected=true.to_numpy()==stage
            stage_rows.append(dict(fruit=fruit,stage=stage,train_specimens=nt,test_specimens=nv,status=status,
                recall=float(np.mean(pred[selected]==stage)) if enough else None))
        summaries.append(dict(fruit=fruit,status='SMOKE ONLY; not a performance estimate' if mode=='smoke' else 'held-out specimen evaluation',
            train_specimens=int(tr.specimen.nunique()),test_specimens=int(te.specimen.nunique()),
            specimen_accuracy=float(accuracy_score(true,pred)),specimen_balanced_accuracy=float(balanced_accuracy_score(true,pred)),
            recording_accuracy=float(accuracy_score(rec.stage,rec.predicted)),
            confusion_matrix=confusion_matrix(true,pred,labels=STAGES).tolist(),confusion_order=STAGES))
    bundle=dict(models=models,feature_order=FEATURES,algorithm=ALGORITHM,mode=mode,stages=STAGES,sensor_ready=False)
    joblib.dump(bundle,output/'deephs_ripeness.joblib')
    atomic_json(output/'feature_order.json',dict(feature_order=FEATURES,centers_nm=CENTERS.tolist(),fwhm_nm=FWHM.tolist(),sensor_ready=False))
    atomic_json(output/'heldout_results.json',summaries)
    pd.DataFrame(stage_rows).to_csv(output/'stage_support.csv',index=False)
    predframe=pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()
    predframe.to_csv(output/'heldout_recordings.csv',index=False)
    features.to_csv(output/'features_and_split.csv',index=False)
    return bundle,pd.DataFrame(summaries),pd.DataFrame(stage_rows),predframe
