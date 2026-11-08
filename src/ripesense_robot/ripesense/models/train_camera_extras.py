"""Train avocado stages and tomato visual quality on fixed ImageNet features."""
from collections import Counter
import csv
import json
import random
import shutil

import numpy as np
import onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.preprocessing import StandardScaler

from ..camera_multi import MultiFruitPredictor
from ..camera_only import ROOT,decode_photo


def main():
    directory=ROOT/'models/camera_extras';directory.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'models/camera_quality/features.onnx',directory/'features.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(directory/'features.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    all_metrics={}
    for item in ['tomato','avocado']:
        manifest=json.loads((ROOT/f'datasets/camera_extras/{item}_manifest.json').read_text())
        # Cap training copies per class; retain all validation and test rows.
        random.Random(42).shuffle(manifest);counts=Counter();records=[]
        for row in manifest:
            key=row['split'],row['label']
            if row['split']=='train' and counts[key]>=500:continue
            counts[key]+=1;records.append(row)
        groups={s:{r['group'] for r in records if r['split']==s} for s in ['train','val','test']}
        assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
        classes=sorted({r['label'] for r in records})
        arrays=[]
        print(item,'rows',dict(counts),'groups',{s:len(v) for s,v in groups.items()},flush=True)
        for start in range(0,len(records),16):
            images=[MultiFruitPredictor.preprocess(decode_photo((ROOT/r['path']).read_bytes()))[0]
                    for r in records[start:start+16]]
            arrays.append(session.run(None,{'image':np.stack(images)})[0])
            if start%512==0:print(item,'features',start,'/',len(records),flush=True)
        X=np.concatenate(arrays);y=np.array([classes.index(r['label']) for r in records])
        masks={s:np.array([r['split']==s for r in records]) for s in ['train','val','test']}
        scaler=StandardScaler().fit(X[masks['train']]);Z=scaler.transform(X)
        best,chosen,selected=-1,None,None
        validations=[]
        for c in [.003,.03,.3]:
            model=LogisticRegression(C=c,class_weight='balanced',max_iter=500,tol=.001,random_state=42)
            model.fit(Z[masks['train']],y[masks['train']])
            score=float(f1_score(y[masks['val']],model.predict(Z[masks['val']]),average='macro'))
            validations.append({'C':c,'macro_f1':score})
            if score>best:best,chosen,selected=score,c,model
        def evaluate(split):
            mask=masks[split];true=y[mask];pred=selected.predict(Z[mask])
            return {'n':int(mask.sum()),'accuracy':float(accuracy_score(true,pred)),
                    'macro_f1':float(f1_score(true,pred,average='macro')),
                    'confusion':confusion_matrix(true,pred,labels=list(range(len(classes)))).tolist()}
        weights=selected.coef_/scaler.scale_;bias=selected.intercept_-weights@scaler.mean_
        if len(classes)==2:
            weights=np.vstack([np.zeros_like(weights),weights]);bias=np.concatenate([[0.],bias])
        np.savez_compressed(directory/f'{item}.npz',weights=weights.astype(np.float32),bias=bias.astype(np.float32))
        metrics={'classes':classes,'val':evaluate('val'),'test':evaluate('test'),'chosen_C':chosen,
                 'validation':validations,'counts':{f'{s}/{label}':n for (s,label),n in counts.items()},
                 'groups':{s:len(v) for s,v in groups.items()},
                 'split':'Whole avocado identities held out' if item=='avocado' else 'Source photo groups held out; original author splits retained',
                 'scope':'Hass avocado cultivar in the source capture setup' if item=='avocado' else 'Tomato visible fresh / spoiled labels',
                 'source':records[0]['source'],'license':'CC BY 4.0'}
        prediction=selected.predict(Z[masks['test']]);test_records=[r for r in records if r['split']=='test']
        with (directory/f'{item}_test_predictions.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['path','group','actual','predicted']);writer.writeheader()
            for row,pred in zip(test_records,prediction):
                writer.writerow({'path':row['path'],'group':row['group'],'actual':row['label'],'predicted':classes[pred]})
        all_metrics[item]=metrics
        (directory/f'{item}_metrics.json').write_text(json.dumps(metrics,indent=2))
        print(item,'TEST',json.dumps(metrics['test']),flush=True)
    (directory/'models.json').write_text(json.dumps(all_metrics,indent=2))


if __name__=='__main__':main()
