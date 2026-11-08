"""Verify the exported multi-fruit model with the app's actual photo decoder."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

from ..camera_multi import MultiFruitPredictor
from ..camera_only import ROOT,decode_photo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",default=str(ROOT/"models/camera_quality"))
    a=parser.parse_args();directory=Path(a.model_dir)
    predictor=MultiFruitPredictor(directory)
    manifest=json.loads((directory/"training_manifest.json").read_text())
    rows=[]
    for record in manifest:
        if record['split']!='test':continue
        image=decode_photo((ROOT/record['path']).read_bytes())
        t0=time.perf_counter()
        logits=predictor.session.run(None,{predictor.input:predictor.preprocess(image)})[0][0]
        p=np.exp(logits-logits.max());p/=p.sum()
        pred=predictor.classes[int(p.argmax())]
        rows.append({'path':record['path'],'actual':record['label'],'predicted':pred,
                     'score':float(p.max()),'inference_ms':round((time.perf_counter()-t0)*1000,1)})
    true=[r['actual'] for r in rows];pred=[r['predicted'] for r in rows]
    fruit_true=[s.split('_',1)[1] for s in true];fruit_pred=[s.split('_',1)[1] for s in pred]
    metrics={'n':len(rows),'joint_accuracy':float(accuracy_score(true,pred)),
             'joint_macro_f1':float(f1_score(true,pred,average='macro')),
             'fruit_accuracy':float(accuracy_score(fruit_true,fruit_pred)),
             'quality_accuracy':float(accuracy_score([s.split('_')[0] for s in true],
                                                    [s.split('_')[0] for s in pred])),
             'classes':predictor.classes,'confusion':confusion_matrix(true,pred,labels=predictor.classes).tolist(),
             'median_inference_ms':statistics.median(r['inference_ms'] for r in rows),
             'model_sha256':hashlib.sha256((directory/'image_model.onnx').read_bytes()).hexdigest()}
    with (directory/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    artifact=predictor.metrics;artifact['onnx_test']=metrics
    (directory/'image_model_metrics.json').write_text(json.dumps(artifact,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k!='confusion'},indent=2))


if __name__=='__main__':main()
