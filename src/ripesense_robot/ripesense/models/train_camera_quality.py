"""Efficient multi-fruit classifier: pretrained image features + trained linear head.

Keeps validation for regularization choice and test for the final evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np


def main():
    import torch
    from torch import nn
    from torchvision import models,transforms
    from PIL import Image
    import onnxruntime as ort
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="datasets/fruit_quality")
    parser.add_argument("--out",default="models/camera_quality")
    parser.add_argument("--max-per-class",type=int,default=600)
    a = parser.parse_args()
    torch.set_num_threads(2)
    root,out = Path(a.data),Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    classes = sorted(p.name for p in (root/"train").iterdir() if p.is_dir())
    seen,records = {},[]
    duplicates = Counter()
    rng = random.Random(42)
    for split in ["test","val","train"]:
        for label in classes:
            paths = sorted((root/split/label).glob("*.jpg"))
            rng.shuffle(paths)
            n = 0
            for path in paths:
                if split == "train" and n>=a.max_per_class:
                    break
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    duplicates[split]+=1
                    continue
                seen[digest] = split
                records.append({"path":str(path),"split":split,"label":label,"sha256":digest})
                n+=1
    (out/"training_manifest.json").write_text(json.dumps(records,indent=1))
    print("images",dict(Counter(r["split"] for r in records)),"duplicates removed",dict(duplicates),flush=True)
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Identity()
    net.eval()
    torch.onnx.export(net,torch.zeros(1,3,224,224),str(out/"features.onnx"),
                      input_names=["image"],output_names=["features"],opset_version=17,
                      dynamic_axes={"image":{0:"batch"},"features":{0:"batch"}},dynamo=False)
    options = ort.SessionOptions(); options.intra_op_num_threads=2
    session = ort.InferenceSession(str(out/"features.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    transform = transforms.Compose([transforms.Resize(240),transforms.CenterCrop(224),transforms.ToTensor(),
                                    transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    X = []
    t0 = time.monotonic()
    for start in range(0,len(records),32):
        arrays = []
        for row in records[start:start+32]:
            with Image.open(row["path"]) as image:
                arrays.append(transform(image.convert("RGB")).numpy())
        X.append(session.run(None,{"image":np.stack(arrays)})[0])
        if start%1024==0:
            print("features",start,"/",len(records),"seconds",round(time.monotonic()-t0),flush=True)
    X = np.concatenate(X)
    y = np.array([classes.index(r["label"]) for r in records])
    masks = {s:np.array([r["split"]==s for r in records]) for s in ["train","val","test"]}
    np.savez_compressed(out/"features.npz",X=X,y=y,**masks)
    scaler = StandardScaler().fit(X[masks["train"]])
    Z = scaler.transform(X)
    best,estimator,chosen = -1,None,None
    candidates=[]
    for c in [.03,.3,3.]:
        model = LogisticRegression(C=c,class_weight="balanced",max_iter=300,tol=.001,random_state=42)
        model.fit(Z[masks["train"]],y[masks["train"]])
        pred = model.predict(Z[masks["val"]])
        score = float(f1_score(y[masks["val"]],pred,average="macro"))
        candidates.append({"C":c,"val_macro_f1":score})
        print("validation",c,score,flush=True)
        if score>best:
            best,estimator,chosen=score,model,c
    def evaluate(split):
        mask=masks[split];true=y[mask]; pred=estimator.predict(Z[mask])
        return {"n":int(mask.sum()),"accuracy":float(accuracy_score(true,pred)),
                "macro_f1":float(f1_score(true,pred,average="macro")),
                "confusion":confusion_matrix(true,pred,labels=list(range(len(classes)))).tolist()}
    head = nn.Linear(1024,len(classes))
    weights = estimator.coef_/scaler.scale_
    biases = estimator.intercept_-weights@scaler.mean_
    head.weight.data.copy_(torch.tensor(weights,dtype=torch.float32))
    head.bias.data.copy_(torch.tensor(biases,dtype=torch.float32))
    full = nn.Sequential(net,head).eval()
    torch.onnx.export(full,torch.zeros(1,3,224,224),str(out/"image_model.onnx"),
                      input_names=["image"],output_names=["logits"],opset_version=17,dynamo=False)
    metrics={"classes":classes,"val":evaluate("val"),"test":evaluate("test"),"chosen_C":chosen,
             "validation_candidates":candidates,"method":"Frozen ImageNet MobileNetV3-Small + standardized multinomial logistic head",
             "train_counts":dict(Counter(r["label"] for r in records if r["split"]=="train")),
             "duplicates_removed":dict(duplicates),"source":"Fresh and Rotten Fruits + FruitNet",
             "split_limit":"Filename-hash split; related photographs may share fruit identity across splits",
             "seed":42}
    (out/"image_model_classes.json").write_text(json.dumps(classes))
    (out/"image_model_metrics.json").write_text(json.dumps(metrics,indent=2))
    print("FINAL TEST",json.dumps(metrics["test"]),flush=True)


if __name__ == "__main__":
    main()
