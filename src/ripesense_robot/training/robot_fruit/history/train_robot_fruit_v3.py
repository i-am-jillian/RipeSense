"""Train the RipeSense robot fruit model: public data + our AiNex robot-camera photos.

One MobileNetV3-Small, 14 classes (fruit_state). Robot photos are split by TIME inside each
collection sweep (the last 25 % of each sweep = other head angles -> val/test), so the robot
score is measured on views the model never trained on.

  cd ~/UB_Hack/ripesense && .venv/bin/python robot/train_robot_fruit.py
Outputs -> /mnt/data/ripesense_train/robot_fruit/{robot_fruit_model.onnx, _classes.json, _metrics.json, report.png}
"""
import json, os, random, shutil, sys, time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/ripesense_train/robot_fruit_v3'); OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'data'
EPOCHS = int(os.environ.get('EPOCHS', 12)); BATCH = 64; LR = 1e-3; CAP = 650; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count())

# ---------------- dataset ----------------
PUBLIC = {   # fruit_quality class -> our class
    'fresh_apple': 'apple_fresh', 'rotten_apple': 'apple_rotten', 'rotten_banana': 'banana_rotten',
    'fresh_orange': 'orange_fresh', 'rotten_orange': 'orange_rotten',
    'fresh_strawberry': 'strawberry_fresh', 'rotten_strawberry': 'strawberry_rotten',
    **{f'{q}_{f}': 'other_fruit' for q in ('fresh', 'rotten') for f in ('grape', 'guava', 'jujube', 'lime', 'pomegranate')},
}
STAGES = {'unripe': 'banana_unripe', 'ripe': 'banana_ripe', 'overripe': 'banana_overripe'}
ROBOT = {'apple': 'apple_fresh', 'strawberry': 'strawberry_fresh', 'banana': 'banana_unripe', 'orange': 'orange_fresh',
         'orange_rotten': 'orange_rotten', 'avocado': 'avocado_ripe', 'mango': 'mango_fresh', 'not_fruit': 'not_fruit'}
RS_DIR = ROOT / 'xarm/rs_collect'          # xArm RealSense crops, folder name = class

def link(src, split, cls, name):
    d = DATA / split / cls; d.mkdir(parents=True, exist_ok=True)
    dst = d / name
    if not dst.exists(): os.symlink(os.path.abspath(src), dst)

shutil.rmtree(DATA, ignore_errors=True)
counts = Counter()
for split in ('train', 'val', 'test'):
    for src_cls, cls in PUBLIC.items():
        fs = sorted((ROOT / 'datasets/fruit_quality' / split / src_cls).glob('*'))
        random.Random(src_cls).shuffle(fs)
        cap = CAP if cls != 'other_fruit' else CAP // 10          # other_fruit: 10 source classes share the cap
        for f in fs[:cap if split == 'train' else cap // 4]:
            link(f, split, cls, 'public__' + f.name); counts[('public', split, cls)] += 1
    for st, cls in STAGES.items():
        for f in sorted((ROOT / 'datasets/banana_stages' / split / st).glob('*')):
            link(f, split, cls, 'public__' + f.name); counts[('public', split, cls)] += 1
for src_cls, cls in ROBOT.items():
    fs = sorted((ROOT / 'robot/own_data' / src_cls).glob('*.jpg'))
    n = len(fs)
    for i, f in enumerate(fs):                       # time order: first 75 % train, next 10 % val, last 15 % test
        split = 'train' if i < 0.75 * n else ('val' if i < 0.85 * n else 'test')
        link(f, split, cls, 'robot__' + src_cls + '__' + f.name); counts[('robot', split, cls)] += 1

for cdir in sorted(RS_DIR.iterdir()) if RS_DIR.exists() else []:
    fs = sorted(cdir.glob('*.jpg'))           # names start with the session time -> time order
    n = len(fs)
    for i, f in enumerate(fs):
        split = 'train' if i < 0.75 * n else ('val' if i < 0.85 * n else 'test')
        link(f, split, cdir.name, 'rs__' + cdir.name + '__' + f.name); counts[('realsense', split, cdir.name)] += 1
CLASSES = sorted({c for (_, _, c) in counts})
print('classes:', CLASSES)
for src in ('public', 'robot', 'realsense'):
    for split in ('train', 'val', 'test'):
        row = {c: counts[(src, split, c)] for c in CLASSES if counts[(src, split, c)]}
        print(f'{src:6s} {split:5s}', row)
for split in ('train', 'val', 'test'):
    for c in CLASSES:
        (DATA / split / c).mkdir(parents=True, exist_ok=True)

# ---------------- training (same recipe as notebooks/train_banana_model_colab.ipynb) ----------------
class BackgroundSwap:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img):
        if random.random() > self.p: return img
        a = np.asarray(img.convert('RGB')).astype(np.int16)
        mx, mn = a.max(2), a.min(2)
        bg = ((mx - mn) / np.maximum(mx, 1) < 0.18) & (mx > 140)
        if bg.mean() < 0.08: return img
        v, t = random.randint(8, 200), random.randint(-6, 6)
        a = np.where(bg[..., None], np.clip(np.array((v + t, v, v - t), np.int16) + np.random.randint(-12, 13, a.shape, dtype=np.int16), 0, 255), a)
        return Image.fromarray(a.astype(np.uint8))

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = transforms.Compose([BackgroundSwap(0.5), transforms.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.6, 1.6)),
                               transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(25),
                               transforms.ColorJitter(0.4, 0.35, 0.25, 0.02),
                               transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.6))], p=0.3),
                               transforms.ToTensor(), norm, transforms.RandomErasing(p=0.2, scale=(0.02, 0.12))])
tf_eval = transforms.Compose([transforms.Resize(240), transforms.CenterCrop(224), transforms.ToTensor(), norm])
tr = datasets.ImageFolder(DATA / 'train', tf_train); va = datasets.ImageFolder(DATA / 'val', tf_eval); te = datasets.ImageFolder(DATA / 'test', tf_eval)
assert tr.classes == va.classes == te.classes == CLASSES
cnt = np.bincount(tr.targets, minlength=len(CLASSES))
# robot photos weigh 3x in sampling: they are the domain the robot actually sees
w = (1.0 / cnt)[tr.targets] * np.array([5.0 if '/rs__' in s else 3.0 if '/robot__' in s else 1.0 for s, _ in tr.samples])
tl = DataLoader(tr, batch_size=BATCH, sampler=WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), True), num_workers=6)
net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(tl))
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

def predict(ds):
    net.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=128, num_workers=6):
            ps.append(net(x).softmax(1).numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)
def scores(y, p):
    cm = np.zeros((len(CLASSES),) * 2, int)
    for t, q in zip(y, p.argmax(1)): cm[t, q] += 1
    rec = cm.diagonal() / np.maximum(1, cm.sum(1)); prec = cm.diagonal() / np.maximum(1, cm.sum(0))
    f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(1e-9, rec + prec), 0); present = cm.sum(1) > 0
    return {'n': int(cm.sum()), 'accuracy': float(np.trace(cm) / max(1, cm.sum())),
            'macro_f1': float(f1[present].mean()) if present.any() else 0.0, 'confusion': cm.tolist(),
            'per_class_recall': {c: round(float(r), 3) for c, r, pr in zip(CLASSES, rec, present) if pr}}
def by_source(ds, y, p):
    src = np.array(['realsense' if '/rs__' in s else 'robot' if '/robot__' in s else 'public' for s, _ in ds.samples])
    return {s: scores(y[src == s], p[src == s]) for s in ('public', 'robot', 'realsense') if (src == s).any()}

print(f'train {len(tr)}  val {len(va)}  test {len(te)}  epochs {EPOCHS}', flush=True)
best, state = -1, None
for ep in range(EPOCHS):
    net.train(); t0 = time.time(); tot = n = 0
    for x, y in tl:
        opt.zero_grad(); loss = lossf(net(x), y); loss.backward(); opt.step(); sched.step()
        tot += loss.item() * len(y); n += len(y)
    yv, pv = predict(va); bs = by_source(va, yv, pv)
    score = 0.25 * bs['public']['macro_f1'] + 0.15 * bs['robot']['macro_f1'] + 0.6 * bs['realsense']['macro_f1']   # RealSense matters most
    print(f"epoch {ep + 1:2d}/{EPOCHS} loss {tot / n:.3f}  val public F1 {bs['public']['macro_f1']:.3f}  "
          f"val robot F1 {bs['robot']['macro_f1']:.3f}  val REALSENSE F1 {bs['realsense']['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if score >= best: best, state = score, {k: v.clone() for k, v in net.state_dict().items()}
net.load_state_dict(state)

yt, pt = predict(te)
metrics = {'classes': CLASSES, 'epochs': EPOCHS, 'train_counts': dict(zip(CLASSES, cnt.tolist())),
           'test': scores(yt, pt), 'test_by_source': by_source(te, yt, pt)}
for name, m in [('ALL', metrics['test'])] + [(k.upper(), v) for k, v in metrics['test_by_source'].items()]:
    print(f"\nTEST {name}: n={m['n']} accuracy {m['accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}")
    print('  recall per class:', m['per_class_recall'])
print('\nREALSENSE test confusion (rows true, cols pred):')
cm = np.array(metrics['test_by_source']['realsense']['confusion'])
for c, row in zip(CLASSES, cm):
    if row.sum(): print(f'  {c:18s}', {CLASSES[j]: int(v) for j, v in enumerate(row) if v})

net.eval()
onnx = OUT / 'robot_fruit_model.onnx'   # v2
torch.onnx.export(net, torch.randn(1, 3, 224, 224), str(onnx), input_names=['input'], output_names=['logits'],
                  opset_version=17, dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}, dynamo=False)
(OUT / 'robot_fruit_model_classes.json').write_text(json.dumps(CLASSES))
(OUT / 'robot_fruit_model_metrics.json').write_text(json.dumps(metrics, indent=1))
# robot runs OpenCV DNN with this exact preprocessing: check it matches PyTorch
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
def prep(bgr):
    h, w = bgr.shape[:2]; s = 240.0 / min(h, w)
    img = cv2.resize(bgr, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]; y0, x0 = (h - 224) // 2, (w - 224) // 2
    x = cv2.cvtColor(img[y0:y0 + 224, x0:x0 + 224], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
cvnet = cv2.dnn.readNetFromONNX(str(onnx)); agree = 0; idx = random.sample(range(len(te)), min(200, len(te)))
for i in idx:
    x = prep(cv2.imread(te.samples[i][0])); cvnet.setInput(x)
    with torch.no_grad(): agree += int(cvnet.forward()[0].argmax() == net(torch.from_numpy(x))[0].argmax())
print(f'\nOpenCV-DNN vs PyTorch: same class {agree}/{len(idx)}')
print('saved', onnx)
