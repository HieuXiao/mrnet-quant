"""Train one plane in FP32 and keep the best checkpoint on the internal validation part.

Usage: python -m mrnetq.train --plane sagittal --seed 0 [--backbone mobilenet_v3_small]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .common import INPUT_DIM, MAX_PIXEL, MEAN, ROOT, STD, env_info, load_volume, preprocess, read_labels, split_cases
from .model import MRNetPlane


class Volumes(Dataset):
    def __init__(self, split, plane, cases, train):
        self.split, self.plane, self.cases, self.train = split, plane, cases, train
        self.y = read_labels(split)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        v = load_volume(self.split, self.plane, c)
        x = torch.from_numpy(v) if self.train else torch.from_numpy(preprocess(v))
        return x, float(self.y[c]), c


def augment(v, rng):
    """Rotation +-25 deg, translation +-25 px, horizontal flip, then crop and normalize as in `preprocess`."""
    x = v.float()[:, None]
    x = TF.affine(x, angle=rng.uniform(-25, 25), translate=[rng.uniform(-25, 25), rng.uniform(-25, 25)],
                  scale=1.0, shear=[0.0, 0.0])
    if rng.random() < 0.5:
        x = x.flip(-1)
    pad = (x.shape[-1] - INPUT_DIM) // 2
    x = x[..., pad:pad + INPUT_DIM, pad:pad + INPUT_DIM]
    lo, hi = x.min(), x.max()
    x = (x - lo) / torch.clamp(hi - lo, min=1e-6) * MAX_PIXEL
    return (x - MEAN) / STD


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    logits, ys, cases = [], [], []
    for x, y, c in loader:
        logits.append(model(x.to(dev)).item())
        ys.append(y)
        cases.append(c)
    return np.array(logits), np.array(ys), cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plane', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backbone', default='alexnet')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15, help='early stopping on internal-validation AUC')
    ap.add_argument('--sched-patience', type=int, default=10, help='ReduceLROnPlateau patience')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: use only N exams per split')
    ap.add_argument('--out', default=str(ROOT / 'runs/train'))
    a = ap.parse_args()

    out = Path(a.out) / f'{a.backbone}_{a.plane}_seed{a.seed}'
    rec_path = out / 'record.json'
    if rec_path.exists() and json.loads(rec_path.read_text()).get('status') == 'done':
        print('already done:', out)
        return
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(a.seed)

    split, itrain = split_cases('itrain')
    _, ival = split_cases('ival')
    if a.limit:
        itrain, ival = itrain[:a.limit], ival[:a.limit]
    y = read_labels(split)
    pos = sum(y[c] for c in itrain)
    pos_weight = torch.tensor([(len(itrain) - pos) / max(pos, 1)], device=dev)
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(Volumes(split, a.plane, itrain, True), batch_size=None, shuffle=True, num_workers=4, generator=g)
    vl = DataLoader(Volumes(split, a.plane, ival, False), batch_size=None, shuffle=False, num_workers=4)

    model = MRNetPlane(a.backbone).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=a.sched_patience)
    rec = {'args': vars(a), 'env': {**env_info(), 'torch': torch.__version__,
                                    'gpu': torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'},
           'n_itrain': len(itrain), 'n_ival': len(ival), 'pos_weight': pos_weight.item(), 'history': [],
           'status': 'running'}
    best, best_ep, t0 = -1.0, -1, time.time()
    for ep in range(a.epochs):
        model.train()
        tr_loss = 0.0
        for v, yy, _ in tl:
            x = augment(v.to(dev, non_blocking=True), rng)
            loss = F.binary_cross_entropy_with_logits(model(x), torch.tensor([yy], device=dev), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        lg, ys, cases = evaluate(model, vl, dev)
        auc = roc_auc_score(ys, lg) if 0 < ys.sum() < len(ys) else float('nan')
        vloss = float(F.binary_cross_entropy_with_logits(torch.tensor(lg), torch.tensor(ys)).item())
        sched.step(vloss)
        rec['history'].append({'epoch': ep, 'train_loss': tr_loss / len(itrain), 'ival_loss': vloss, 'ival_auc': auc,
                               'lr': opt.param_groups[0]['lr'], 'sec': round(time.time() - t0, 1)})
        print(json.dumps(rec['history'][-1]), flush=True)
        if not math.isnan(auc) and auc > best:
            best, best_ep = auc, ep
            torch.save(model.state_dict(), out / 'best.pt')
            (out / 'ival_logits.json').write_text(json.dumps({str(c): float(l) for c, l in zip(cases, lg)}))
        rec_path.write_text(json.dumps(rec, indent=1))
        if ep - best_ep >= a.patience:
            break
    rec.update(status='done', best_ival_auc=best, best_epoch=best_ep, train_minutes=round((time.time() - t0) / 60, 2))
    rec_path.write_text(json.dumps(rec, indent=1))
    print('done', out, 'best ival AUC', round(best, 4), 'epoch', best_ep)


if __name__ == '__main__':
    main()
