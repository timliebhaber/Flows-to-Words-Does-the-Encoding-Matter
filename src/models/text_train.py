"""Shared training loop for text classifiers (small BERT now, DistilBERT later).

Train for `epochs`, evaluate val macro-F1 after each epoch, keep the weights of the best epoch.
Only train and val are ever passed in here; the test split is used by the caller after training.
"""

import copy
import math
import random
import time

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.eval.metrics import compute_metrics


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def tokenize(tok, texts):
    return [e.ids for e in tok.encode_batch(texts)]


def make_loader(ids, labels, batch_size, shuffle, seed, pad_id):
    def collate(batch):
        seqs, ys = zip(*batch)
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return input_ids, (input_ids != pad_id).long(), torch.tensor(ys, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(list(zip(ids, labels)), batch_size=batch_size, shuffle=shuffle, generator=gen,
                      collate_fn=collate)


def autocast(device, amp_dtype):
    return torch.autocast(device_type=torch.device(device).type, dtype=getattr(torch, amp_dtype))


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    preds = []
    for input_ids, mask, _ in loader:
        with autocast(device, amp_dtype):
            logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def train_and_select(model, train_ids, y_train, val_ids, y_val, names, benign, lr, seed, tcfg, pad_id):
    """Returns (best_state_dict, curve, best_epoch, train_seconds). Model ends with the best weights loaded."""
    device = tcfg["device"]
    model.to(device)
    train_loader = make_loader(train_ids, y_train, tcfg["batch_size"], True, seed, pad_id)
    val_loader = make_loader(val_ids, y_val, tcfg["eval_batch_size"], False, seed, pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tcfg["weight_decay"])
    accum = tcfg["grad_accum_steps"]  # optimizer step every `accum` batches
    total = tcfg["epochs"] * math.ceil(len(train_loader) / accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total), total)
    weight = None
    if tcfg["class_weight"] == "balanced":
        present = np.unique(y_train)
        w = np.ones(len(names))
        w[present] = compute_class_weight("balanced", classes=present, y=y_train)
        weight = torch.tensor(w, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    best_f1, best_state, best_epoch, curve = -1.0, None, None, []
    t0 = time.perf_counter()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        losses = []
        for step, (input_ids, mask, ys) in enumerate(train_loader, 1):
            with autocast(device, tcfg["amp_dtype"]):
                logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
            loss = loss_fn(logits.float(), ys.to(device))
            (loss / accum).backward()
            losses.append(loss.item())
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        f1 = compute_metrics(y_val, predict(model, val_loader, device, tcfg["amp_dtype"]), names, benign)["macro_f1"]
        curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "val_macro_f1": round(f1, 5)})
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return best_state, curve, best_epoch, time.perf_counter() - t0
