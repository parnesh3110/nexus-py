"""Train the gesture model and compare it to the rule-based baseline on the same held-out data.

    python -m nexus.train

Test data: with 2+ recorded sessions, the newest session is held out entirely (a new sitting,
lighting, hand position), which is the honest number. With one session, a stratified 20%
split is used and the report says so.
False activations: both detectors run over the free-movement streams (never used in training)
with the same event rules as the live app, reported per minute.
Writes models/gesture.pt and results/gesture-<time>.json / .md
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .gesture_features import CLASSES, WINDOW, clip_features, rule_baseline
from .gesture_model import GestureNet, save


def load_sessions(folder: str):
    files = sorted(Path(folder).glob("*.npz"))
    if not files:
        raise SystemExit(f"No recordings in {folder}. Run: python -m nexus.collect")
    return [(f.stem, np.load(f, allow_pickle=False)) for f in files]


def featurize(img: np.ndarray, world: np.ndarray) -> np.ndarray:
    return np.stack([clip_features(i, w) for i, w in zip(img, world)])


def stratified_split(y: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    test = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        test.extend(idx[: max(1, int(round(len(idx) * frac)))])
    mask = np.zeros(len(y), bool)
    mask[test] = True
    return ~mask, mask


def augment(x: np.ndarray, rng) -> np.ndarray:
    """Noise, small time shifts and path scaling: the model shouldn't depend on exact timing or size."""
    x = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
    shift = rng.integers(-3, 4)
    if shift:
        x = np.roll(x, shift, axis=1)
        if shift > 0:
            x[:, :shift] = x[:, shift:shift + 1]
        else:
            x[:, shift:] = x[:, shift - 1:shift]
    x[:, :, 63:67] *= rng.uniform(0.85, 1.15, (x.shape[0], 1, 1)).astype(np.float32)
    return x


def metrics(y_true, y_pred) -> dict:
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per = {}
    for k, c in enumerate(CLASSES):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k, :].sum() - cm[k, k]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(float(prec), 3), "recall": round(float(rec), 3),
                  "f1": round(float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0, 3), "n": int(cm[k].sum())}
    return {"accuracy": round(float(np.trace(cm) / max(cm.sum(), 1)), 4),
            "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
            "per_class": per, "confusion": cm.tolist()}


def events_per_minute(sessions, predict_window, threshold=0.85, confirm=3, cooldown=0.9) -> tuple[float, float, dict]:
    """Slide over free-movement streams frame by frame, with the live app's event rules."""
    events, seconds, kinds = 0, 0.0, {}
    for _, d in sessions:
        if "free_t" not in d or len(d["free_t"]) < WINDOW:
            continue
        img, world, valid, ts = d["free_img"], d["free_world"], d["free_valid"], d["free_t"]
        seconds += float(ts[-1] - ts[0])
        buf, streak, last_event = [], ("idle", 0), -1e9
        for k in range(len(ts)):
            if not valid[k]:
                buf, streak = [], ("idle", 0)
                continue
            buf.append(k)
            if len(buf) < WINDOW:
                continue
            idx = buf[-WINDOW:]
            label, prob = predict_window(img[idx], world[idx])
            if label == "idle" or prob < threshold:
                streak = ("idle", 0)
                continue
            n = streak[1] + 1 if streak[0] == label else 1
            streak = (label, n)
            if n >= confirm and ts[k] - last_event > cooldown:
                events += 1
                kinds[label] = kinds.get(label, 0) + 1
                last_event, streak, buf = ts[k], ("idle", 0), []
    minutes = seconds / 60
    return (events / minutes if minutes else float("nan")), minutes, kinds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nexus.train")
    ap.add_argument("--data", default="data/gestures")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/gesture.pt")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sessions = load_sessions(args.data)
    X_all, y_all, sess_all, raw = [], [], [], []
    for si, (name, d) in enumerate(sessions):
        X_all.append(featurize(d["img"], d["world"]))
        y_all.append(d["labels"])
        sess_all.append(np.full(len(d["labels"]), si))
        raw.extend(zip(d["img"], d["world"]))
    X, y, sess = np.concatenate(X_all), np.concatenate(y_all), np.concatenate(sess_all)

    if len(sessions) >= 2:
        test_mask = sess == len(sessions) - 1
        split = f"held-out session {sessions[-1][0]}"
    else:
        _, test_mask = stratified_split(y, 0.2, args.seed)
        split = "random stratified 20% (record a second session for a held-out-session test)"
    train_idx = np.flatnonzero(~test_mask)
    tr_mask, va_mask = stratified_split(y[train_idx], 0.15, args.seed + 1)
    tr, va, te = train_idx[tr_mask], train_idx[va_mask], np.flatnonzero(test_mask)
    print(f"clips: train {len(tr)}, val {len(va)}, test {len(te)}  ({split})")

    mean = X[tr].reshape(-1, X.shape[-1]).mean(0)
    std = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    norm = lambda a: ((a - mean) / std).astype(np.float32)

    model = GestureNet()
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    counts = np.bincount(y[tr], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(CLASSES), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    Xva, yva = torch.from_numpy(norm(X[va])), torch.from_numpy(y[va])
    best, best_state = -1.0, None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), 32):
            idx = order[b:b + 32]
            xb = torch.from_numpy(augment(norm(X[idx]), rng))
            loss = loss_fn(model(xb), torch.from_numpy(y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item() if len(va) else 0.0
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")
    model.load_state_dict(best_state)
    model.eval()
    train_s = time.perf_counter() - start

    # --- test: model vs rules, same clips ---
    with torch.no_grad():
        pred_model = model(torch.from_numpy(norm(X[te]))).argmax(1).numpy()
    pred_rules = np.array([CLASSES.index(rule_baseline(*raw[i])) for i in te])
    m_model, m_rules = metrics(y[te], pred_model), metrics(y[te], pred_rules)

    # --- false activations on free movement, same event rules for both ---
    def model_window(img, world):
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(norm(clip_features(img, world))[None])), 1)[0].numpy()
        return CLASSES[int(p.argmax())], float(p.max())
    fa_model, minutes, kinds_model = events_per_minute(sessions, model_window)
    fa_rules, _, kinds_rules = events_per_minute(sessions, lambda i, w: (rule_baseline(i, w), 1.0))

    # --- inference speed ---
    torch.set_num_threads(1)
    one = torch.from_numpy(norm(X[te[:1]] if len(te) else X[:1]))
    with torch.no_grad():
        for _ in range(20):
            model(one)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            model(one)
            times.append((time.perf_counter() - t0) * 1000)
    infer_ms = float(np.median(times))

    info = {"test_accuracy": m_model["accuracy"], "split": split, "trained": datetime.now().isoformat(timespec="seconds")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save(args.out, model, mean, std, info)

    report = {
        "split": split, "sessions": [n for n, _ in sessions], "clips": {"train": len(tr), "val": len(va), "test": len(te)},
        "model": {"params": params, "train_seconds": round(train_s, 1), "inference_ms_p50": round(infer_ms, 3), **m_model,
                  "false_activations_per_min": round(fa_model, 2), "false_activation_kinds": kinds_model},
        "rules_baseline": {**m_rules, "false_activations_per_min": round(fa_rules, 2), "false_activation_kinds": kinds_rules},
        "free_movement_minutes": round(minutes, 2), "classes": CLASSES,
    }
    Path("results").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    Path(f"results/gesture-{stamp}.json").write_text(json.dumps(report, indent=2))
    md = [f"# Gesture model vs. rules ({stamp})", "", f"Test: {split}, {len(te)} clips. "
          f"False activations: {minutes:.1f} min of free movement.", "",
          "| | Model (PyTorch) | Rules |", "|---|---|---|",
          f"| Accuracy | {m_model['accuracy']:.1%} | {m_rules['accuracy']:.1%} |",
          f"| Macro F1 | {m_model['macro_f1']:.3f} | {m_rules['macro_f1']:.3f} |",
          f"| False activations / min | {fa_model:.2f} | {fa_rules:.2f} |",
          f"| Inference | {infer_ms:.2f} ms, {params:,} params | - |", "",
          "| Gesture | Model P / R | Rules P / R |", "|---|---|---|"]
    for c in CLASSES:
        a, b = m_model["per_class"][c], m_rules["per_class"][c]
        md.append(f"| {c} | {a['precision']:.2f} / {a['recall']:.2f} | {b['precision']:.2f} / {b['recall']:.2f} |")
    Path(f"results/gesture-{stamp}.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nSaved model to {args.out} and results to results/gesture-{stamp}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
