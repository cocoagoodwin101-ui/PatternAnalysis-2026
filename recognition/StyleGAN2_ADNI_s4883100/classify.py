"""
Dedicated AD/CN slice classifier, used to measure the class fidelity of generated cohorts.

Inception-v3 features barely separate AD from CN (10-NN vote: 56% balanced accuracy on real
validation slices, Section 11.6), so they cannot say reliably how AD-like a generated cohort
is. This script trains a small CNN on the real training split instead, selects it on the
validation split, reports it on the test split, and then classifies generated cohorts.

Two checks keep the result honest:
  * Anatomical-level confound (Discovery 9): AD slice windows start about 10 slices lower than
    CN ones, so a classifier could learn which level of the brain a slice shows rather than
    disease anatomy. A level-only predictor (AD share of training slices at each absolute slice
    index) is reported alongside the CNN, and the CNN is also scored within single slice indices,
    where level cannot help.
  * Classifier bias: no classifier is perfect, so a generated cohort's predicted AD share is
    compared with the share a faithful generator would receive: TPR x (training AD share) +
    FPR x (1 - training AD share), with the error rates measured on real test slices.

For class-conditional runs the requested label of every slice is known, so label fidelity
(share of AD-requested slices classified AD, and of CN-requested slices classified CN) is
reported too.

Usage (GPU node):
    python classify.py --runs sg2_adni64:10000 sg2_adni64_augp02:17500 sg2_adni64_cond:15100
    python classify.py --eval_only --runs ...   # reuse the trained classifier
Outputs: outputs/classifier/{classifier.pt, summary.json}, figures/classifier_<tag>.png
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DEFAULT_CACHE, DEFAULT_SEED, get_adni_splits
from predict import build_generator, class_labels, generate, load_run, snapshot_iter, wilson


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SliceClassifier(nn.Module):
    """Small CNN: four conv blocks (conv-BN-ReLU twice, then 2x2 max-pool), global average
    pooling, dropout and a linear AD-vs-CN logit (1.17 M parameters at width 32)."""

    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        chans = [1, width, width * 2, width * 4, width * 8]
        blocks = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
                       nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(chans[-1], 1))

    def forward(self, x):
        """x: (B, 1, R, R) in [-1, 1]. Returns AD logits of shape (B,)."""
        return self.head(self.features(x).mean(dim=(2, 3))).squeeze(1)


def augment(x, gen, max_shift=4, intensity_std=0.1):
    """Light training augmentation: random integer translation (up to max_shift pixels) and a
    random brightness/contrast change. No flips: a flipped sagittal slice swaps front and back."""
    b, _, r, _ = x.shape
    pad = F.pad(x, (max_shift,) * 4, value=-1.0)  # background is -1 after scaling
    dx = torch.randint(0, 2 * max_shift + 1, (b,), generator=gen, device=x.device)
    dy = torch.randint(0, 2 * max_shift + 1, (b,), generator=gen, device=x.device)
    rows = (dy.view(b, 1) + torch.arange(r, device=x.device).view(1, r))  # (B, R)
    cols = (dx.view(b, 1) + torch.arange(r, device=x.device).view(1, r))
    out = pad[torch.arange(b, device=x.device).view(b, 1, 1), 0, rows.view(b, r, 1), cols.view(b, 1, r)]
    out = out.unsqueeze(1)
    scale = 1 + intensity_std * torch.randn(b, 1, 1, 1, generator=gen, device=x.device)
    shift = intensity_std * torch.randn(b, 1, 1, 1, generator=gen, device=x.device)
    return ((out + 1) * scale - 1 + shift).clamp(-1, 1)


def to_input(images_uint8):
    """uint8 (N, 1, R, R) -> float in [-1, 1], as in dataset.py."""
    return images_uint8.float() / 127.5 - 1.0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def balanced_accuracy(pred, true):
    """Mean per-class recall (pred, true: 1-D 0/1 tensors)."""
    return float(sum((pred[true == c] == c).float().mean() for c in (0, 1)) / 2)


def auroc(scores, true):
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic; ties get average ranks."""
    scores, true = scores.double().cpu(), true.cpu()
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):  # average ranks over ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos, n_neg = int((true == 1).sum()), int((true == 0).sum())
    return float((ranks[true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def patient_bootstrap_share(pred, subject_idx, n_boot=2000, seed=0):
    """95% interval for the share of slices predicted AD, resampling patients (slices of one
    patient are strongly correlated, so resampling slices would understate the uncertainty)."""
    g = torch.Generator().manual_seed(seed)
    subjects = subject_idx.unique()
    per = [(pred[subject_idx == s].sum().item(), int((subject_idx == s).sum())) for s in subjects]
    pos = torch.tensor([p for p, _ in per], dtype=torch.double)
    cnt = torch.tensor([n for _, n in per], dtype=torch.double)
    draws = torch.randint(0, len(subjects), (n_boot, len(subjects)), generator=g)
    shares = pos[draws].sum(1) / cnt[draws].sum(1)
    return [float(torch.quantile(shares, 0.025)), float(torch.quantile(shares, 0.975))]


def slice_index(ds):
    """Absolute slice index (anatomical level) of every slice in a split, from its records."""
    return torch.tensor([r.slice_index for r in ds.records])


def position_baseline(train, other):
    """Anatomical-level-only predictor (Discovery 9: AD windows start about 10 slices lower).

    Scores each slice by the AD share of training slices at the same absolute slice index and
    predicts AD where that share exceeds the overall training AD share (the balanced-accuracy
    rule). Returns its balanced accuracy and AUROC on `other`: the confound reference that the
    CNN must beat to be reading anatomy rather than level."""
    idx_tr, idx_o = slice_index(train), slice_index(other)
    overall = float(train.labels.float().mean())
    share = {}
    for i in idx_tr.unique().tolist():
        m = idx_tr == i
        share[i] = (float(train.labels[m].sum()) + overall) / (int(m.sum()) + 1)  # one pseudo-count
    score = torch.tensor([share.get(int(i), overall) for i in idx_o])
    pred = (score > overall).long()
    return dict(balanced_accuracy=balanced_accuracy(pred, other.labels), auroc=auroc(score, other.labels))


def within_level_accuracy(pred, ds, min_per_class=20):
    """CNN balanced accuracy computed separately at each absolute slice index that has at least
    min_per_class slices of each class. Level cannot help within one index, so this isolates
    anatomy. Returns {index: balanced accuracy}."""
    idx = slice_index(ds)
    out = {}
    for i in idx.unique().tolist():
        m = idx == i
        y = ds.labels[m]
        if int((y == 0).sum()) >= min_per_class and int((y == 1).sum()) >= min_per_class:
            out[int(i)] = balanced_accuracy(pred[m], y)
    return out


@torch.no_grad()
def predict_proba(model, images_uint8, device, batch=1000):
    """AD probability for each uint8 slice."""
    model.eval()
    out = [torch.sigmoid(model(to_input(images_uint8[s:s + batch].to(device)))).cpu()
           for s in range(0, len(images_uint8), batch)]
    return torch.cat(out)


def evaluate_real(model, ds, device, threshold=0.5):
    """Balanced accuracy, AUROC and predicted AD share (with a patient-bootstrap interval) on a real split."""
    p = predict_proba(model, ds.images, device)
    pred = (p >= threshold).long()
    return dict(balanced_accuracy=balanced_accuracy(pred, ds.labels), auroc=auroc(p, ds.labels),
                predicted_ad_share=float(pred.float().mean()),
                predicted_ad_share_ci95=patient_bootstrap_share(pred, ds.subject_idx),
                true_ad_share=float(ds.labels.float().mean()), n=len(ds)), p


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_classifier(splits, device, args):
    """Train on the training split with class-weighted BCE; keep the epoch with the best
    validation balanced accuracy (ties broken by AUROC)."""
    torch.manual_seed(args.seed)
    train, val = splits["train"], splits["val"]
    model = SliceClassifier(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    x_all, y_all = train.images.to(device), train.labels.to(device)
    pos_weight = (y_all == 0).sum().float() / (y_all == 1).sum().float()  # balances AD vs CN
    gen = torch.Generator(device=device).manual_seed(args.seed)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_all), device=device, generator=gen)
        total, t0 = 0.0, time.time()
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            x = augment(to_input(x_all[idx]), gen)
            loss = F.binary_cross_entropy_with_logits(model(x), y_all[idx].float(), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_stats, _ = evaluate_real(model, val, device)
        history.append(dict(epoch=epoch, loss=total / len(perm), val_bal_acc=val_stats["balanced_accuracy"],
                            val_auroc=val_stats["auroc"]))
        print(f"epoch {epoch:>3}  loss {total / len(perm):.4f}  val balanced acc {val_stats['balanced_accuracy']:.3f}  "
              f"val AUROC {val_stats['auroc']:.3f}  ({time.time() - t0:.1f}s)", flush=True)
        key = (val_stats["balanced_accuracy"], val_stats["auroc"])
        if best is None or key > best[0]:
            best = (key, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    print(f"Selected epoch {best[1]} (val balanced acc {best[0][0]:.3f}, AUROC {best[0][1]:.3f})")
    return model, best[1], history


# --------------------------------------------------------------------------- #
# Generated cohorts
# --------------------------------------------------------------------------- #
def parse_run(spec):
    """'run_name' or 'run_name:iteration' -> (run_name, iteration or None for the last snapshot)."""
    name, _, it = spec.partition(":")
    return name, int(it) if it else None


def evaluate_run(spec, model, train_labels, device, args):
    """Generate a cohort from one snapshot (same seed and size as predict.py) and classify it."""
    name, it = parse_run(spec)
    config, snaps = load_run(os.path.join(args.runs_root, name))
    if it is None:
        it = snapshot_iter(snaps[-1])
    match = [p for p in snaps if snapshot_iter(p) == it]
    if not match:
        raise SystemExit(f"No snapshot {it} for {name}; available {[snapshot_iter(p) for p in snaps]}")
    if config["resolution"] != args.resolution:
        raise SystemExit(f"{name} is {config['resolution']} px; the classifier is {args.resolution} px")
    g = build_generator(config, device)
    g.load_state_dict(torch.load(match[0], map_location=device))
    labels = class_labels(config, args.n_samples, train_labels, args.gen_seed)
    with torch.no_grad():
        images = generate(g, args.n_samples, config["latent_dim"], device, args.gen_seed, labels=labels)
    p = predict_proba(model, images, device)
    pred = (p >= 0.5).long()
    k = int(pred.sum())
    out = dict(run=name, iter=it, n=len(pred), predicted_ad_share=k / len(pred),
               predicted_ad_share_ci95=list(wilson(k, len(pred))), mean_ad_probability=float(p.mean()))
    if labels is not None:  # class-conditional: label fidelity
        ad_req, cn_req = labels == 1, labels == 0
        out.update(requested_ad_share=float(ad_req.float().mean()),
                   ad_requested_classified_ad=float(pred[ad_req].float().mean()),
                   cn_requested_classified_cn=float((1 - pred[cn_req]).float().mean()),
                   label_balanced_agreement=balanced_accuracy(pred, labels))
    return out, p, labels


def plot_summary(real_test, p_test, test_labels, runs, path, expected):
    """Left: AD-probability histograms (real CN, real AD, each cohort). Right: predicted AD share
    per cohort against the real-test reference band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
    bins = torch.linspace(0, 1, 26).numpy()
    a1.hist(p_test[test_labels == 0].numpy(), bins=bins, density=True, histtype="step", lw=2, color="#4C72B0",
            label="real test CN")
    a1.hist(p_test[test_labels == 1].numpy(), bins=bins, density=True, histtype="step", lw=2, color="#C44E52",
            label="real test AD")
    for r, p, _ in runs:
        a1.hist(p.numpy(), bins=bins, density=True, histtype="step", lw=1.2, ls="--", label=f"{r['run']} @ {r['iter']}")
    a1.set_xlabel("classifier AD probability")
    a1.set_ylabel("density")
    a1.legend(fontsize=6.5, loc="upper center")
    a1.set_title("What the classifier sees", fontsize=10)

    names = [f"{r['run']} @ {r['iter']}" for r, _, _ in runs]
    shares = [r["predicted_ad_share"] for r, _, _ in runs]
    errs = [[s - r["predicted_ad_share_ci95"][0] for s, (r, _, _) in zip(shares, runs)],
            [r["predicted_ad_share_ci95"][1] - s for s, (r, _, _) in zip(shares, runs)]]
    a2.bar(range(len(runs)), shares, yerr=errs, color="#8C8C8C", capsize=3)
    lo, hi = real_test["predicted_ad_share_ci95"]
    a2.axhspan(lo, hi, color="#C44E52", alpha=0.15, label="real test slices, predicted AD (patient-bootstrap 95%)")
    a2.axhline(real_test["predicted_ad_share"], color="#C44E52", lw=1.5)
    a2.axhline(expected, color="black", ls="--", lw=1.2, label="expected for a faithful generator")
    a2.set_xticks(range(len(runs)))
    a2.set_xticklabels(names, fontsize=7, rotation=35, ha="right", rotation_mode="anchor")
    a2.set_ylabel("share classified AD")
    a2.legend(fontsize=7, loc="upper right")
    a2.set_title("Predicted AD share of each generated cohort", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    """Train (or load) the classifier, report it on real data with the position-only baseline,
    then classify each requested generated cohort and save a summary and figure."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="AD/CN slice classifier for class fidelity of generated cohorts.")
    p.add_argument("--runs", nargs="*", default=[], help="run_name or run_name:iteration (default: last snapshot)")
    p.add_argument("--eval_only", action="store_true", help="load the saved classifier instead of training")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="classifier training seed")
    p.add_argument("--split_seed", type=int, default=DEFAULT_SEED, help="patient split seed (must match training)")
    p.add_argument("--n_samples", type=int, default=3100, help="generated slices per cohort (as in predict.py)")
    p.add_argument("--gen_seed", type=int, default=0, help="latent seed (as in predict.py)")
    p.add_argument("--tag", default="adcn")
    p.add_argument("--runs_root", default=os.path.join(here, "outputs", "runs"))
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_dir", default=os.path.join(here, "outputs", "classifier"))
    p.add_argument("--fig_dir", default=os.path.join(here, "figures"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    splits = get_adni_splits(resolution=args.resolution, seed=args.split_seed, cache_dir=args.cache_dir,
                             num_workers=args.num_workers)
    train, val, test = splits["train"], splits["val"], splits["test"]
    print(f"Slices: train {len(train)}, val {len(val)}, test {len(test)} | device {device}")

    ckpt = os.path.join(args.out_dir, f"classifier_r{args.resolution}.pt")
    if args.eval_only:
        state = torch.load(ckpt, map_location=device)
        model = SliceClassifier(state["width"]).to(device)
        model.load_state_dict(state["model"])
        epoch, history = state["epoch"], state["history"]
    else:
        t0 = time.time()
        model, epoch, history = train_classifier(splits, device, args)
        print(f"Training took {time.time() - t0:.0f}s")
        torch.save(dict(model=model.state_dict(), width=args.width, epoch=epoch, history=history), ckpt)
    n_params = sum(q.numel() for q in model.parameters())

    val_stats, _ = evaluate_real(model, val, device)
    test_stats, p_test = evaluate_real(model, test, device)
    pos_val, pos_test = position_baseline(train, val), position_baseline(train, test)
    print("\n== Classifier on real slices ==")
    print(f"parameters {n_params:,}, selected epoch {epoch}")
    for name, s, pos in (("val", val_stats, pos_val), ("test", test_stats, pos_test)):
        print(f"{name:<5} CNN balanced acc {s['balanced_accuracy']:.3f}, AUROC {s['auroc']:.3f} | "
              f"slice-level-only balanced acc {pos['balanced_accuracy']:.3f}, AUROC {pos['auroc']:.3f} | "
              f"predicted AD share {s['predicted_ad_share']:.3f} "
              f"[{s['predicted_ad_share_ci95'][0]:.3f}, {s['predicted_ad_share_ci95'][1]:.3f}], "
              f"true {s['true_ad_share']:.3f}")
    per_level = within_level_accuracy((p_test >= 0.5).long(), test)
    if per_level:
        v = torch.tensor(list(per_level.values()))
        print(f"test CNN balanced acc within a single slice index (level held fixed): median {v.median():.3f}, "
              f"range {v.min():.3f}-{v.max():.3f}, over {len(v)} indices with >= 20 slices per class")
    else:
        print("no slice index has >= 20 test slices of each class; within-level check skipped")

    # A faithful generator reproduces the training class mix; given the classifier's test error
    # rates, it would be classified AD at this rate (the reference for generated cohorts).
    pred_t = (p_test >= 0.5).long()
    tpr = float(pred_t[test.labels == 1].float().mean())
    fpr = float(pred_t[test.labels == 0].float().mean())
    train_ad = float(train.labels.float().mean())
    expected = tpr * train_ad + fpr * (1 - train_ad)
    reference = dict(train_true_ad_share=train_ad, test_tpr=tpr, test_fpr=fpr, expected_faithful_share=expected)

    runs = []
    if args.runs:
        print("\n== Generated cohorts ==")
        print(f"reference: a faithful generator (training AD share {train_ad:.3f}) would be classified AD "
              f"{expected:.3f} of the time (test TPR {tpr:.3f}, FPR {fpr:.3f}); real test slices: "
              f"{test_stats['predicted_ad_share']:.3f}")
    for spec in args.runs:
        r, pr, labels = evaluate_run(spec, model, train.labels, device, args)
        r["share_of_expected"] = r["predicted_ad_share"] / max(expected, 1e-9)
        runs.append((r, pr, labels))
        line = (f"{r['run']:<26} {r['iter']:>6}  predicted AD {r['predicted_ad_share']:.3f} "
                f"[{r['predicted_ad_share_ci95'][0]:.3f}, {r['predicted_ad_share_ci95'][1]:.3f}]  "
                f"= {r['predicted_ad_share'] / max(expected, 1e-9):.0%} of expected")
        if "ad_requested_classified_ad" in r:
            line += (f"  | AD-requested -> AD {r['ad_requested_classified_ad']:.3f}, "
                     f"CN-requested -> CN {r['cn_requested_classified_cn']:.3f}")
        print(line, flush=True)

    summary = dict(resolution=args.resolution, parameters=n_params, selected_epoch=epoch, history=history,
                   val=val_stats, test=test_stats, reference=reference, position_only=dict(val=pos_val, test=pos_test),
                   test_within_level=per_level, runs=[r for r, _, _ in runs])
    with open(os.path.join(args.out_dir, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if runs:
        path = os.path.join(args.fig_dir, f"classifier_{args.tag}.png")
        plot_summary(test_stats, p_test, test.labels, runs, path, expected)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()