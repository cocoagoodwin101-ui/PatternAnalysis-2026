"""Memorisation audit, part 1: calibrate the distance ladder on real ADNI data only.

Design: README Section 7. Every rung is a nearest-neighbour similarity, measured with
SSIM (higher = more similar), both over the whole image and over the brain only:

    near-duplicate    adjacent slices of the same scan (training split)
    same person       each slice of one visit -> nearest slice in any OTHER visit of the
                      same patient (training patients with 2+ scans)
    different person  each held-out test slice -> nearest slice in the whole training split

The memorisation threshold for each class is a high percentile (default 95th) of the
same-person similarities, i.e. "closer than 95% of genuine repeat visits of one brain".
Part 2 will compare generated slices against the same training split and threshold.

The key diagnostic printed at the end is the false-flag rate: the share of real,
genuinely new test brains that already exceed the threshold. If it is high, that metric
cannot tell "same brain" from "any brain" and must not be used to judge memorisation.

Usage (GPU job on Rangpur, see audit.sh):
    python audit.py --resolution 64 --max_test_queries 500   # quick first run
    python audit.py --resolution 64                           # full run

Outputs: raw per-slice similarities and thresholds in outputs/audit/ (gitignored; contains
local split indices only, no patient IDs) and a histogram figure in figures/.
"""

import argparse
import json
import os
import re
import statistics
import time
from datetime import date
from collections import defaultdict

import torch
import torch.nn.functional as F

from dataset import ADNI_ROOT, CLASS_NAMES, DEFAULT_CACHE, DEFAULT_SEED, get_adni_splits

# SSIM constants from Wang et al. (2004): 11x11 Gaussian window, sigma 1.5, K1 = 0.01,
# K2 = 0.03, data range 1. The map is computed with 'valid' convolution (no padding), so
# border pixels never mix with artificial padding; this matches the mean that
# skimage.metrics.structural_similarity(gaussian_weights=True, data_range=1,
# use_sample_covariance=False) reports.
WIN, SIGMA = 11, 1.5
C1, C2 = 0.01 ** 2, 0.03 ** 2

# Pixels brighter than this (uint8) count as brain. Same cut-off as the near-black
# background measurement in data_audit.py (Discovery 8).
BRAIN_THRESHOLD = 10

# Upper bound on elements per pairwise tensor; keeps peak memory at a few GB on an A100.
PAIR_BUDGET = 2 ** 27


def gaussian_kernels(device):
    """Separable 1D Gaussian kernels shaped for conv2d (vertical, horizontal)."""
    coords = torch.arange(WIN, dtype=torch.float32, device=device) - WIN // 2
    g = torch.exp(-coords ** 2 / (2 * SIGMA ** 2))
    g = g / g.sum()
    return g.view(1, 1, WIN, 1), g.view(1, 1, 1, WIN)


class SSIMIndex:
    """Pairwise SSIM between slices of one preloaded image tensor.

    Per-image statistics (local mean, local variance, brain mask) are computed once.
    For each pair only the cross term blur(x * y) is new, so a query is compared with
    thousands of candidates per batched GPU call.
    """

    def __init__(self, images_uint8, device, brain_threshold=BRAIN_THRESHOLD):
        self.device = device
        self.kv, self.kh = gaussian_kernels(device)
        self.x = images_uint8.to(device).float() / 255.0  # (N, 1, R, R) in [0, 1]
        self.mu = self._blur(self.x)  # (N, 1, R', R'), R' = R - WIN + 1
        self.var = self._blur(self.x * self.x) - self.mu ** 2
        c = WIN // 2
        brain = (images_uint8 > brain_threshold).to(device).float()
        self.mask = brain[..., c:-c, c:-c]  # aligned with the 'valid' SSIM map
        self.n, self.res = self.x.shape[0], self.x.shape[-1]

    def _blur(self, x):
        return F.conv2d(F.conv2d(x, self.kv), self.kh)

    @staticmethod
    def _ssim(mq, mc, vq, vc, bq, bc, cross):
        """Whole-image and brain-masked mean SSIM; all inputs broadcast to the cross map.

        The brain mask is the union of the two images' brain pixels, so background that
        is black in both images is ignored but a brain region missing from one image
        still counts against the pair.
        """
        cov = cross - mq * mc
        ssim_map = ((2 * mq * mc + C1) * (2 * cov + C2)) / ((mq ** 2 + mc ** 2 + C1) * (vq + vc + C2))
        whole = ssim_map.mean(dim=(-2, -1))
        union = bq + bc - bq * bc
        masked = (ssim_map * union).sum(dim=(-2, -1)) / union.sum(dim=(-2, -1)).clamp_min(1.0)
        return whole, masked

    def pairwise(self, qi, ci):
        """(q, c) whole and masked SSIM between every query qi and candidate ci."""
        prod = (self.x[qi][:, None] * self.x[ci][None]).flatten(0, 1)  # (q * c, 1, R, R)
        cross = self._blur(prod).view(len(qi), len(ci), *self.mu.shape[-2:])
        q = lambda t: t[qi][:, None, 0]  # (q, 1, R', R')
        c = lambda t: t[ci][None, :, 0]  # (1, c, R', R')
        return self._ssim(q(self.mu), c(self.mu), q(self.var), c(self.var), q(self.mask), c(self.mask), cross)

    def paired(self, qi, ci, chunk=4096):
        """Whole and masked SSIM for aligned pairs (qi[k], ci[k])."""
        whole, masked = [], []
        for s in range(0, len(qi), chunk):
            a, b = qi[s:s + chunk], ci[s:s + chunk]
            cross = self._blur(self.x[a] * self.x[b])  # (k, 1, R', R')
            w, m = self._ssim(self.mu[a], self.mu[b], self.var[a], self.var[b],
                              self.mask[a], self.mask[b], cross)
            whole.append(w[:, 0])
            masked.append(m[:, 0])
        return torch.cat(whole), torch.cat(masked)

    def nearest(self, qi, ci, q_chunk=32):
        """For each query in qi: best whole SSIM, best masked SSIM, and their argmax in ci.

        Each metric picks its own nearest neighbour. Candidates are processed in chunks
        sized from PAIR_BUDGET, keeping a running maximum.
        """
        c_chunk = max(64, PAIR_BUDGET // (q_chunk * self.res * self.res))
        out = {k: [] for k in ("whole", "masked", "whole_arg", "masked_arg")}
        for qs in range(0, len(qi), q_chunk):
            q = qi[qs:qs + q_chunk]
            best_w = torch.full((len(q),), -2.0, device=self.device)
            best_m = best_w.clone()
            arg_w = torch.zeros(len(q), dtype=torch.long, device=self.device)
            arg_m = arg_w.clone()
            for cs in range(0, len(ci), c_chunk):
                c = ci[cs:cs + c_chunk]
                w, m = self.pairwise(q, c)
                wv, wa = w.max(dim=1)
                mv, ma = m.max(dim=1)
                upd_w, upd_m = wv > best_w, mv > best_m
                best_w = torch.where(upd_w, wv, best_w)
                arg_w = torch.where(upd_w, c[wa], arg_w)
                best_m = torch.where(upd_m, mv, best_m)
                arg_m = torch.where(upd_m, c[ma], arg_m)
            for k, v in zip(out, (best_w, best_m, arg_w, arg_m)):
                out[k].append(v)
        return {k: torch.cat(v).cpu() for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Ladder rungs
# --------------------------------------------------------------------------- #
def near_duplicate(train, index):
    """SSIM between each pair of adjacent slices (slice_pos k and k+1) of the same scan."""
    by_scan = defaultdict(dict)
    for i, (s, p) in enumerate(zip(train.scan_idx.tolist(), train.slice_pos.tolist())):
        by_scan[s][p] = i
    a, b = [], []
    for slices in by_scan.values():
        for p, i in slices.items():
            if p + 1 in slices:
                a.append(i)
                b.append(slices[p + 1])
    a = torch.tensor(a, device=index.device)
    b = torch.tensor(b, device=index.device)
    whole, masked = index.paired(a, b)
    return dict(whole=whole.cpu(), masked=masked.cpu(), labels=train.labels[a.cpu()])


def same_person(train, index):
    """Each slice of a multi-visit patient -> nearest slice in any other visit of that patient.

    Taking the nearest over all other visits mirrors the generated-sample audit, which
    searches every training slice (including every visit of every patient).
    """
    by_subject = defaultdict(list)
    for i, s in enumerate(train.subject_idx.tolist()):
        by_subject[s].append(i)

    whole, masked, query, n_patients = [], [], [], defaultdict(int)
    for slices in by_subject.values():
        scans = train.scan_idx[slices]
        if len(scans.unique()) < 2:
            continue
        n_patients[int(train.labels[slices[0]])] += 1
        idx = torch.tensor(slices, device=index.device)
        w, m = index.pairwise(idx, idx)
        same_scan = (scans[:, None] == scans[None, :]).to(index.device)
        whole.append(w.masked_fill(same_scan, -2.0).max(dim=1).values.cpu())
        masked.append(m.masked_fill(same_scan, -2.0).max(dim=1).values.cpu())
        query.extend(slices)
    query = torch.tensor(query)
    return dict(whole=torch.cat(whole), masked=torch.cat(masked), labels=train.labels[query],
                subject_idx=train.subject_idx[query], n_patients=dict(n_patients))


# --------------------------------------------------------------------------- #
# Visit-pair diagnostic
# --------------------------------------------------------------------------- #
# The metadata 'raw' path ends in ..._br_raw_<YYYYMMDDhhmmss...>_<n>_S<series>_I<image>.nii.
# The timestamp is assumed to be the acquisition date; the diagnostic itself tests this
# (scans from genuinely different days should never be near-identical).
_RAW_RE = re.compile(r"_(\d{8})\d*_\d+_S(\d+)_I(\d+)\.nii")
GAP_BINS = ((1, 180, "1-179 d"), (180, 730, "180-729 d"), (730, 100000, ">=730 d"))


def load_sessions(records, root=ADNI_ROOT):
    """Map scan_id -> (acquisition date, series ID), or None if the path does not parse."""
    with open(os.path.join(root, "meta_data_with_label.json")) as f:
        meta = json.load(f)
    sessions = {}
    for scan in {r.scan_id for r in records}:
        m = _RAW_RE.search(meta[scan]["raw"])
        d = m.group(1) if m else None
        sessions[scan] = (date(int(d[:4]), int(d[4:6]), int(d[6:])), m.group(2)) if m else None
    return sessions


def visit_pairs(train, index, sessions):
    """Compare every pair of scans of the same training patient and categorise the pair.

    For scans A and B (A before B in scan order) the pair similarity is the median, over
    A's slices, of the best masked SSIM to any slice of B. Also recorded: the median
    absolute slice-index offset between each A slice and its best match (near 0 means
    slices correspond by index, i.e. visits are aligned), and how many absolute slice
    indices the two 20-slice windows share.
    """
    by_subject = defaultdict(list)
    for i, s in enumerate(train.subject_idx.tolist()):
        by_subject[s].append(i)
    slice_index = torch.tensor([r.slice_index for r in train.records])

    pairs = []
    for slices in by_subject.values():
        scans = train.scan_idx[slices]
        if len(scans.unique()) < 2:
            continue
        members = torch.tensor(slices)
        _, m = index.pairwise(members.to(index.device), members.to(index.device))
        m = m.cpu()
        local_scans = scans.unique().tolist()
        for a_pos, a in enumerate(local_scans):
            for b in local_scans[a_pos + 1:]:
                rows = (scans == a).nonzero().flatten()
                cols = (scans == b).nonzero().flatten()
                best, arg = m[rows][:, cols].max(dim=1)
                ia, ib = members[rows], members[cols[arg]]
                ses_a = sessions.get(train.records[int(ia[0])].scan_id)
                ses_b = sessions.get(train.records[int(members[cols[0]])].scan_id)
                if ses_a is None or ses_b is None:
                    category, gap = "unparsed", None
                elif ses_a[1] == ses_b[1]:
                    category, gap = "same series", 0
                elif ses_a[0] == ses_b[0]:
                    category, gap = "same day", 0
                else:
                    category, gap = "different day", abs((ses_a[0] - ses_b[0]).days)
                win_a = set(slice_index[members[rows]].tolist())
                win_b = set(slice_index[members[cols]].tolist())
                pairs.append(dict(
                    label=int(train.labels[slices[0]]), category=category, gap_days=gap,
                    ssim=float(best.median()),
                    offset=float((slice_index[ib] - slice_index[ia]).abs().float().median()),
                    overlap=len(win_a & win_b)))
    return pairs


def summarise_pairs(pairs):
    """Print visit-pair statistics by category (and day gap) and class."""
    def group(p):
        if p["category"] != "different day":
            return p["category"]
        return next(name for lo, hi, name in GAP_BINS if lo <= p["gap_days"] < hi)

    order = ["same series", "same day"] + [name for _, _, name in GAP_BINS] + ["unparsed"]
    print(f"\nVisit pairs of training patients (pair SSIM = median over A's slices of best masked SSIM in B)")
    print(f"{'pair type':<14} {'class':<5} {'pairs':>6} {'SSIM p50':>9} {'SSIM>0.99':>9} "
          f"{'|offset| p50':>12} {'overlap p50':>11}")
    for g in order:
        for label, cls in enumerate(CLASS_NAMES):
            sel = [p for p in pairs if group(p) == g and p["label"] == label]
            if not sel:
                continue
            ssim = [p["ssim"] for p in sel]
            print(f"{g:<14} {cls:<5} {len(sel):>6} {statistics.median(ssim):>9.4f} "
                  f"{sum(v > 0.99 for v in ssim) / len(sel):>9.1%} "
                  f"{statistics.median(p['offset'] for p in sel):>12.1f} "
                  f"{statistics.median(p['overlap'] for p in sel):>11.0f}")


def different_person(train, test, train_index, test_index, max_queries=None, seed=DEFAULT_SEED):
    """Each held-out test slice -> nearest slice in the whole training split.

    Test and train images live in separate SSIMIndex objects, so a small adapter
    concatenates them into one index for the search.
    """
    q = torch.arange(len(test))
    if max_queries is not None and max_queries < len(test):
        q = torch.randperm(len(test), generator=torch.Generator().manual_seed(seed))[:max_queries].sort().values

    joint = _join(train_index, test_index)
    offset = train_index.n
    cand = torch.arange(train_index.n, device=joint.device)
    start = time.time()
    res = joint.nearest(q.to(joint.device) + offset, cand)
    if joint.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    res.update(labels=test.labels[q], nn_label_whole=train.labels[res["whole_arg"]],
               nn_label_masked=train.labels[res["masked_arg"]], seconds=elapsed, n_queries=len(q))
    return res


def _join(a, b):
    """Concatenate two SSIMIndex objects (cheap: only the stored tensors are joined)."""
    j = object.__new__(SSIMIndex)
    j.device, j.kv, j.kh, j.res = a.device, a.kv, a.kh, a.res
    for name in ("x", "mu", "var", "mask"):
        setattr(j, name, torch.cat([getattr(a, name), getattr(b, name)]))
    j.n = a.n + b.n
    return j


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def percentiles(v, ps=(5, 50, 95)):
    v = v.float()
    return [float(torch.quantile(v, p / 100)) for p in ps] if len(v) else [float("nan")] * len(ps)


def summarise(rungs, threshold_pct):
    """Print the ladder table, compute per-class thresholds and the false-flag rate."""
    print(f"\n{'rung':<17} {'class':<5} {'metric':<7} {'n':>7} {'p5':>7} {'p50':>7} {'p95':>7}")
    for name, r in rungs.items():
        for label, cls in enumerate(CLASS_NAMES):
            sel = r["labels"] == label
            for metric in ("whole", "masked"):
                p5, p50, p95 = percentiles(r[metric][sel])
                print(f"{name:<17} {cls:<5} {metric:<7} {int(sel.sum()):>7} {p5:>7.4f} {p50:>7.4f} {p95:>7.4f}")

    thresholds, sp, dp = {}, rungs["same_person"], rungs["different_person"]
    print(f"\nMemorisation threshold = {threshold_pct}th percentile of same-person SSIM, per class.")
    print("False-flag rate = share of real held-out test slices (new brains) at or above it.")
    print(f"{'class':<5} {'metric':<7} {'threshold':>9} {'false-flag':>10} {'NN same class':>13}")
    for label, cls in enumerate(CLASS_NAMES):
        thresholds[cls] = {}
        for metric in ("whole", "masked"):
            t = float(torch.quantile(sp[metric][sp["labels"] == label].float(), threshold_pct / 100))
            sel = dp["labels"] == label
            flag = float((dp[metric][sel] >= t).float().mean())
            same_cls = float((dp[f"nn_label_{metric}"][sel] == label).float().mean())
            thresholds[cls][metric] = t
            print(f"{cls:<5} {metric:<7} {t:>9.4f} {flag:>10.1%} {same_cls:>13.1%}")
    return thresholds


def plot_ladder(rungs, thresholds, path, resolution):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"near_duplicate": "#8172B2", "same_person": "#C44E52", "different_person": "#4C72B0"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for row, cls in enumerate(CLASS_NAMES):
        for col, metric in enumerate(("whole", "masked")):
            ax = axes[row, col]
            for name, r in rungs.items():
                v = r[metric][r["labels"] == row].numpy()
                ax.hist(v, bins=60, range=(0, 1), density=True, alpha=0.55, color=colours[name],
                        label=name.replace("_", " "))
            ax.axvline(thresholds[cls][metric], color="k", ls="--", lw=1, label="threshold")
            ax.set_title(f"{cls}, {metric}-image SSIM ({resolution} px)")
            ax.set_xlabel("nearest-neighbour SSIM (higher = more similar)")
            if col == 0:
                ax.set_ylabel("density")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Calibrate the memorisation-audit distance ladder on real ADNI data.")
    p.add_argument("--resolution", type=int, default=64,
                   help="must match the resolution the generator will be audited at")
    p.add_argument("--max_test_queries", type=int, default=None,
                   help="subsample test queries for the different-person rung (quick runs)")
    p.add_argument("--threshold_pct", type=float, default=95.0)
    p.add_argument("--brain_threshold", type=int, default=BRAIN_THRESHOLD)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_dir", default=os.path.join(here, "outputs", "audit"))
    p.add_argument("--fig_dir", default=os.path.join(here, "figures"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    splits = get_adni_splits(resolution=args.resolution, seed=args.seed, cache_dir=args.cache_dir,
                             num_workers=args.num_workers)
    train, test = splits["train"], splits["test"]
    train_index = SSIMIndex(train.images, device, args.brain_threshold)
    test_index = SSIMIndex(test.images, device, args.brain_threshold)
    print(f"Train {len(train)} slices, test {len(test)} slices at {args.resolution} px")

    rungs = {}
    t0 = time.time()
    rungs["near_duplicate"] = near_duplicate(train, train_index)
    print(f"near-duplicate: {len(rungs['near_duplicate']['whole'])} adjacent pairs ({time.time() - t0:.1f}s)")

    t0 = time.time()
    rungs["same_person"] = same_person(train, train_index)
    print(f"same person: {len(rungs['same_person']['whole'])} query slices from patients "
          f"{ {CLASS_NAMES[k]: v for k, v in rungs['same_person']['n_patients'].items()} } "
          f"({time.time() - t0:.1f}s)")

    sessions = load_sessions(train.records)
    n_bad = sum(v is None for v in sessions.values())
    print(f"Parsed acquisition date and series for {len(sessions) - n_bad}/{len(sessions)} training scans")
    pairs = visit_pairs(train, train_index, sessions)
    summarise_pairs(pairs)

    rungs["different_person"] = different_person(train, test, train_index, test_index,
                                                 args.max_test_queries, args.seed)
    dp = rungs["different_person"]
    rate = dp["n_queries"] / dp["seconds"]
    print(f"different person: {dp['n_queries']} test queries x {len(train)} train slices in "
          f"{dp['seconds']:.1f}s ({rate:.1f} queries/s; all {len(test)} would take ~{len(test) / rate / 60:.1f} min)")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    thresholds = summarise(rungs, args.threshold_pct)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    tag = f"r{args.resolution}"
    torch.save(dict(rungs=rungs, visit_pairs=pairs), os.path.join(args.out_dir, f"ladder_{tag}.pt"))
    with open(os.path.join(args.out_dir, f"thresholds_{tag}.json"), "w") as f:
        json.dump(dict(resolution=args.resolution, percentile=args.threshold_pct,
                       brain_threshold=args.brain_threshold, seed=args.seed,
                       test_queries=dp["n_queries"], thresholds=thresholds), f, indent=2)
    fig_path = os.path.join(args.fig_dir, f"audit_ladder_{tag}.png")
    plot_ladder(rungs, thresholds, fig_path, args.resolution)
    print(f"\nSaved {args.out_dir}/ladder_{tag}.pt, thresholds_{tag}.json and {fig_path}")


if __name__ == "__main__":
    main()