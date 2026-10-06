"""Memorisation audit, part 1: calibrate the similarity ladder on real ADNI data only.

Design: README Section 7. Every rung is a nearest-neighbour similarity measured with
SSIM (higher = more similar), over the whole image and over the brain only. Rungs:

    Copy references (what memorising a training image looks like):
      near-duplicate   adjacent slices of the same training scan
      same session     training slice -> nearest slice of another scan of the same patient
                       acquired on the same day (repeat acquisition)
    Identity reference (reported, not used for the threshold):
      different day    training slice -> nearest slice of the same patient's scans from
                       other days (same brain, different head position)
    Strangers (genuinely new brains):
      stranger val     validation slice -> nearest training slice   [calibrates threshold]
      stranger test    test slice -> nearest training slice         [checks false flags]

Threshold: per class and metric, a high percentile (default 99th) of the validation
stranger similarities. A generated slice above it is closer to a training slice than
almost any real new brain gets, and is inspected as a possible copy. The test strangers
estimate the false-flag rate on data not used for calibration; the copy references show
whether the threshold catches copies.

Why not "same person" as the threshold: the first calibration run (README Section 7)
showed that same-day repeat scans are near-identical (SSIM ~0.99), while visits on
different days are not aligned to each other, so pixel SSIM cannot recognise a patient
across visits. The same-person threshold was therefore ~1.0 and would pass most copies.

Also printed: a visit-pair table categorising every pair of scans of the same patient by
series ID and acquisition date (parsed from the metadata 'raw' path).

Usage (GPU job on Rangpur, see audit.sh):
    python audit.py --resolution 64 --max_queries 500   # quick run
    python audit.py --resolution 64                     # full run

Outputs: raw per-slice similarities and thresholds in outputs/audit/ (gitignored; contains
local split indices only, no patient IDs) and a histogram figure in figures/.
"""

import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import date

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


def pair_category(ses_a, ses_b):
    """Categorise two scans of one patient: (category, days apart or None)."""
    if ses_a is None or ses_b is None:
        return "unparsed", None
    if ses_a[1] == ses_b[1]:
        return "same series", 0
    if ses_a[0] == ses_b[0]:
        return "same day", 0
    return "different day", abs((ses_a[0] - ses_b[0]).days)


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
                category, gap = pair_category(sessions.get(train.records[int(ia[0])].scan_id),
                                              sessions.get(train.records[int(members[cols[0]])].scan_id))
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
    print("\nVisit pairs of training patients (pair SSIM = median over A's slices of best masked SSIM in B)")
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


def same_patient(train, index, sessions):
    """Same-session and different-day rungs for training patients with 2+ scans.

    For each slice, the nearest slice among the patient's other scans in each category:
    'same session' covers scans from the same day (including the same series reprocessed),
    'different day' covers scans acquired on another day. A slice contributes to a rung
    only if its patient has at least one scan in that category. Scans whose date does not
    parse are left out of both.
    """
    to_rung = {"same series": "same_session", "same day": "same_session", "different day": "different_day"}
    by_subject = defaultdict(list)
    for i, s in enumerate(train.subject_idx.tolist()):
        by_subject[s].append(i)

    out = {name: dict(whole=[], masked=[], query=[]) for name in ("same_session", "different_day")}
    for slices in by_subject.values():
        scans = train.scan_idx[slices].tolist()
        if len(set(scans)) < 2:
            continue
        ses = {s: sessions.get(train.records[i].scan_id) for s, i in zip(scans, slices)}
        rung_of = {(a, b): to_rung.get(pair_category(ses[a], ses[b])[0])
                   for a in ses for b in ses if a != b}
        members = torch.tensor(slices, device=index.device)
        w, m = index.pairwise(members, members)
        for name in out:
            allowed = torch.tensor([[rung_of.get((a, b)) == name for b in scans] for a in scans],
                                   device=index.device)
            rows = allowed.any(dim=1)
            if not rows.any():
                continue
            out[name]["whole"].append(w.masked_fill(~allowed, -2.0).max(dim=1).values[rows].cpu())
            out[name]["masked"].append(m.masked_fill(~allowed, -2.0).max(dim=1).values[rows].cpu())
            out[name]["query"].append(torch.tensor(slices)[rows.cpu()])

    rungs = {}
    for name, r in out.items():
        query = torch.cat(r["query"]) if r["query"] else torch.zeros(0, dtype=torch.long)
        rungs[name] = dict(whole=torch.cat(r["whole"]) if r["whole"] else torch.zeros(0),
                           masked=torch.cat(r["masked"]) if r["masked"] else torch.zeros(0),
                           labels=train.labels[query], subject_idx=train.subject_idx[query])
    return rungs


def stranger(train, queries, train_index, query_index, max_queries=None, seed=DEFAULT_SEED):
    """Each query slice (validation or test split) -> nearest slice in the whole training split.

    Patients are split at patient level, so every query is a genuinely new brain.
    """
    q = torch.arange(len(queries))
    if max_queries is not None and max_queries < len(queries):
        q = torch.randperm(len(queries), generator=torch.Generator().manual_seed(seed))[:max_queries].sort().values

    joint = _join(train_index, query_index)
    cand = torch.arange(train_index.n, device=joint.device)
    start = time.time()
    res = joint.nearest(q.to(joint.device) + train_index.n, cand)
    if joint.device.type == "cuda":
        torch.cuda.synchronize()
    res.update(labels=queries.labels[q], nn_label_whole=train.labels[res["whole_arg"]],
               nn_label_masked=train.labels[res["masked_arg"]], seconds=time.time() - start,
               n_queries=len(q))
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
RUNG_ORDER = ("near_duplicate", "same_session", "different_day", "stranger_val", "stranger_test")
RUNG_ROLE = {"near_duplicate": "copy", "same_session": "copy", "different_day": "identity",
             "stranger_val": "calibration", "stranger_test": "false flags"}


def percentiles(v, ps=(5, 50, 95)):
    v = v.float()
    return [float(torch.quantile(v, p / 100)) for p in ps] if len(v) else [float("nan")] * len(ps)


def calibrate(rungs, threshold_pct):
    """Per-class, per-metric threshold: the given percentile of validation-stranger SSIM."""
    val = rungs["stranger_val"]
    return {cls: {metric: float(torch.quantile(val[metric][val["labels"] == label].float(), threshold_pct / 100))
                  for metric in ("whole", "masked")}
            for label, cls in enumerate(CLASS_NAMES)}


def summarise(rungs, thresholds, threshold_pct):
    """Print every rung with the share of its slices at or above the class threshold.

    For copy rungs that share is the detection rate (should be high); for test strangers
    it is the false-flag rate (should be near 100 - threshold_pct); for different-day
    pairs it shows whether the metric recognises the same patient across visits.
    """
    print(f"\nThreshold = {threshold_pct:g}th percentile of validation-stranger SSIM, per class and metric.")
    print(f"{'rung':<15} {'role':<12} {'class':<5} {'metric':<7} {'n':>6} {'p5':>7} {'p50':>7} {'p95':>7} "
          f"{'>= thr':>7}")
    for name in RUNG_ORDER:
        r = rungs[name]
        for label, cls in enumerate(CLASS_NAMES):
            sel = r["labels"] == label
            for metric in ("whole", "masked"):
                v = r[metric][sel]
                p5, p50, p95 = percentiles(v)
                share = float((v >= thresholds[cls][metric]).float().mean()) if len(v) else float("nan")
                print(f"{name:<15} {RUNG_ROLE[name]:<12} {cls:<5} {metric:<7} {len(v):>6} "
                      f"{p5:>7.4f} {p50:>7.4f} {p95:>7.4f} {share:>7.1%}")

    print(f"\n{'class':<5} {'metric':<7} {'threshold':>9}  test: nearest training slice has same class")
    test = rungs["stranger_test"]
    for label, cls in enumerate(CLASS_NAMES):
        sel = test["labels"] == label
        for metric in ("whole", "masked"):
            same = float((test[f"nn_label_{metric}"][sel] == label).float().mean())
            print(f"{cls:<5} {metric:<7} {thresholds[cls][metric]:>9.4f}  {same:.1%}")


def plot_ladder(rungs, thresholds, path, resolution, threshold_pct):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"near_duplicate": "#8172B2", "same_session": "#C44E52", "different_day": "#DD8452",
               "stranger_val": "#4C72B0", "stranger_test": "#55A868"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    for row, cls in enumerate(CLASS_NAMES):
        for col, metric in enumerate(("whole", "masked")):
            ax = axes[row, col]
            for name in RUNG_ORDER:
                r = rungs[name]
                v = r[metric][r["labels"] == row].numpy()
                if len(v):
                    ax.hist(v, bins=80, range=(0, 1), density=True, histtype="step", lw=1.5,
                            color=colours[name], label=f"{name.replace('_', ' ')} ({RUNG_ROLE[name]})")
            ax.axvline(thresholds[cls][metric], color="k", ls="--", lw=1,
                       label=f"threshold (val p{threshold_pct:g})")
            ax.set_title(f"{cls}, {metric}-image SSIM ({resolution} px)")
            ax.set_xlabel("nearest-neighbour SSIM (higher = more similar)")
            if col == 0:
                ax.set_ylabel("density")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Calibrate the memorisation-audit similarity ladder on real ADNI data.")
    p.add_argument("--resolution", type=int, default=64,
                   help="must match the resolution the generator will be audited at")
    p.add_argument("--max_queries", type=int, default=None,
                   help="subsample validation and test queries for the stranger rungs (quick runs)")
    p.add_argument("--threshold_pct", type=float, default=99.0)
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
    train, val, test = splits["train"], splits["val"], splits["test"]
    index = {name: SSIMIndex(ds.images, device, args.brain_threshold) for name, ds in splits.items()}
    print(f"Slices at {args.resolution} px: train {len(train)}, val {len(val)}, test {len(test)}")

    rungs = {}
    t0 = time.time()
    rungs["near_duplicate"] = near_duplicate(train, index["train"])
    print(f"near-duplicate: {len(rungs['near_duplicate']['whole'])} adjacent pairs ({time.time() - t0:.1f}s)")

    sessions = load_sessions(train.records)
    n_bad = sum(v is None for v in sessions.values())
    print(f"Parsed acquisition date and series for {len(sessions) - n_bad}/{len(sessions)} training scans")
    t0 = time.time()
    rungs.update(same_patient(train, index["train"], sessions))
    for name in ("same_session", "different_day"):
        r = rungs[name]
        print(f"{name.replace('_', ' ')}: {len(r['whole'])} query slices from "
              f"{len(r['subject_idx'].unique())} patients")
    pairs = visit_pairs(train, index["train"], sessions)
    print(f"Same-patient rungs and visit pairs: {time.time() - t0:.1f}s")
    summarise_pairs(pairs)

    for name, split in (("stranger_val", "val"), ("stranger_test", "test")):
        r = stranger(train, splits[split], index["train"], index[split], args.max_queries, args.seed)
        rungs[name] = r
        print(f"{name.replace('_', ' ')}: {r['n_queries']} queries x {len(train)} train slices in "
              f"{r['seconds']:.1f}s ({r['n_queries'] / r['seconds']:.1f} queries/s)")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    thresholds = calibrate(rungs, args.threshold_pct)
    summarise(rungs, thresholds, args.threshold_pct)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    tag = f"r{args.resolution}"
    torch.save(dict(rungs=rungs, visit_pairs=pairs, thresholds=thresholds),
               os.path.join(args.out_dir, f"ladder_{tag}.pt"))
    with open(os.path.join(args.out_dir, f"thresholds_{tag}.json"), "w") as f:
        json.dump(dict(resolution=args.resolution, percentile=args.threshold_pct,
                       calibrated_on="validation strangers", brain_threshold=args.brain_threshold,
                       seed=args.seed, queries=dict(val=rungs["stranger_val"]["n_queries"],
                                                    test=rungs["stranger_test"]["n_queries"]),
                       thresholds=thresholds), f, indent=2)
    fig_path = os.path.join(args.fig_dir, f"audit_ladder_{tag}.png")
    plot_ladder(rungs, thresholds, fig_path, args.resolution, args.threshold_pct)
    print(f"\nSaved {args.out_dir}/ladder_{tag}.pt, thresholds_{tag}.json and {fig_path}")


if __name__ == "__main__":
    main()