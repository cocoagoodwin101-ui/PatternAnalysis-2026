"""Run a trained generator and audit what it produces.

Loads generator weights saved by train.py, generates slices, and runs the memorisation
audit's generated rung (README Section 7): each generated slice is compared with its
nearest training slice by whole-image and brain-masked SSIM, against the thresholds that
audit.py calibrated on real validation patients.

The generator is unconditional, so slices have no class and the audit uses a pooled
threshold: the same percentile of validation-stranger SSIM, over both classes. Its false-
flag rate on real test patients (also pooled) is the reference a non-memorising generator
should match. Flag rates are reported with 95% Wilson intervals.

Two checks separate memorisation from reduced diversity when the flag rate is high:
  - neighbour concentration: how many distinct training slices (and patients) are the
    nearest neighbours of the generated set, against the same count for real test
    patients. Flagged samples that keep landing on a few training slices or patients point
    to memorisation; flags spread over many point to concentration on typical slices.
  - copy level: how many generated slices reach the near-duplicate rung's 5th percentile,
    i.e. are as close to a training slice as an adjacent slice of the same scan.
The neighbour-concentration comparison is confounded: test queries come in runs of 20
adjacent slices per scan, so they share neighbours more than independent samples do.

Matched-reference test (after Meehan et al., 2020, "A non-parametric test to detect data-
copying in generative models"): is each generated slice closer to the training patients
than to equally many unseen patients? References are the validation split and several
random subsets of training patients with the same number of patients per class and the
same number of slices. If the generator fits features specific to its training patients,
generated slices are closer to the training subsets; if it only produces typical slices,
they are equally close to both. Real test slices are run through the same comparison as a
control, since they are unseen by both references and should show no preference.

The reported excess is generated minus control, paired by subset (both use the same
references). Its 95% interval comes from a bootstrap that resamples test patients (their
slices are correlated, so patients, not slices, are the independent units) and generated
samples (independent draws), after averaging each query's difference over the subsets.
The single validation split is not resampled; pairing removes its shared bias but not
all of its sampling variability.

Fidelity and coverage: FID (Heusel et al., 2017) and improved precision and recall
(Kynkaanniemi et al., 2019), all on Inception-v3 pool features of generated slices
against the held-out test slices, at equal sample sizes. FID is lower for distributions
that match better; precision is the share of generated slices that fall within the real
data's feature manifold (realism), recall the share of real slices that fall within the
generated manifold (coverage). The validation split scored against the test split gives a
real-vs-real reference: what a perfect generator would score at these sample sizes. Uses
the standard FID Inception weights from the pytorch-fid package.

Usage (GPU job on Rangpur, see predict.sh; run audit.py at the same resolution first):
    python predict.py --run_name wgan_adni64                    # final snapshot
    python predict.py --run_name wgan_adni64 --snapshots all    # flag rate across training

Outputs: results in outputs/eval/<run_name>/ (gitignored), and two figures in figures/:
<run_name>_audit.png (similarity distributions) and <run_name>_nearest.png (generated
slices beside their nearest training slices: the most similar ones and a random set).
"""

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch

from audit import SSIMIndex, _join
from dataset import DEFAULT_CACHE, get_adni_splits
from modules import WGANGenerator


# --------------------------------------------------------------------------- #
# Loading and generation
# --------------------------------------------------------------------------- #
def load_run(run_dir):
    """The run's training config and its generator snapshots, oldest first."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshots", "generator_*.pt")))
    if not snaps:
        raise SystemExit(f"No generator snapshots in {run_dir}/snapshots")
    return config, snaps


def snapshot_iter(path):
    return int(re.search(r"generator_(\d+)\.pt$", path).group(1))


def build_generator(config, device):
    if config["model"] == "wgan-gp":
        return WGANGenerator(config["latent_dim"], config["resolution"]).to(device)
    raise ValueError(f"Unknown model {config['model']}")


@torch.no_grad()
def generate(generator, n, latent_dim, device, seed, batch=500):
    """n generated slices as uint8 (n, 1, R, R), quantised the same way as the real data."""
    generator.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, n, batch):
        z = torch.randn(min(batch, n - s), latent_dim, device=device, generator=g)
        x = generator(z).clamp(-1, 1)
        out.append(((x + 1) * 127.5).round().to(torch.uint8))
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def pooled_reference(ladder, pct):
    """Pooled threshold from validation strangers, and the pooled test false-flag rate."""
    rungs = ladder["rungs"]
    ref = {}
    for metric in ("whole", "masked"):
        t = float(torch.quantile(rungs["stranger_val"][metric].float(), pct / 100))
        ref[metric] = dict(threshold=t, test_flag_rate=float((rungs["stranger_test"][metric] >= t).float().mean()))
    return ref


def audit_generated(gen_images, train_index, device, brain_threshold):
    """Nearest training slice for every generated slice (whole and masked SSIM)."""
    gen_index = SSIMIndex(gen_images, device, brain_threshold)
    joint = _join(train_index, gen_index)
    q = torch.arange(gen_index.n, device=device) + train_index.n
    return joint.nearest(q, torch.arange(train_index.n, device=device))


def summarise(res, ref, n):
    out = {}
    for metric in ("whole", "masked"):
        v = res[metric]
        k = int((v >= ref[metric]["threshold"]).sum())
        lo, hi = wilson(k, n)
        out[metric] = dict(flagged=k, flag_rate=k / n, ci95=[lo, hi],
                           nn_ssim_p5_p50_p95=[float(torch.quantile(v.float(), p)) for p in (0.05, 0.5, 0.95)],
                           threshold=ref[metric]["threshold"], test_flag_rate=ref[metric]["test_flag_rate"])
    return out


def concentration(nn_idx, flagged, subject_idx):
    """How concentrated a query set's nearest training neighbours are, overall and among flags."""
    nn_idx, flagged = nn_idx.cpu(), flagged.cpu()
    counts = torch.bincount(nn_idx)
    f_idx = nn_idx[flagged]
    f_counts = torch.bincount(f_idx) if len(f_idx) else torch.zeros(1, dtype=torch.long)
    return dict(queries=len(nn_idx), distinct_nn=int((counts > 0).sum()), max_repeat=int(counts.max()),
                flagged=int(flagged.sum()), flagged_distinct_nn=int((f_counts > 0).sum()) if len(f_idx) else 0,
                flagged_distinct_patients=int(subject_idx[f_idx].unique().numel()) if len(f_idx) else 0,
                flagged_max_repeat=int(f_counts.max()) if len(f_idx) else 0)


def memorisation_checks(res, ladder, ref, subject_idx):
    """Neighbour concentration (generated and test strangers) and copy-level counts, per metric."""
    rungs = ladder["rungs"]
    out = {}
    for metric in ("whole", "masked"):
        t = ref[metric]["threshold"]
        copy_level = float(torch.quantile(rungs["near_duplicate"][metric].float(), 0.05))
        test = rungs["stranger_test"]
        out[metric] = dict(
            generated=concentration(res[f"{metric}_arg"], res[metric] >= t, subject_idx),
            test_strangers=concentration(test[f"{metric}_arg"], test[metric] >= t, subject_idx),
            copy_level=copy_level, copy_level_count=int((res[metric] >= copy_level).sum()),
            max_nn_ssim=float(res[metric].max()))
    return out


def print_checks(it, checks):
    for metric in ("whole", "masked"):
        c = checks[metric]
        for name in ("generated", "test_strangers"):
            g = c[name]
            label = f"{it:>6}" if name == "generated" else f"{'':>6}"
            print(f"{label} {metric:<7} {name.replace('_', ' '):<15} {g['queries']:>5} {g['distinct_nn']:>9} "
                  f"{g['max_repeat']:>7} {g['flagged']:>8} {g['flagged_distinct_nn']:>10} "
                  f"{g['flagged_distinct_patients']:>9} {g['flagged_max_repeat']:>8}")
        print(f"{'':>6} {metric:<7} copy level (near-duplicate p5 = {c['copy_level']:.4f}): "
              f"{c['copy_level_count']} generated slices at or above; max generated NN SSIM {c['max_nn_ssim']:.4f}")


def matched_subsets(train, val, k, seed):
    """k random sets of training slices matched to the validation split.

    Each set has the validation split's number of patients per class; if those patients
    have more slices than the validation split, slices are subsampled to the same count.
    """
    n_val = len(val)
    per_class = {lab: int(val.subject_idx[val.labels == lab].unique().numel()) for lab in (0, 1)}
    g = torch.Generator().manual_seed(seed)
    subsets = []
    for _ in range(k):
        picks = []
        for lab, n in per_class.items():
            patients = train.subject_idx[train.labels == lab].unique()
            picks.append(patients[torch.randperm(len(patients), generator=g)[:n]])
        idx = torch.isin(train.subject_idx, torch.cat(picks)).nonzero().flatten()
        if len(idx) > n_val:
            idx = idx[torch.randperm(len(idx), generator=g)[:n_val]].sort().values
        subsets.append(idx)
    return subsets


def reference_gap(joint, queries, subsets, val_cand):
    """Nearest-neighbour SSIM of each query to the validation split and to each training subset.

    Returns per metric: median similarity to the validation split and (averaged over
    subsets) to the training subsets, the mean paired difference (training minus
    validation), and the share of queries closer to the training subset, each with its
    range over subsets.
    """
    device = joint.device
    to_val = joint.nearest(queries, val_cand)
    to_train = [joint.nearest(queries, c.to(device)) for c in subsets]
    out = {}
    for metric in ("whole", "masked"):
        v = to_val[metric]
        diffs = [t[metric] - v for t in to_train]
        mean_diff = [float(d.mean()) for d in diffs]
        closer = [float((d > 0).float().mean()) for d in diffs]
        out[metric] = dict(val_p50=float(v.median()),
                           train_p50=sum(float(t[metric].median()) for t in to_train) / len(to_train),
                           diff_mean=sum(mean_diff) / len(mean_diff), diff_range=[min(mean_diff), max(mean_diff)],
                           closer_to_train=sum(closer) / len(closer), closer_range=[min(closer), max(closer)],
                           diff_per_subset=mean_diff, closer_per_subset=closer,
                           per_query=torch.stack(diffs, dim=1).cpu())  # (queries, subsets); not saved
    return out


def paired_excess(gen_gap, ctrl_gap, ctrl_groups, n_boot=2000, seed=0):
    """Generated minus control difference, per subset and with a patient-level bootstrap interval."""
    g = torch.Generator().manual_seed(seed)
    groups = ctrl_groups.cpu()
    patients, inverse = groups.unique(return_inverse=True)
    out = {}
    for metric in ("whole", "masked"):
        gq = gen_gap[metric]["per_query"].double()
        cq = ctrl_gap[metric]["per_query"].double()
        per_subset = (gq.mean(0) - cq.mean(0)).tolist()
        gq, cq = gq.mean(1), cq.mean(1)  # average each query over the subsets
        # Control: resample patients with replacement, weighting by their slices.
        p_sum = torch.zeros(len(patients), dtype=torch.float64).index_add_(0, inverse, cq)
        p_cnt = torch.zeros(len(patients), dtype=torch.float64).index_add_(0, inverse, torch.ones_like(cq))
        pick = torch.randint(len(patients), (n_boot, len(patients)), generator=g)
        ctrl_b = p_sum[pick].sum(1) / p_cnt[pick].sum(1)
        # Generated: independent samples, resampled directly.
        gen_b = gq[torch.randint(len(gq), (n_boot, len(gq)), generator=g)].mean(1)
        boot = gen_b - ctrl_b
        out[metric] = dict(excess=float(gq.mean() - cq.mean()),
                           ci95=[float(torch.quantile(boot, 0.025)), float(torch.quantile(boot, 0.975))],
                           share_boot_le_0=float((boot <= 0).double().mean()),
                           per_subset=per_subset, subsets_positive=int(sum(x > 0 for x in per_subset)),
                           control_ci95=[float(torch.quantile(ctrl_b, 0.025)), float(torch.quantile(ctrl_b, 0.975))])
    return out


def strip_per_query(gap):
    return {m: {k: v for k, v in d.items() if k != "per_query"} for m, d in gap.items()}


def print_gap(label, metric, g):
    print(f"{label:>14} {metric:<7} {g['train_p50']:>9.4f} {g['val_p50']:>9.4f} "
          f"{g['diff_mean']:>+9.4f} [{g['diff_range'][0]:+.4f}, {g['diff_range'][1]:+.4f}] "
          f"{g['closer_to_train']:>7.1%} [{g['closer_range'][0]:.1%}, {g['closer_range'][1]:.1%}]")


# --------------------------------------------------------------------------- #
# Fidelity and coverage
# --------------------------------------------------------------------------- #
def inception_model(device):
    """The Inception-v3 used for standard FID (pytorch-fid), returning 2048-d pool features."""
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as e:
        raise SystemExit("pytorch-fid is not installed: pip install pytorch-fid (or run with --no_fidelity)") from e
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()


@torch.no_grad()
def inception_features(model, images_uint8, device, batch=250):
    """(N, 2048) features. Grey slices are repeated to 3 channels; the model resizes to 299 px."""
    feats = []
    for s in range(0, len(images_uint8), batch):
        x = images_uint8[s:s + batch].to(device).float().div(255).repeat(1, 3, 1, 1)
        feats.append(model(x)[0].flatten(1).double().cpu())
    return torch.cat(feats)


def fid_from_features(a, b):
    from pytorch_fid.fid_score import calculate_frechet_distance
    a, b = a.numpy(), b.numpy()
    return float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)))


def precision_recall(real, fake, k=3, device="cpu"):
    """Improved precision and recall: k-nearest-neighbour balls around each set's samples."""
    real, fake = real.float().to(device), fake.float().to(device)

    def radii(x):
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.kthvalue(k, dim=1).values

    r_real, r_fake = radii(real), radii(fake)
    d = torch.cdist(fake, real)  # (fake, real)
    precision = (d <= r_real[None, :]).any(dim=1).float().mean()
    recall = (d.T <= r_fake[None, :]).any(dim=1).float().mean()
    return float(precision), float(recall)


def fidelity(feats, test_feats, device):
    p, r = precision_recall(test_feats, feats, device=device)
    return dict(fid=fid_from_features(feats, test_feats), precision=p, recall=r, n=len(feats), n_test=len(test_feats))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_distributions(res, ladder, ref, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rungs = ladder["rungs"]
    series = [("near_duplicate", "near duplicate (copy)", "#8172B2"),
              ("same_session", "same session (copy)", "#C44E52"),
              ("stranger_test", "stranger test (new real brains)", "#55A868")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("whole", "masked")):
        for key, label, colour in series:
            ax.hist(rungs[key][metric].numpy(), bins=80, range=(0, 1), density=True, histtype="step",
                    lw=1.5, color=colour, label=label)
        ax.hist(res[metric].numpy(), bins=80, range=(0, 1), density=True, histtype="stepfilled",
                alpha=0.35, color="#4C72B0", label="generated")
        ax.axvline(ref[metric]["threshold"], color="k", ls="--", lw=1, label="pooled threshold")
        ax.set_title(f"{metric}-image SSIM to nearest training slice")
        ax.set_xlabel("nearest-neighbour SSIM (higher = more similar)")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nearest(gen_images, train_images, res, path, title, seed, k=8):
    """Generated slices above their nearest training slices: the k most similar, then k random."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v, arg = res["masked"], res["masked_arg"]
    top = torch.argsort(v, descending=True)[:k]
    rand = torch.randperm(len(v), generator=torch.Generator().manual_seed(seed))[:k]
    fig, axes = plt.subplots(4, k, figsize=(1.6 * k, 7.2))
    for block, (idx, name) in enumerate(((top, "most similar"), (rand, "random"))):
        for j, i in enumerate(idx.tolist()):
            for r, img in enumerate((gen_images[i, 0], train_images[int(arg[i]), 0])):
                ax = axes[2 * block + r, j]
                ax.imshow(img.cpu().numpy(), cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{v[i]:.3f}", fontsize=8)
        axes[2 * block, 0].set_ylabel(f"generated\n({name})", fontsize=8)
        axes[2 * block + 1, 0].set_ylabel("nearest\ntraining", fontsize=8)
    fig.suptitle(f"{title}\n(number = masked SSIM to the nearest training slice)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Generate slices from a trained model and audit them for memorisation.")
    p.add_argument("--run_name", required=True)
    p.add_argument("--runs_root", default=os.path.join(here, "outputs", "runs"))
    p.add_argument("--snapshots", choices=["last", "all"], default="last")
    p.add_argument("--n_samples", type=int, default=3100, help="default matches the number of test-stranger slices")
    p.add_argument("--seed", type=int, default=0, help="latent sampling seed")
    p.add_argument("--no_fidelity", action="store_true", help="skip FID and precision/recall")
    p.add_argument("--ref_subsets", type=int, default=5,
                   help="random training-patient subsets for the matched-reference test")
    p.add_argument("--audit_dir", default=os.path.join(here, "outputs", "audit"))
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_root", default=os.path.join(here, "outputs", "eval"))
    p.add_argument("--fig_dir", default=os.path.join(here, "figures"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, snaps = load_run(os.path.join(args.runs_root, args.run_name))
    tag = f"r{config['resolution']}"
    with open(os.path.join(args.audit_dir, f"thresholds_{tag}.json")) as f:
        calib = json.load(f)
    ladder = torch.load(os.path.join(args.audit_dir, f"ladder_{tag}.pt"), weights_only=False)  # our own file
    ref = pooled_reference(ladder, calib["percentile"])
    print(f"Run {args.run_name}: {config['model']} trained on {config['dataset']} at {config['resolution']} px")
    print(f"Pooled thresholds ({calib['percentile']:g}th percentile of validation strangers): "
          f"whole {ref['whole']['threshold']:.4f}, masked {ref['masked']['threshold']:.4f}")
    print(f"Real test patients flagged at these thresholds: whole {ref['whole']['test_flag_rate']:.2%}, "
          f"masked {ref['masked']['test_flag_rate']:.2%}")

    splits = get_adni_splits(resolution=config["resolution"], seed=calib["seed"], cache_dir=args.cache_dir,
                             num_workers=args.num_workers)
    train, val, test_split = splits["train"], splits["val"], splits["test"]
    train_index = SSIMIndex(train.images, device, calib["brain_threshold"])

    # Matched-reference test: references are the validation split (appended after the
    # training slices in one joint index) and random training subsets matched to it.
    refs = _join(train_index, SSIMIndex(val.images, device, calib["brain_threshold"]))
    val_cand = torch.arange(train_index.n, refs.n, device=device)
    subsets = matched_subsets(train, val, args.ref_subsets, calib["seed"])
    print(f"Matched references: validation {len(val)} slices; {args.ref_subsets} training subsets of "
          f"{min(len(c) for c in subsets)}-{max(len(c) for c in subsets)} slices from the same patient numbers")
    fid_ref = None
    if not args.no_fidelity:
        inception = inception_model(device)
        test_feats = inception_features(inception, test_split.images, device)
        fid_ref = fidelity(inception_features(inception, val.images, device), test_feats, device)
        print(f"Fidelity reference (real validation vs real test slices): FID {fid_ref['fid']:.2f}, "
              f"precision {fid_ref['precision']:.3f}, recall {fid_ref['recall']:.3f}")
    joint = _join(refs, SSIMIndex(test_split.images, device, calib["brain_threshold"]))
    test_gap = reference_gap(joint, torch.arange(refs.n, joint.n, device=device), subsets, val_cand)
    del joint

    chosen = snaps if args.snapshots == "all" else snaps[-1:]
    generator = build_generator(config, device)
    results = {}
    print(f"\n{'iter':>6} {'metric':<7} {'flagged':>8} {'rate':>7} {'95% CI':>17} "
          f"{'NN p5':>7} {'NN p50':>7} {'NN p95':>7}")
    for path in chosen:
        it = snapshot_iter(path)
        generator.load_state_dict(torch.load(path, map_location=device))
        gen = generate(generator, args.n_samples, config["latent_dim"], device, args.seed)
        t0 = time.time()
        res = audit_generated(gen, train_index, device, calib["brain_threshold"])
        s = summarise(res, ref, args.n_samples)
        checks = memorisation_checks(res, ladder, ref, train.subject_idx)
        joint = _join(refs, SSIMIndex(gen, device, calib["brain_threshold"]))
        gap = reference_gap(joint, torch.arange(refs.n, joint.n, device=device), subsets, val_cand)
        del joint
        excess = paired_excess(gap, test_gap, test_split.subject_idx, seed=args.seed)
        fid_res = None if args.no_fidelity else fidelity(inception_features(inception, gen, device), test_feats, device)
        results[it] = dict(s, checks=checks, matched_reference=strip_per_query(gap), paired_excess=excess,
                           fidelity=fid_res, search_seconds=round(time.time() - t0, 1))
        for metric in ("whole", "masked"):
            r = s[metric]
            print(f"{it:>6} {metric:<7} {r['flagged']:>8} {r['flag_rate']:>7.2%} "
                  f"[{r['ci95'][0]:6.2%}, {r['ci95'][1]:6.2%}] "
                  + " ".join(f"{q:>7.4f}" for q in r["nn_ssim_p5_p50_p95"]))

    if fid_ref is not None:
        print(f"\nFidelity and coverage against held-out test slices (Inception-v3 features, "
              f"{args.n_samples} generated vs {fid_ref['n_test']} test)")
        print(f"{'set':>14} {'FID':>8} {'precision':>10} {'recall':>8}")
        print(f"{'real val':>14} {fid_ref['fid']:>8.2f} {fid_ref['precision']:>10.3f} {fid_ref['recall']:>8.3f}"
              f"   <- real-vs-real reference ({fid_ref['n']} validation slices)")
        for it_k, r in results.items():
            f = r["fidelity"]
            print(f"{'gen ' + str(it_k):>14} {f['fid']:>8.2f} {f['precision']:>10.3f} {f['recall']:>8.3f}")
        print("(lower FID = closer distribution; precision = realism; recall = coverage of the real data)")

    print("\nMatched-reference test: nearest SSIM to training subsets vs to the validation split "
          "(equal patients and slices)")
    print(f"{'query set':>14} {'metric':<7} {'to train':>9} {'to val':>9} {'mean diff':>9} {'range':>17} "
          f"{'closer':>7} {'range':>15}")
    for metric in ("whole", "masked"):
        print_gap("real test", metric, test_gap[metric])
    for it_k, r in results.items():
        for metric in ("whole", "masked"):
            print_gap(f"gen {it_k}", metric, r["matched_reference"][metric])
    print("(to train / to val = median nearest SSIM; mean diff = mean per-slice difference, train minus val; "
          "closer = share of slices closer to the training subset; ranges over subsets)")

    first = next(iter(results.values()))["paired_excess"]
    print("\nPaired excess over the real-test control (generated mean diff minus control mean diff)")
    for metric in ("whole", "masked"):
        lo, hi = first[metric]["control_ci95"]
        print(f"  control {metric:<7} mean diff {test_gap[metric]['diff_mean']:+.4f}, "
              f"95% CI over test patients [{lo:+.4f}, {hi:+.4f}]")
    print(f"{'iter':>6} {'metric':<7} {'excess':>8} {'95% CI':>19} {'P(<=0)':>7} "
          f"{'per-subset range':>19} {'positive':>8}")
    for it_k, r in results.items():
        for metric in ("whole", "masked"):
            e = r["paired_excess"][metric]
            print(f"{it_k:>6} {metric:<7} {e['excess']:>+8.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
                  f"{e['share_boot_le_0']:>7.3f} [{min(e['per_subset']):+.4f}, {max(e['per_subset']):+.4f}] "
                  f"{e['subsets_positive']:>4}/{len(e['per_subset'])}")
    print("(P(<=0) = share of bootstrap replicates with no excess; positive = subsets where generated exceeds control)")

    print("\nNeighbour concentration (same number of queries; NN = nearest training slice)")
    print(f"{'iter':>6} {'metric':<7} {'query set':<15} {'n':>5} {'distinct':>9} {'max rep':>7} "
          f"{'flagged':>8} {'flag dist':>10} {'flag pat':>9} {'flag rep':>8}")
    for it_k, r in results.items():
        print_checks(it_k, r["checks"])

    test = ladder["rungs"]["stranger_test"]
    if args.n_samples != len(test["masked"]):
        print(f"Note: {args.n_samples} generated vs {len(test['masked'])} test queries; distinct-neighbour "
              f"counts are only comparable at equal numbers of queries")
    print("\nFor comparison, real test patients' nearest-training SSIM (p5 / p50 / p95):")
    for metric in ("whole", "masked"):
        q = [float(torch.quantile(test[metric].float(), p)) for p in (0.05, 0.5, 0.95)]
        print(f"  {metric:<7} " + " / ".join(f"{x:.4f}" for x in q))

    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"memorisation_{tag}.json"), "w") as f:
        json.dump(dict(run=args.run_name, n_samples=args.n_samples, seed=args.seed,
                       percentile=calib["percentile"], reference=ref, matched_reference_test_control=strip_per_query(test_gap),
                       fidelity_reference=fid_ref,
                       snapshots=results), f, indent=2)
    title = f"{args.run_name}, iteration {it}, {args.n_samples} samples"
    plot_distributions(res, ladder, ref, os.path.join(args.fig_dir, f"{args.run_name}_audit.png"), title)
    plot_nearest(gen, train.images, res, os.path.join(args.fig_dir, f"{args.run_name}_nearest.png"), title,
                 args.seed)
    print(f"\nSaved {out_dir}/memorisation_{tag}.json, figures/{args.run_name}_audit.png and "
          f"figures/{args.run_name}_nearest.png (figures show iteration {it})")


if __name__ == "__main__":
    main()