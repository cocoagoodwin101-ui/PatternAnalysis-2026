"""Run a trained generator and audit what it produces.

Loads generator weights saved by train.py, generates slices, and runs the memorisation
audit's generated rung (README Section 7): each generated slice is compared with its
nearest training slice by whole-image and brain-masked SSIM, against the thresholds that
audit.py calibrated on real validation patients.

The generator is unconditional, so slices have no class and the audit uses a pooled
threshold: the same percentile of validation-stranger SSIM, over both classes. Its false-
flag rate on real test patients (also pooled) is the reference a non-memorising generator
should match. Flag rates are reported with 95% Wilson intervals.

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

    train = get_adni_splits(resolution=config["resolution"], seed=calib["seed"], cache_dir=args.cache_dir,
                            num_workers=args.num_workers)["train"]
    train_index = SSIMIndex(train.images, device, calib["brain_threshold"])

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
        results[it] = dict(s, search_seconds=round(time.time() - t0, 1))
        for metric in ("whole", "masked"):
            r = s[metric]
            print(f"{it:>6} {metric:<7} {r['flagged']:>8} {r['flag_rate']:>7.2%} "
                  f"[{r['ci95'][0]:6.2%}, {r['ci95'][1]:6.2%}] "
                  + " ".join(f"{q:>7.4f}" for q in r["nn_ssim_p5_p50_p95"]))

    test = ladder["rungs"]["stranger_test"]
    print("\nFor comparison, real test patients' nearest-training SSIM (p5 / p50 / p95):")
    for metric in ("whole", "masked"):
        q = [float(torch.quantile(test[metric].float(), p)) for p in (0.05, 0.5, 0.95)]
        print(f"  {metric:<7} " + " / ".join(f"{x:.4f}" for x in q))

    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"memorisation_{tag}.json"), "w") as f:
        json.dump(dict(run=args.run_name, n_samples=args.n_samples, seed=args.seed,
                       percentile=calib["percentile"], reference=ref, snapshots=results), f, indent=2)
    title = f"{args.run_name}, iteration {it}, {args.n_samples} samples"
    plot_distributions(res, ladder, ref, os.path.join(args.fig_dir, f"{args.run_name}_audit.png"), title)
    plot_nearest(gen, train.images, res, os.path.join(args.fig_dir, f"{args.run_name}_nearest.png"), title,
                 args.seed)
    print(f"\nSaved {out_dir}/memorisation_{tag}.json, figures/{args.run_name}_audit.png and "
          f"figures/{args.run_name}_nearest.png (figures show iteration {it})")


if __name__ == "__main__":
    main()