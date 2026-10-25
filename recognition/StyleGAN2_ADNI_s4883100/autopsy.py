"""Failure autopsy: rule-selected flawed generations with their nearest training slices.

To avoid cherry-picking, every case is chosen by a fixed rule over all generated slices of
one snapshot (default 3,100), and the top k per rule are shown (no slice appears twice):

  near-copy          highest masked SSIM to the nearest training slice (memorisation risk);
                     taken from --copy_iter if given, e.g. the snapshot where the audit
                     found a copy-level slice
  least like real    lowest masked SSIM to every training slice (implausible anatomy)
  off-manifold       farthest outside the real test slices' Inception k-NN balls, measured
                     as the smallest ratio distance / ball radius over real slices; > 1
                     means outside every real ball (a precision failure)
  dark               lowest mean intensity over brain pixels
  dropped mode       real AD test slices whose Inception ball contains no generated slice
                     and that are farthest from any generated slice (a coverage failure),
                     shown above their nearest generated slice

Each generated case is shown above its nearest training slice with the masked SSIM, so
the memorisation check is part of the figure: values below the near-duplicate copy level
mean the slice is not a copy of that training slice.

Also saves two uncurated 8 x 8 grids from different seeds, the evidence that the
autopsy cases are exceptions rather than typical output.

Usage (GPU job on Rangpur, see autopsy.sh; needs audit.py outputs at the same resolution):
    python autopsy.py --run_name sg2_adni64 --iter 10000 --copy_iter 12500
Outputs: figures/<run>_iterNNNNNN_autopsy.png, figures/<run>_iterNNNNNN_uncurated_seedS.png
and outputs/eval/<run>/autopsy_iterNNNNNN.json.
"""

import argparse
import json
import os
import time

import torch
from torchvision.utils import save_image

from audit import SSIMIndex
from dataset import DEFAULT_CACHE, get_adni_splits
from predict import (audit_generated, build_generator, generate, inception_features, inception_model, knn_radii,
                     load_run, snapshot_iter)

CLASS_AD = 1


def load_snapshot(run_dir, it, device):
    config, snaps = load_run(run_dir)
    match = [p for p in snaps if snapshot_iter(p) == it]
    if not match:
        raise SystemExit(f"No snapshot at iteration {it} in {run_dir}; available {[snapshot_iter(p) for p in snaps]}")
    g = build_generator(config, device)
    g.load_state_dict(torch.load(match[0], map_location=device))
    return config, g


def brain_mean(images, threshold):
    """Mean intensity of brain pixels (above the background threshold) per image."""
    x = images.float()
    mask = (images > threshold).float()
    return (x * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1)


def pick(scores, k, taken, largest=True):
    """Top-k indices by score, skipping indices already used by an earlier rule."""
    order = torch.argsort(scores, descending=largest).tolist()
    out = [i for i in order if i not in taken][:k]
    taken.update(out)
    return out


def plot_autopsy(rows, path, title, copy_level):
    """One row per rule; each case is a pair of panels (case, then its neighbour) side by side.

    rows: list of ((case label, neighbour label), [(case image, neighbour image, case caption,
    neighbour caption)]).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = max(len(cases) for _, cases in rows)
    fig, axes = plt.subplots(len(rows), 2 * k, figsize=(1.75 * 2 * k + 0.6, 2.0 * len(rows) + 0.6), squeeze=False)
    for r, ((case_label, nb_label), cases) in enumerate(rows):
        for c in range(k):
            for half in range(2):
                ax = axes[r, 2 * c + half]
                ax.set_xticks([])
                ax.set_yticks([])
                if c >= len(cases):
                    ax.axis("off")
                    continue
                ax.imshow(cases[c][half][0].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
                ax.set_title(cases[c][2 + half], fontsize=7)
        axes[r, 0].set_ylabel(f"{case_label}\nvs {nb_label}", fontsize=8)
    fig.suptitle(f"{title}\nEach pair: case (left), its nearest neighbour (right). Masked SSIM copy level = "
                 f"{copy_level:.3f}; a generated slice below it is not a copy of its nearest training slice.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Rule-based failure autopsy with nearest-neighbour checks.")
    p.add_argument("--run_name", required=True)
    p.add_argument("--iter", type=int, required=True, help="snapshot iteration to autopsy")
    p.add_argument("--copy_iter", type=int, default=None, help="snapshot for the near-copy row (default: --iter)")
    p.add_argument("--n_samples", type=int, default=3100)
    p.add_argument("--k", type=int, default=3, help="cases per rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs_root", default=os.path.join(here, "outputs", "runs"))
    p.add_argument("--audit_dir", default=os.path.join(here, "outputs", "audit"))
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_root", default=os.path.join(here, "outputs", "eval"))
    p.add_argument("--fig_dir", default=os.path.join(here, "figures"))
    args = p.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.runs_root, args.run_name)
    config, G = load_snapshot(run_dir, args.iter, device)
    tag = f"r{config['resolution']}"
    with open(os.path.join(args.audit_dir, f"thresholds_{tag}.json")) as f:
        calib = json.load(f)
    ladder = torch.load(os.path.join(args.audit_dir, f"ladder_{tag}.pt"), weights_only=False)  # our own file
    copy_level = float(torch.quantile(ladder["rungs"]["near_duplicate"]["masked"].float(), 0.05))
    bt = calib["brain_threshold"]

    splits = get_adni_splits(resolution=config["resolution"], seed=calib["seed"], cache_dir=args.cache_dir,
                             num_workers=args.num_workers)
    train, test = splits["train"], splits["test"]
    train_index = SSIMIndex(train.images, device, bt)

    gen = generate(G, args.n_samples, config["latent_dim"], device, args.seed)
    nn = audit_generated(gen, train_index, device, bt)
    ssim, arg = nn["masked"], nn["masked_arg"]

    inception = inception_model(device)
    gen_f = inception_features(inception, gen, device).float().to(device)
    test_f = inception_features(inception, test.images, device).float().to(device)
    radius = knn_radii(test_f, 5)
    d = torch.cdist(gen_f, test_f)  # (generated, real)
    off_manifold = (d / radius[None, :]).min(dim=1).values.cpu()  # > 1: outside every real ball
    covered = (d <= radius[None, :]).any(dim=0).cpu()
    isolation = (d / radius[None, :]).min(dim=0).values.cpu()  # per real slice: nearest generated, in radii
    nearest_gen = d.argmin(dim=0).cpu()
    darkness = brain_mean(gen, bt).cpu()

    def gen_case(i, caption):
        return (gen[i], train.images[int(arg[i])], caption, f"nearest train {ssim[i]:.3f}")

    taken, rows, record = set(), [], {}

    # 1. Near-copy, possibly from another snapshot.
    if args.copy_iter is not None and args.copy_iter != args.iter:
        _, G_copy = load_snapshot(run_dir, args.copy_iter, device)
        gen_c = generate(G_copy, args.n_samples, config["latent_dim"], device, args.seed)
        nn_c = audit_generated(gen_c, train_index, device, bt)
        top = torch.argsort(nn_c["masked"], descending=True)[:args.k].tolist()
        cases = [(gen_c[i], train.images[int(nn_c["masked_arg"][i])], f"it {args.copy_iter}, SSIM {nn_c['masked'][i]:.3f}",
                  "nearest training slice") for i in top]
        record["near_copy"] = dict(iteration=args.copy_iter, masked_ssim=[float(nn_c["masked"][i]) for i in top])
    else:
        top = pick(ssim, args.k, taken)
        cases = [(gen[i], train.images[int(arg[i])], f"SSIM {ssim[i]:.3f}", "nearest training slice") for i in top]
        record["near_copy"] = dict(iteration=args.iter, masked_ssim=[float(ssim[i]) for i in top])
    rows.append((("near-copy (generated)", "nearest training"), cases))

    # 2-4. Rules over the main snapshot.
    for name, scores, largest, label, fmt in (
            ("least_like_real", ssim, False, "least like real", lambda i: f"max SSIM {ssim[i]:.3f}"),
            ("off_manifold", off_manifold, True, "off-manifold", lambda i: f"{off_manifold[i]:.2f} x radius"),
            ("dark", darkness, False, "dark", lambda i: f"brain mean {darkness[i]:.0f}")):
        idx = pick(scores, args.k, taken, largest=largest)
        rows.append(((f"{label} (generated)", "nearest training"), [gen_case(i, fmt(i)) for i in idx]))
        record[name] = [dict(index=i, score=float(scores[i]), nn_masked_ssim=float(ssim[i])) for i in idx]

    # 5. Dropped mode: real AD test slices no generated slice reaches, most isolated first.
    ad_uncovered = ((test.labels == CLASS_AD) & ~covered).nonzero().flatten()
    order = ad_uncovered[torch.argsort(isolation[ad_uncovered], descending=True)][:args.k].tolist()
    rows.append((("dropped mode (real AD test)", "nearest generated"),
                 [(test.images[j], gen[int(nearest_gen[j])], f"{isolation[j]:.2f} x radius", "nearest generated")
                  for j in order]))
    n_ad = int((test.labels == CLASS_AD).sum())
    n_cn_uncovered = int(((test.labels != CLASS_AD) & ~covered).sum())
    record["dropped_mode"] = dict(real_ad_uncovered=len(ad_uncovered), real_ad_total=n_ad,
                                  real_cn_uncovered=n_cn_uncovered, real_cn_total=len(test) - n_ad,
                                  isolation=[float(isolation[j]) for j in order])

    os.makedirs(args.fig_dir, exist_ok=True)
    stem = f"{args.run_name}_iter{args.iter:06d}"
    path = os.path.join(args.fig_dir, f"{stem}_autopsy.png")
    plot_autopsy(rows, path, f"Failure autopsy: {args.run_name}, iteration {args.iter}, top {args.k} of "
                 f"{args.n_samples} per rule", copy_level)

    grids = []
    for s in (args.seed + 1, args.seed + 2):
        g = generate(G, 64, config["latent_dim"], device, s)
        grid_path = os.path.join(args.fig_dir, f"{stem}_uncurated_seed{s}.png")
        save_image(g.float() / 255, grid_path, nrow=8)
        grids.append(grid_path)

    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"autopsy_iter{args.iter:06d}.json"), "w") as f:
        json.dump(dict(run=args.run_name, iteration=args.iter, n_samples=args.n_samples, seed=args.seed,
                       copy_level=copy_level, cases=record,
                       share_outside_all_real_balls=float((off_manifold > 1).float().mean())), f, indent=2)

    print(f"Copy level (masked SSIM): {copy_level:.4f}; near-copy SSIMs: {record['near_copy']['masked_ssim']}")
    for name in ("least_like_real", "off_manifold", "dark"):
        print(f"{name}: " + ", ".join(f"score {c['score']:.3f} (NN SSIM {c['nn_masked_ssim']:.3f})" for c in record[name]))
    print(f"Generated slices outside every real Inception ball: {(off_manifold > 1).float().mean():.1%}")
    dm = record["dropped_mode"]
    print(f"Real test slices with no generated slice in their ball: AD {dm['real_ad_uncovered']}/{dm['real_ad_total']} "
          f"({dm['real_ad_uncovered'] / dm['real_ad_total']:.1%}), CN {dm['real_cn_uncovered']}/{dm['real_cn_total']} "
          f"({dm['real_cn_uncovered'] / dm['real_cn_total']:.1%})")
    print(f"Saved {path}, {', '.join(os.path.basename(g) for g in grids)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()