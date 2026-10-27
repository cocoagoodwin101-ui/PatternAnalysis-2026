"""t-SNE embedding of real and generated ADNI slices (spec requirement for style-based models).

All slices are embedded together, so every panel shares one 2D map:
  - real held-out test slices coloured by ground-truth class (CN / AD),
  - the same real slices coloured by slice position within their scan (0-19),
  - one panel per run: its generated slices over the real slices in grey.

Features are the 2048-d Inception-v3 pool features used for FID (predict.py), reduced to 50
dimensions with PCA and then embedded with t-SNE. t-SNE preserves local neighbourhoods,
not global distances: overlapping or neighbouring points are meaningful, the distance
between separate clusters and cluster sizes are not.

Class check: the generators are unconditional, so generated slices have no label. Each is
assigned the majority class of its 10 nearest real test slices in Inception space, giving
the share of the generated cohort that looks AD-like. The same vote applied to real
validation slices (whose labels are known) shows how much class information these
features carry at all; if that accuracy is near chance, the generated class shares mean
little. For a class-conditional run, the vote is reported separately for CN-labelled and
AD-labelled slices, and its panel colours generated slices by their requested class.

Usage (GPU job on Rangpur, see embed.sh):
    python embed.py --runs wgan_adni64:10000 sg2_adni64:10000
Outputs: figures/<tag>.png and outputs/eval/<tag>.json.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from dataset import CLASS_NAMES, DEFAULT_CACHE, DEFAULT_SEED, get_adni_splits
from predict import (build_generator, class_labels, generate, inception_features, inception_model, load_run,
                     snapshot_iter)


def knn_vote(query, ref, ref_labels, k=10, device="cpu", chunk=1024):
    """Majority class among each query's k nearest reference slices (Euclidean, Inception space)."""
    ref = ref.float().to(device)
    labels = ref_labels.to(device)
    votes = []
    for s in range(0, len(query), chunk):
        d = torch.cdist(query[s:s + chunk].float().to(device), ref)
        idx = d.topk(k, largest=False).indices
        votes.append((labels[idx].float().mean(1) > 0.5).long().cpu())
    return torch.cat(votes)


def balanced_accuracy(pred, true):
    return float(np.mean([(pred[true == c] == c).float().mean() for c in true.unique()]))


def embed(features, seed, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = features.numpy().astype(np.float64)
    x = PCA(n_components=50, random_state=seed).fit_transform(x)
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed).fit_transform(x)


def plot(xy, groups, real_labels, real_pos, runs, path, gen_labels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = groups == 0
    n_panels = 2 + len(runs)
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    axes = axes.ravel()
    kw = dict(s=3, linewidths=0, rasterized=True)

    ax = axes[0]
    for c, colour in zip((0, 1), ("#4C72B0", "#C44E52")):
        m = real.copy()
        m[real] = real_labels == c
        ax.scatter(xy[m, 0], xy[m, 1], c=colour, label=f"real {CLASS_NAMES[c]}", alpha=0.6, **kw)
    ax.set_title("Real test slices by class")
    ax.legend(markerscale=4, fontsize=8)

    ax = axes[1]
    sc = ax.scatter(xy[real, 0], xy[real, 1], c=real_pos, cmap="viridis", alpha=0.7, **kw)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label("slice position in scan", fontsize=8)
    ax.set_title("Real test slices by slice position")

    for i, (name, _) in enumerate(runs, start=1):
        ax = axes[1 + i]
        ax.scatter(xy[real, 0], xy[real, 1], c="#BBBBBB", label="real test", alpha=0.5, **kw)
        g = groups == i
        if gen_labels[i - 1] is None:
            ax.scatter(xy[g, 0], xy[g, 1], c="#DD8452", label="generated", alpha=0.6, **kw)
        else:
            lab = gen_labels[i - 1].numpy()
            pts = xy[g]
            for c, colour in zip((0, 1), ("#4C72B0", "#C44E52")):
                ax.scatter(pts[lab == c, 0], pts[lab == c, 1], c=colour, label=f"generated {CLASS_NAMES[c]}",
                           alpha=0.6, **kw)
        ax.set_title(name)
        ax.legend(markerscale=4, fontsize=8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Joint t-SNE embedding of real and generated slices.")
    p.add_argument("--runs", nargs="+", required=True, help="run_name or run_name:iteration (default: last snapshot)")
    p.add_argument("--n_generated", type=int, default=1500, help="generated slices per run")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--k", type=int, default=10, help="neighbours for the class vote")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="embedding_tsne")
    p.add_argument("--runs_root", default=os.path.join(here, "outputs", "runs"))
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_root", default=os.path.join(here, "outputs", "eval"))
    p.add_argument("--fig_dir", default=os.path.join(here, "figures"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    inception = inception_model(device)

    specs, resolution = [], None
    for spec in args.runs:
        name, _, it = spec.partition(":")
        config, snaps = load_run(os.path.join(args.runs_root, name))
        if it:
            snaps = [s for s in snaps if snapshot_iter(s) == int(it)]
            if not snaps:
                raise SystemExit(f"{name}: no snapshot at iteration {it}")
        if resolution not in (None, config["resolution"]):
            raise SystemExit("all runs must share one resolution")
        resolution = config["resolution"]
        specs.append((name, config, snaps[-1]))

    splits = get_adni_splits(resolution=resolution, seed=DEFAULT_SEED, cache_dir=args.cache_dir,
                             num_workers=args.num_workers)
    runs, gen_feats, gen_labels = [], [], []
    for name, config, snap in specs:
        generator = build_generator(config, device)
        generator.load_state_dict(torch.load(snap, map_location=device))
        labels = class_labels(config, args.n_generated, splits["train"].labels, args.seed)
        images = generate(generator, args.n_generated, config["latent_dim"], device, args.seed, labels=labels)
        gen_feats.append(inception_features(inception, images, device))
        gen_labels.append(labels)
        runs.append((f"{name}, iteration {snapshot_iter(snap)}", name))
        print(f"{runs[-1][0]}: {args.n_generated} slices generated" + (" (class-conditional)" if labels is not None else ""))

    test, val = splits["test"], splits["val"]
    test_feats = inception_features(inception, test.images, device)
    val_feats = inception_features(inception, val.images, device)

    # How much class information do these features carry? Real validation slices voted by test slices.
    val_pred = knn_vote(val_feats, test_feats, test.labels, args.k, device)
    val_bacc = balanced_accuracy(val_pred, val.labels)
    real_ad = float(test.labels.float().mean())
    print(f"\nClass check ({args.k}-NN vote against {len(test)} real test slices, Inception features)")
    print(f"  real validation slices: balanced accuracy {val_bacc:.1%} (chance 50%); "
          f"AD share predicted {float(val_pred.float().mean()):.1%}, true {float(val.labels.float().mean()):.1%}")
    print(f"  real test slices: AD share {real_ad:.1%}")
    class_shares = {}
    for (label, _), f, lab in zip(runs, gen_feats, gen_labels):
        votes = knn_vote(f, test_feats, test.labels, args.k, device)
        class_shares[label] = float(votes.float().mean())
        print(f"  {label}: AD-like share {class_shares[label]:.1%}")
        if lab is not None:
            for c in (0, 1):
                share = float(votes[lab == c].float().mean())
                class_shares[f"{label}, {CLASS_NAMES[c]}-labelled"] = share
                print(f"    {CLASS_NAMES[c]}-labelled slices: AD-like share {share:.1%}")

    feats = torch.cat([test_feats] + gen_feats)
    groups = np.concatenate([np.zeros(len(test_feats), int)] + [np.full(len(f), i + 1) for i, f in enumerate(gen_feats)])
    print(f"\nEmbedding {len(feats)} slices (PCA 50 -> t-SNE, perplexity {args.perplexity:g})...", flush=True)
    xy = embed(feats, args.seed, args.perplexity)

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)
    path = os.path.join(args.fig_dir, f"{args.tag}.png")
    plot(xy, groups, test.labels.numpy(), test.slice_pos.numpy(), runs, path, gen_labels,
         f"t-SNE of Inception features: {len(test)} real test slices and {args.n_generated} generated per run "
         f"(local neighbourhoods are meaningful; distances between clusters are not)")
    with open(os.path.join(args.out_root, f"{args.tag}.json"), "w") as f:
        json.dump(dict(runs=[r[0] for r in runs], n_generated=args.n_generated, k=args.k,
                       perplexity=args.perplexity, seed=args.seed, real_test_ad_share=real_ad,
                       val_knn_balanced_accuracy=val_bacc, generated_ad_like_share=class_shares), f, indent=2)
    print(f"Saved {path} and outputs/eval/{args.tag}.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()