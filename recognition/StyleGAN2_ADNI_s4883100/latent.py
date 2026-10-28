"""Latent-space analysis: interpolation and inversion (spec 2.4, memorisation audit).

1. Interpolation. Pairs of latents are interpolated in small steps (StyleGAN2: linearly in
   the style space W, where interpolation is smoothest; WGAN-GP: spherically in z). Every
   frame is matched to its nearest training slice by masked SSIM. A generator that
   generalises morphs smoothly, with nearest-neighbour similarity staying at the level of
   its endpoints; a memorising one tends to "snap" between stored training images, with
   similarity peaking in the middle of a path and reaching copy level. The interior peak
   is reported as (highest interior-frame SSIM) - (mean endpoint SSIM).
   Rows also interpolate from a generated slice to the inversion of a real training scan
   and of a real test scan, i.e. between synthetic images and real scans.

2. Inversion (membership test). For one random slice from each of N training patients and
   N unseen test patients, the latent that best reproduces the slice is found by gradient
   descent (StyleGAN2: one w per slice, starting at the average w, noise switched off;
   WGAN-GP: z). If the generator has memorised its training patients, it reproduces
   training slices noticeably better than test slices. Reported: mean masked-SSIM
   reconstruction quality for each set and their difference, with a bootstrap 95%
   interval (one slice per patient, so slices are independent).

Usage (GPU job on Rangpur, see latent.sh; needs audit.py outputs):
    python latent.py --run_name sg2_adni64 --iter 10000
    python latent.py --run_name wgan_adni64 --iter 10000
Outputs: figures/<run>_iterNNNNNN_interpolation.png, figures/<run>_iterNNNNNN_inversion.png,
outputs/eval/<run>/latent_iterNNNNNN.json.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from audit import SSIMIndex
from dataset import DEFAULT_CACHE, get_adni_splits
from predict import audit_generated, build_generator, load_run, snapshot_iter


def to_uint8(x):
    """Map images from [-1, 1] to uint8 [0, 255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def is_stylegan(config):
    """True if the run's config describes a StyleGAN2 generator."""
    return config["model"] == "stylegan2"


def latent_to_image(G, config, latent):
    """StyleGAN2: latent is w (N, w_dim), noise off; WGAN-GP: latent is z (N, latent_dim)."""
    if is_stylegan(config):
        return G.synthesis(latent.unsqueeze(1).expand(-1, G.num_ws, -1), noise_mode="none")
    return G(latent)


def sample_latents(G, config, n, gen):
    """Sample n latents: w = mapping(z) for StyleGAN2, z for the WGAN-GP baseline."""
    z = torch.randn(n, config["latent_dim"], device=gen.device, generator=gen)
    if is_stylegan(config):
        with torch.no_grad():
            return G.mapping(z)
    return z


def interpolate(a, b, steps, config):
    """Return `steps` latents from a to b: linear in W for StyleGAN2, spherical in z for the baseline."""
    t = torch.linspace(0, 1, steps, device=a.device).view(-1, 1)
    if is_stylegan(config):
        return a * (1 - t) + b * t  # linear in W
    an, bn = F.normalize(a, dim=-1), F.normalize(b, dim=-1)  # spherical in z, keeping the norm
    omega = torch.acos((an * bn).sum().clamp(-1 + 1e-7, 1 - 1e-7))
    path = (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)
    return path


def invert(G, config, targets, steps, lr, seed):
    """Latents that best reproduce each target image (independent per image, optimised in one batch)."""
    gen = torch.Generator(device=targets.device).manual_seed(seed)
    if is_stylegan(config):
        init = G.mapping.w_avg.detach().expand(len(targets), -1).clone()
    else:
        init = torch.randn(len(targets), config["latent_dim"], device=targets.device, generator=gen) * 0.1
    latent = init.requires_grad_(True)
    opt = torch.optim.Adam([latent], lr=lr)
    target = targets.float() / 127.5 - 1
    for _ in range(steps):
        loss = F.mse_loss(latent_to_image(G, config, latent), target, reduction="none").mean(dim=(1, 2, 3)).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return latent.detach(), latent_to_image(G, config, latent)


def paired_ssim(a, b, device, threshold):
    """Masked SSIM between a[i] and b[i]."""
    index = SSIMIndex(torch.cat([a, b]), device, threshold)
    n = len(a)
    _, masked = index.paired(torch.arange(n, device=device), torch.arange(n, 2 * n, device=device))
    return masked.cpu()


def one_per_patient(split, n, gen):
    """One random slice from each of n random patients."""
    patients = split.subject_idx.unique()
    patients = patients[torch.randperm(len(patients), generator=gen)[:n]]
    idx = []
    for p in patients.tolist():
        rows = (split.subject_idx == p).nonzero().flatten()
        idx.append(int(rows[torch.randint(len(rows), (1,), generator=gen)]))
    return torch.tensor(idx)


def plot_interpolation(rows, path, title, copy_level):
    """Plot interpolation paths with each frame's nearest-training SSIM (red at or above copy level)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = max(len(frames) + (1 if real is not None else 0) for _, frames, _, real in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.45 * n_cols + 1.2, 1.7 * len(rows) + 0.8), squeeze=False)
    for r, (label, frames, ssims, real) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(frames):
                ax.imshow(frames[c][0].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
                colour = "red" if ssims[c] >= copy_level else "black"
                ax.set_title(f"{ssims[c]:.2f}", fontsize=7, color=colour)
            elif c == len(frames) and real is not None:
                ax.imshow(real[0].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
                ax.set_title("real scan", fontsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#C44E52")
                    spine.set_linewidth(1.5)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(label, fontsize=7)
    fig.suptitle(f"{title}\nNumber above each frame: masked SSIM to its nearest training slice "
                 f"(red if at or above the copy level {copy_level:.3f})", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_inversion(train_q, test_q, examples, path, title):
    """Plot the inversion membership test: reconstruction quality for training vs unseen patients, plus examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_axes([0.05, 0.14, 0.32, 0.72])
    bins = torch.linspace(min(train_q.min(), test_q.min()).item(), 1.0, 30).numpy()
    ax.hist(train_q.numpy(), bins=bins, alpha=0.6, label=f"training patients (mean {train_q.mean():.3f})")
    ax.hist(test_q.numpy(), bins=bins, alpha=0.6, label=f"unseen test patients (mean {test_q.mean():.3f})")
    ax.set_xlabel("reconstruction quality (masked SSIM)")
    ax.set_ylabel("slices")
    ax.legend(fontsize=8)
    ax.set_title("How well can the generator reproduce a real slice?", fontsize=9)
    k = len(examples)
    for i, (label, target, recon, q) in enumerate(examples):
        for j, (img, cap) in enumerate(((target, f"{label}: real"), (recon, f"inverted, SSIM {q:.2f}"))):
            a = fig.add_axes([0.42 + i * (0.56 / k), 0.52 - j * 0.42, 0.56 / k - 0.01, 0.36])
            a.imshow(img[0].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(cap, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Run latent interpolation with nearest-neighbour tracking and the inversion membership test for one snapshot."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Latent interpolation and inversion-based membership test.")
    p.add_argument("--run_name", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--pairs", type=int, default=40, help="generated-to-generated paths measured")
    p.add_argument("--steps", type=int, default=9, help="frames per path, including endpoints")
    p.add_argument("--n_invert", type=int, default=64, help="patients per set for the inversion test")
    p.add_argument("--invert_steps", type=int, default=500)
    p.add_argument("--invert_lr", type=float, default=None, help="default: 0.01 (stylegan2), 0.05 (wgan-gp)")
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
    config, snaps = load_run(os.path.join(args.runs_root, args.run_name))
    match = [s for s in snaps if snapshot_iter(s) == args.iter]
    if not match:
        raise SystemExit(f"No snapshot at iteration {args.iter}; available {[snapshot_iter(s) for s in snaps]}")
    G = build_generator(config, device)
    G.load_state_dict(torch.load(match[0], map_location=device))
    G.eval().requires_grad_(False)
    lr = args.invert_lr if args.invert_lr is not None else (0.01 if is_stylegan(config) else 0.05)

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
    gen = torch.Generator(device=device).manual_seed(args.seed)
    cpu_gen = torch.Generator().manual_seed(args.seed)

    # --- 2. Inversion membership test (done first: its latents feed the real-scan paths) ---
    tr_idx, te_idx = one_per_patient(train, args.n_invert, cpu_gen), one_per_patient(test, args.n_invert, cpu_gen)
    targets = torch.cat([train.images[tr_idx], test.images[te_idx]]).to(device)
    latents, recons = invert(G, config, targets, args.invert_steps, lr, args.seed)
    quality = paired_ssim(targets, to_uint8(recons), device, bt)
    n_tr = len(tr_idx)
    q_tr, q_te = quality[:n_tr], quality[n_tr:]
    diff = float(q_tr.mean() - q_te.mean())
    boot = torch.stack([q_tr[torch.randint(n_tr, (n_tr,), generator=cpu_gen)].mean()
                        - q_te[torch.randint(len(q_te), (len(q_te),), generator=cpu_gen)].mean() for _ in range(2000)])
    ci = [float(torch.quantile(boot, 0.025)), float(torch.quantile(boot, 0.975))]

    # --- 1. Interpolation ---
    a, b = sample_latents(G, config, args.pairs, gen), sample_latents(G, config, args.pairs, gen)
    peaks, path_max, n_snaps, rows = [], [], [], []
    with torch.no_grad():
        for i in range(args.pairs):
            frames = to_uint8(latent_to_image(G, config, interpolate(a[i], b[i], args.steps, config)))
            nn = audit_generated(frames, train_index, device, bt)
            s = nn["masked"]
            peaks.append(float(s[1:-1].max() - s[[0, -1]].mean()))
            path_max.append(float(s.max()))
            n_snaps.append(int(nn["masked_arg"].unique().numel()))
            if i < 3:
                rows.append((f"generated to\ngenerated ({i + 1})", frames, s.tolist(), None))
        for label, j in (("generated to\ntraining scan", 0), ("generated to\ntest scan", n_tr)):
            path_lat = interpolate(a[-1 - (j > 0)], latents[j], args.steps, config)
            frames = to_uint8(latent_to_image(G, config, path_lat))
            s = audit_generated(frames, train_index, device, bt)["masked"]
            rows.append((label, frames, s.tolist(), targets[j]))
    peaks, path_max = torch.tensor(peaks), torch.tensor(path_max)

    os.makedirs(args.fig_dir, exist_ok=True)
    stem = f"{args.run_name}_iter{args.iter:06d}"
    space = "W (linear)" if is_stylegan(config) else "z (spherical)"
    plot_interpolation(rows, os.path.join(args.fig_dir, f"{stem}_interpolation.png"),
                       f"Latent interpolation in {space}: {args.run_name}, iteration {args.iter}", copy_level)
    ex = [("train", targets[0], to_uint8(recons[:1])[0], float(q_tr[0])),
          ("train", targets[1], to_uint8(recons[1:2])[0], float(q_tr[1])),
          ("test", targets[n_tr], to_uint8(recons[n_tr:n_tr + 1])[0], float(q_te[0])),
          ("test", targets[n_tr + 1], to_uint8(recons[n_tr + 1:n_tr + 2])[0], float(q_te[1]))]
    plot_inversion(q_tr, q_te, ex, os.path.join(args.fig_dir, f"{stem}_inversion.png"),
                   f"Inversion membership test: {args.run_name}, iteration {args.iter} "
                   f"(train minus test {diff:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])")

    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    result = dict(run=args.run_name, iteration=args.iter, copy_level=copy_level,
                  interpolation=dict(pairs=args.pairs, steps=args.steps, space=space,
                                     interior_peak_mean=float(peaks.mean()),
                                     interior_peak_p95=float(torch.quantile(peaks, 0.95)),
                                     paths_reaching_copy_level=int((path_max >= copy_level).sum()),
                                     distinct_nn_per_path_mean=float(sum(n_snaps) / len(n_snaps))),
                  inversion=dict(n_per_set=n_tr, steps=args.invert_steps, lr=lr,
                                 train_mean=float(q_tr.mean()), test_mean=float(q_te.mean()),
                                 difference=diff, ci95=ci))
    with open(os.path.join(out_dir, f"latent_iter{args.iter:06d}.json"), "w") as f:
        json.dump(result, f, indent=2)

    it = result["interpolation"]
    print(f"Interpolation ({args.pairs} paths x {args.steps} frames in {space}):")
    print(f"  interior peak (max interior NN SSIM minus mean endpoint NN SSIM): mean {it['interior_peak_mean']:+.4f}, "
          f"95th percentile {it['interior_peak_p95']:+.4f}")
    print(f"  paths with any frame at copy level ({copy_level:.3f}): {it['paths_reaching_copy_level']}/{args.pairs}")
    print(f"  distinct nearest training slices per path: {it['distinct_nn_per_path_mean']:.1f} of {args.steps} frames")
    print(f"Inversion ({n_tr} training vs {len(q_te)} test patients, one slice each, {args.invert_steps} steps):")
    print(f"  reconstruction masked SSIM: training {q_tr.mean():.4f}, test {q_te.mean():.4f}, "
          f"difference {diff:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"Saved figures/{stem}_interpolation.png, figures/{stem}_inversion.png ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()