"""Train a generative model on ADNI (or OASIS, for verification) and log everything needed
to compare models: losses, sample grids, time per iteration, and peak GPU memory.

Two models share the same data, logging, snapshots and checkpointing, so both are trained
and measured the same way:
  --model wgan-gp    the baseline (modules.WGANGenerator / WGANCritic, ported from Lab 2)
  --model stylegan2  modules.StyleGAN2Generator / StyleGAN2Discriminator

Training length is set in generator iterations, not epochs, because ADNI's training split
(24,520 slices) is 2.5x larger than OASIS (9,664): the Lab 2 run's 100 OASIS epochs equal
15,100 generator iterations at batch size 64, which is the default here.

WGAN-GP training follows Lab 2 exactly (Adam lr 1e-4, betas (0, 0.9), 5 critic steps per
generator step, gradient-penalty weight 10). Like Lab 2, the 5 critic steps of one
iteration reuse the same real batch with fresh fake batches; Gulrajani et al. (2017)
draw a new real batch for every critic step. Kept as is so the port reproduces the
known-good Lab 2 behaviour.

StyleGAN2 training follows Karras et al. (2020) and the small-data settings of Karras et
al. (2020b, ADA paper): Adam lr 0.0025, betas (0, 0.99); one discriminator and one
generator step per iteration with the non-saturating logistic loss; R1 regularisation on
real images applied lazily every 16 steps (scaled by 16, with the optimiser corrected
for the interval), gamma = 0.0002 * resolution^2 / batch size unless --r1_gamma is set;
and an exponential moving average (EMA) of the generator, with a half-life of
10 * batch / 32 thousand images, which is used for all samples, snapshots and
evaluation. With --ada, adaptive discriminator augmentation (modules.AdaptiveAugment)
augments every image the discriminator sees, including in the R1 step, with p updated
every 4 iterations towards E[sign(D(real))] = 0.6 (Karras et al., 2020b). With
--aug_p P, the same augmentations are applied with a fixed probability P instead (no
adaptation). Not included: style-mixing and path-length regularisation. One iteration
shows the discriminator one real batch, so thousands of real images seen (kimg) =
iterations * batch size / 1000, the same count as for WGAN-GP.

With --conditional (StyleGAN2 on ADNI only), both networks receive the diagnosis (CN or
AD) as a one-hot label: real batches carry their own labels, and generated batches use
labels drawn from the training slices, so both classes are generated in their training
proportions. Fixed-noise sample grids show CN in the top four rows and AD in the bottom
four.

Usage (GPU job on Rangpur, see train.sh):
    python train.py --dataset oasis --iters 200 --run_name smoke       # quick check
    python train.py --dataset oasis --run_name wgan_oasis64            # reproduce Lab 2
    python train.py --dataset adni --run_name wgan_adni64              # baseline on ADNI
    python train.py --model stylegan2 --dataset oasis --iters 200 --run_name sg2_smoke

Outputs go to outputs/runs/<run_name>/ (gitignored): config.json, history.json,
summary.json, loss_curves.png, samples/iter_XXXXXX.png (fixed noise), snapshots/
(generator weights at fixed intervals, kept for later evaluation) and checkpoint.pt
(latest full training state, overwritten, used by --resume).
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from dataset import DEFAULT_CACHE, DEFAULT_SEED, OASISSlices, get_adni_splits
from modules import (AdaptiveAugment, StyleGAN2Discriminator, StyleGAN2Generator, WGANCritic, WGANGenerator,
                     count_parameters, d_logistic_loss, g_nonsaturating_loss, gradient_penalty, r1_penalty)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class GPUBatches:
    """Shuffled minibatches drawn from a uint8 image tensor held on the GPU.

    The training split at 64 x 64 is about 100 MB as uint8, so it fits on the GPU and no
    DataLoader workers are needed. Each epoch is a fresh permutation; the last partial
    batch is dropped, as in Lab 2.
    """

    def __init__(self, images_uint8, batch_size, device, seed, labels=None):
        self.images = images_uint8.to(device)
        self.labels = labels.to(device) if labels is not None else None
        self.last_labels = None
        self.batch_size = batch_size
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.epoch, self._perm, self._pos = 0, None, len(self.images)

    def next(self):
        if self._pos + self.batch_size > len(self.images):
            self._perm = torch.randperm(len(self.images), device=self.images.device, generator=self.gen)
            self._pos = 0
            self.epoch += 1
        idx = self._perm[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        if self.labels is not None:
            self.last_labels = self.labels[idx]
        return self.images[idx].float() / 127.5 - 1.0  # [-1, 1], as in dataset.py

    def sample_labels(self, n):
        """Labels of n random training slices, so generated classes follow the training mix."""
        return self.labels[torch.randint(len(self.labels), (n,), device=self.labels.device, generator=self.gen)]


def load_training_images(dataset, resolution, cache_dir, num_workers):
    """Training images, and their class labels for ADNI (None for OASIS, which is unlabelled here)."""
    if dataset == "adni":
        train = get_adni_splits(resolution=resolution, cache_dir=cache_dir, num_workers=num_workers)["train"]
        return train.images, train.labels
    return OASISSlices(resolution=resolution, cache_dir=cache_dir, num_workers=num_workers).images, None


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def save_samples(generator, z, path, c=None):
    """Save a grid of generated images (torchvision handles the tiling)."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images = generator(z) if c is None else generator(z, c)
        save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))
    generator.train(was_training)


WGAN_PANELS = [("WGAN-GP losses", [("critic_loss", "critic loss"), ("gen_loss", "generator loss")]),
               ("Wasserstein estimate  E[D(real)] - E[D(fake)]", [("w_est", None)]),
               ("Gradient penalty", [("gp", None)])]
STYLEGAN_PANELS = [("StyleGAN2 losses", [("d_loss", "discriminator loss"), ("g_loss", "generator loss")]),
                   ("Discriminator scores", [("real_score", "D(real)"), ("fake_score", "D(fake)")]),
                   ("R1 penalty  ||grad D(real)||^2", [("r1", None)])]
ADA_PANEL = ("ADA: augmentation probability p and r_t", [("ada_p", "p"), ("r_t", "r_t = E[sign(D(real))]")])


def plot_history(history, path, panels):
    """One panel per (title, [(history key, legend label)]) entry, against generator iteration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = history["iter"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for ax, (title, series) in zip(axes, panels):
        for key, label in series:
            ax.plot(it, history[key], label=label)
        if any(label for _, label in series):
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel("generator iteration")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_wgan_gp(args, batches, device, run_dir):
    generator = WGANGenerator(args.latent_dim, args.resolution).to(device)
    critic = WGANCritic(args.resolution).to(device)
    g_opt = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    c_opt = optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.9))
    fixed_z = torch.randn(64, args.latent_dim, device=device, generator=torch.Generator(device=device).manual_seed(args.seed))

    history = {k: [] for k in ("iter", "epoch", "critic_loss", "gen_loss", "w_est", "gp", "sec_per_iter")}
    start_iter, ckpt_path = 0, os.path.join(run_dir, "checkpoint.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        critic.load_state_dict(ckpt["critic"])
        g_opt.load_state_dict(ckpt["g_opt"])
        c_opt.load_state_dict(ckpt["c_opt"])
        fixed_z, history, start_iter = ckpt["fixed_z"].to(device), ckpt["history"], ckpt["iter"]
        # The shuffling RNG continues where it stopped; the interrupted epoch's remaining
        # batches are skipped and a new permutation starts.
        batches.gen.set_state(ckpt["batch_rng"].cpu())
        batches.epoch = ckpt["epoch"]
        print(f"Resumed from iteration {start_iter}")
    print(f"Generator {count_parameters(generator):,} parameters, critic {count_parameters(critic):,}")

    acc = dict(critic_loss=0.0, gen_loss=0.0, w_est=0.0, gp=0.0, n=0)
    t_log = time.time()
    for it in range(start_iter + 1, args.iters + 1):
        real = batches.next()

        for _ in range(args.n_critic):
            z = torch.randn(real.size(0), args.latent_dim, device=device)
            fake = generator(z).detach()
            d_real, d_fake = critic(real).mean(), critic(fake).mean()
            gp = gradient_penalty(critic, real, fake)
            c_loss = d_fake - d_real + args.lambda_gp * gp
            c_opt.zero_grad(set_to_none=True)
            c_loss.backward()
            c_opt.step()

        z = torch.randn(real.size(0), args.latent_dim, device=device)
        g_loss = -critic(generator(z)).mean()
        g_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        g_opt.step()

        acc["critic_loss"] += c_loss.item()
        acc["gen_loss"] += g_loss.item()
        acc["w_est"] += (d_real - d_fake).item()
        acc["gp"] += gp.item()
        acc["n"] += 1

        if it % args.log_every == 0 or it == args.iters:
            if device.type == "cuda":
                torch.cuda.synchronize()
            sec = (time.time() - t_log) / acc["n"]
            history["iter"].append(it)
            history["epoch"].append(batches.epoch)
            for k in ("critic_loss", "gen_loss", "w_est", "gp"):
                history[k].append(acc[k] / acc["n"])
            history["sec_per_iter"].append(sec)
            print(f"iter {it:6d}/{args.iters} | epoch {batches.epoch:3d} | critic {history['critic_loss'][-1]:8.4f} "
                  f"| gen {history['gen_loss'][-1]:8.4f} | W {history['w_est'][-1]:7.4f} "
                  f"| gp {history['gp'][-1]:.4f} | {sec * 1000:.1f} ms/iter", flush=True)
            acc = dict(critic_loss=0.0, gen_loss=0.0, w_est=0.0, gp=0.0, n=0)
            t_log = time.time()

        if it % args.sample_every == 0 or it == args.iters:
            save_samples(generator, fixed_z, os.path.join(run_dir, "samples", f"iter_{it:06d}.png"))
        if it % args.snapshot_every == 0 or it == args.iters:
            torch.save(generator.state_dict(), os.path.join(run_dir, "snapshots", f"generator_{it:06d}.pt"))
        if it % args.checkpoint_every == 0 or it == args.iters:
            torch.save(dict(generator=generator.state_dict(), critic=critic.state_dict(),
                            g_opt=g_opt.state_dict(), c_opt=c_opt.state_dict(), fixed_z=fixed_z.cpu(),
                            history=history, iter=it, epoch=batches.epoch,
                            batch_rng=batches.gen.get_state()), ckpt_path)
            plot_history(history, os.path.join(run_dir, "loss_curves.png"), WGAN_PANELS)
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f)

    # Diversity check: fresh noise, not the fixed noise used for progression grids.
    save_samples(generator, torch.randn(64, args.latent_dim, device=device),
                 os.path.join(run_dir, "random_samples_final.png"))
    return dict(generator_params=count_parameters(generator), critic_params=count_parameters(critic))


def stylegan2_settings(args):
    """R1 weight and EMA decay per iteration, from the ADA paper's heuristics unless overridden."""
    gamma = args.r1_gamma if args.r1_gamma is not None else 0.0002 * args.resolution ** 2 / args.batch_size
    ema_kimg = args.batch_size * 10 / 32
    ema_beta = 0.5 ** (args.batch_size / (ema_kimg * 1000))
    return gamma, ema_beta


def train_stylegan2(args, batches, device, run_dir):
    c_dim = 2 if args.conditional else 0
    G = StyleGAN2Generator(args.resolution, z_dim=args.latent_dim, channel_base=args.channel_base,
                           max_channels=args.max_channels, c_dim=c_dim).to(device)
    D = StyleGAN2Discriminator(args.resolution, channel_base=args.channel_base,
                               max_channels=args.max_channels, c_dim=c_dim).to(device)

    def onehot(labels):
        return F.one_hot(labels, 2).float() if c_dim else None
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    gamma, ema_beta = stylegan2_settings(args)
    # Lazy regularisation: the R1 step runs every r1_every iterations, so the optimiser's
    # learning rate and betas are adjusted as in the StyleGAN2 paper (Appendix B).
    c = args.r1_every / (args.r1_every + 1)
    g_opt = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_opt = optim.Adam(D.parameters(), lr=args.lr * c, betas=(0.0, 0.99 ** c))
    fixed_z = torch.randn(64, args.latent_dim, device=device,
                          generator=torch.Generator(device=device).manual_seed(args.seed))
    fixed_c = onehot(torch.arange(64, device=device) // 32)  # rows 0-3 CN, rows 4-7 AD
    aug = None
    if args.ada or args.aug_p is not None:
        aug = AdaptiveAugment(target=args.ada_target, ada_kimg=args.ada_kimg, fixed_p=args.aug_p).to(device)

    def A(x):  # what the discriminator sees: augmented when ADA is on
        return aug(x) if aug is not None else x

    keys = ("iter", "epoch", "kimg", "d_loss", "g_loss", "real_score", "fake_score", "r1", "sec_per_iter")
    if aug is not None:
        keys += ("ada_p", "r_t")
    panels = STYLEGAN_PANELS + ([ADA_PANEL] if aug is not None else [])
    history = {k: [] for k in keys}
    start_iter, ckpt_path = 0, os.path.join(run_dir, "checkpoint.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        for module, key in ((G, "generator"), (D, "discriminator"), (G_ema, "generator_ema"),
                            (g_opt, "g_opt"), (d_opt, "d_opt")):
            module.load_state_dict(ckpt[key])
        if aug is not None:
            aug.load_state_dict(ckpt["ada"])
        fixed_z, history, start_iter = ckpt["fixed_z"].to(device), ckpt["history"], ckpt["iter"]
        batches.gen.set_state(ckpt["batch_rng"].cpu())
        batches.epoch = ckpt["epoch"]
        print(f"Resumed from iteration {start_iter}")
    print(f"Generator {count_parameters(G):,} parameters (mapping {count_parameters(G.mapping):,}), "
          f"discriminator {count_parameters(D):,}")
    print(f"R1 gamma {gamma:.4g} every {args.r1_every} iterations; EMA beta {ema_beta:.6f} per iteration")
    if aug is not None:
        if aug.adaptive:
            print(f"ADA on: target r_t {args.ada_target}, p updated every {args.ada_interval} iterations, "
                  f"p can change by 1 per {args.ada_kimg} kimg; starting p {float(aug.p):.3f}")
        else:
            print(f"Fixed-probability augmentation: p = {float(aug.p):.3f} throughout (r_t measured, not used)")
    last_rt = float("nan")

    acc = dict(d_loss=0.0, g_loss=0.0, real_score=0.0, fake_score=0.0, n=0)
    last_r1 = float("nan")
    t_log = time.time()
    for it in range(start_iter + 1, args.iters + 1):
        real = batches.next()

        # Discriminator step: logistic loss on a real and a generated batch.
        z = torch.randn(real.size(0), args.latent_dim, device=device)
        with torch.no_grad():
            c_fake = onehot(batches.sample_labels(real.size(0))) if c_dim else None
            fake = G(z, c_fake)
        c_real = onehot(batches.last_labels) if c_dim else None
        real_scores, fake_scores = D(A(real), c_real), D(A(fake), c_fake)
        d_loss = d_logistic_loss(real_scores, fake_scores)
        d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        d_opt.step()
        if aug is not None:
            aug.accumulate(real_scores)
            if it % args.ada_interval == 0:
                last_rt = aug.update(args.batch_size, args.ada_interval)

        # Lazy R1 on real images, scaled by the interval so its average weight is gamma / 2.
        if it % args.r1_every == 0:
            r1 = r1_penalty(D, A(real).detach(), c_real)
            d_opt.zero_grad(set_to_none=True)
            (gamma / 2 * r1 * args.r1_every).backward()
            d_opt.step()
            last_r1 = r1.item()

        # Generator step: non-saturating loss; also updates the running average of w.
        z = torch.randn(real.size(0), args.latent_dim, device=device)
        c_gen = onehot(batches.sample_labels(real.size(0))) if c_dim else None
        g_loss = g_nonsaturating_loss(D(A(G(z, c_gen, track_w_avg=True)), c_gen))
        g_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        g_opt.step()

        # Exponential moving average of the generator weights (buffers such as w_avg copied).
        with torch.no_grad():
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.lerp_(p, 1 - ema_beta)
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        acc["d_loss"] += d_loss.item()
        acc["g_loss"] += g_loss.item()
        acc["real_score"] += real_scores.mean().item()
        acc["fake_score"] += fake_scores.mean().item()
        acc["n"] += 1

        if it % args.log_every == 0 or it == args.iters:
            if device.type == "cuda":
                torch.cuda.synchronize()
            sec = (time.time() - t_log) / acc["n"]
            history["iter"].append(it)
            history["epoch"].append(batches.epoch)
            history["kimg"].append(it * args.batch_size / 1000)
            for k in ("d_loss", "g_loss", "real_score", "fake_score"):
                history[k].append(acc[k] / acc["n"])
            history["r1"].append(last_r1)
            history["sec_per_iter"].append(sec)
            if aug is not None:
                history["ada_p"].append(float(aug.p))
                history["r_t"].append(last_rt)
            print(f"iter {it:6d}/{args.iters} | kimg {history['kimg'][-1]:7.1f} | epoch {batches.epoch:3d} "
                  f"| D {history['d_loss'][-1]:.4f} | G {history['g_loss'][-1]:.4f} "
                  f"| D(real) {history['real_score'][-1]:+.3f} D(fake) {history['fake_score'][-1]:+.3f} "
                  f"| R1 {last_r1:.4f} | {sec * 1000:.1f} ms/iter"
                  + (f" | p {float(aug.p):.3f} r_t {last_rt:.3f}" if aug is not None else ""), flush=True)
            acc = dict(d_loss=0.0, g_loss=0.0, real_score=0.0, fake_score=0.0, n=0)
            t_log = time.time()

        if it % args.sample_every == 0 or it == args.iters:
            save_samples(G_ema, fixed_z, os.path.join(run_dir, "samples", f"iter_{it:06d}.png"), fixed_c)
        if it % args.snapshot_every == 0 or it == args.iters:
            torch.save(G_ema.state_dict(), os.path.join(run_dir, "snapshots", f"generator_{it:06d}.pt"))
        if it % args.checkpoint_every == 0 or it == args.iters:
            torch.save(dict(generator=G.state_dict(), discriminator=D.state_dict(), generator_ema=G_ema.state_dict(),
                            g_opt=g_opt.state_dict(), d_opt=d_opt.state_dict(), fixed_z=fixed_z.cpu(),
                            history=history, iter=it, epoch=batches.epoch, batch_rng=batches.gen.get_state(),
                            ada=aug.state_dict() if aug is not None else None), ckpt_path)
            plot_history(history, os.path.join(run_dir, "loss_curves.png"), panels)
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f)

    save_samples(G_ema, torch.randn(64, args.latent_dim, device=device), c=fixed_c, path=
                 os.path.join(run_dir, "random_samples_final.png"))
    return dict(generator_params=count_parameters(G), discriminator_params=count_parameters(D),
                final_ada_p=float(aug.p) if aug is not None else None,
                r1_gamma=gamma, ema_beta=ema_beta)


MODEL_DEFAULTS = {"wgan-gp": dict(lr=1e-4, latent_dim=100), "stylegan2": dict(lr=0.0025, latent_dim=512)}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Train a generative model on ADNI or OASIS brain slices.")
    p.add_argument("--model", choices=["wgan-gp", "stylegan2"], default="wgan-gp")
    p.add_argument("--dataset", choices=["adni", "oasis"], default="adni")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--run_name", required=True)
    p.add_argument("--iters", type=int, default=15100, help="generator iterations (Lab 2: 100 OASIS epochs = 15,100)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=None, help="default: 100 (wgan-gp), 512 (stylegan2)")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-4 (wgan-gp), 0.0025 (stylegan2)")
    p.add_argument("--n_critic", type=int, default=5, help="wgan-gp only")
    p.add_argument("--lambda_gp", type=float, default=10.0, help="wgan-gp only")
    p.add_argument("--r1_gamma", type=float, default=None, help="stylegan2: default 0.0002 * res^2 / batch")
    p.add_argument("--r1_every", type=int, default=16, help="stylegan2: lazy R1 interval")
    p.add_argument("--channel_base", type=int, default=8192, help="stylegan2: channels = base / resolution")
    p.add_argument("--max_channels", type=int, default=256, help="stylegan2: channel cap")
    p.add_argument("--ada", action="store_true", help="stylegan2: adaptive discriminator augmentation")
    p.add_argument("--ada_target", type=float, default=0.6, help="stylegan2 ADA: target E[sign(D(real))]")
    p.add_argument("--ada_interval", type=int, default=4, help="stylegan2 ADA: iterations between p updates")
    p.add_argument("--ada_kimg", type=float, default=500, help="stylegan2 ADA: kimg for p to change by 1")
    p.add_argument("--conditional", action="store_true",
                   help="stylegan2 on adni: condition both networks on the diagnosis (CN / AD)")
    p.add_argument("--aug_p", type=float, default=None,
                   help="stylegan2: augment with this fixed probability instead of adapting it (no --ada needed)")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--snapshot_every", type=int, default=2500, help="keep generator weights at this interval")
    p.add_argument("--checkpoint_every", type=int, default=1000, help="overwrite the resumable full checkpoint")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--out_root", default=os.path.join(here, "outputs", "runs"))
    args = p.parse_args()
    if args.conditional and (args.model != "stylegan2" or args.dataset != "adni"):
        raise SystemExit("--conditional needs --model stylegan2 and --dataset adni (class labels)")
    for k, v in MODEL_DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # faster matmuls on A100; convolutions already use TF32
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.out_root, args.run_name)
    if os.path.exists(os.path.join(run_dir, "checkpoint.pt")) and not args.resume:
        raise SystemExit(f"{run_dir} already has a checkpoint: pass --resume or choose another --run_name")
    for sub in ("samples", "snapshots"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Run directory: {run_dir}")

    images, labels = load_training_images(args.dataset, args.resolution, args.cache_dir, args.num_workers)
    batches = GPUBatches(images, args.batch_size, device, args.seed, labels=labels if args.conditional else None)
    print(f"{args.dataset}: {len(images)} training slices at {args.resolution} px, "
          f"{len(images) // args.batch_size} iterations per epoch")

    t0 = time.time()
    info = (train_wgan_gp if args.model == "wgan-gp" else train_stylegan2)(args, batches, device, run_dir)
    elapsed = time.time() - t0

    summary = dict(info, model=args.model, dataset=args.dataset, resolution=args.resolution,
                   iters=args.iters, kimg=args.iters * args.batch_size / 1000, epochs=batches.epoch,
                   train_seconds=round(elapsed, 1),
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()