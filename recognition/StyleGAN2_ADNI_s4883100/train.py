"""Train a generative model on ADNI (or OASIS, for verification) and log everything needed
to compare models: losses, sample grids, time per iteration, and peak GPU memory.

Currently supports the WGAN-GP baseline (modules.WGANGenerator / WGANCritic, ported from my
Lab 2 model). StyleGAN2 will be added as another --model choice sharing the same data,
logging and checkpointing, so both models are trained and measured the same way.

Training length is set in generator iterations, not epochs, because ADNI's training split
(24,520 slices) is 2.5x larger than OASIS (9,664): the Lab 2 run's 100 OASIS epochs equal
15,100 generator iterations at batch size 64, which is the default here.

WGAN-GP training follows Lab 2 exactly (Adam lr 1e-4, betas (0, 0.9), 5 critic steps per
generator step, gradient-penalty weight 10). Like Lab 2, the 5 critic steps of one
iteration reuse the same real batch with fresh fake batches; Gulrajani et al. (2017)
draw a new real batch for every critic step. Kept as is so the port reproduces the
known-good Lab 2 behaviour.

Usage (GPU job on Rangpur, see train.sh):
    python train.py --dataset oasis --iters 200 --run_name smoke       # quick check
    python train.py --dataset oasis --run_name wgan_oasis64            # reproduce Lab 2
    python train.py --dataset adni --run_name wgan_adni64              # baseline on ADNI

Outputs go to outputs/runs/<run_name>/ (gitignored): config.json, history.json,
summary.json, loss_curves.png, samples/iter_XXXXXX.png (fixed noise), snapshots/
(generator weights at fixed intervals, kept for later evaluation) and checkpoint.pt
(latest full training state, overwritten, used by --resume).
"""

import argparse
import json
import os
import time

import torch
import torch.optim as optim
from torchvision.utils import save_image

from dataset import DEFAULT_CACHE, DEFAULT_SEED, OASISSlices, get_adni_splits
from modules import WGANCritic, WGANGenerator, count_parameters, gradient_penalty


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class GPUBatches:
    """Shuffled minibatches drawn from a uint8 image tensor held on the GPU.

    The training split at 64 x 64 is about 100 MB as uint8, so it fits on the GPU and no
    DataLoader workers are needed. Each epoch is a fresh permutation; the last partial
    batch is dropped, as in Lab 2.
    """

    def __init__(self, images_uint8, batch_size, device, seed):
        self.images = images_uint8.to(device)
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
        return self.images[idx].float() / 127.5 - 1.0  # [-1, 1], as in dataset.py


def load_training_images(dataset, resolution, cache_dir, num_workers):
    if dataset == "adni":
        return get_adni_splits(resolution=resolution, cache_dir=cache_dir, num_workers=num_workers)["train"].images
    return OASISSlices(resolution=resolution, cache_dir=cache_dir, num_workers=num_workers).images


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def save_samples(generator, z, path):
    """Save a grid of generated images (torchvision handles the tiling)."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        save_image(generator(z), path, nrow=8, normalize=True, value_range=(-1, 1))
    generator.train(was_training)


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = history["iter"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(it, history["critic_loss"], label="critic loss")
    axes[0].plot(it, history["gen_loss"], label="generator loss")
    axes[0].set_title("WGAN-GP losses")
    axes[0].legend()
    axes[1].plot(it, history["w_est"], color="green")
    axes[1].set_title("Wasserstein estimate  E[D(real)] - E[D(fake)]")
    axes[2].plot(it, history["gp"], color="purple")
    axes[2].set_title("Gradient penalty")
    for ax in axes:
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
            plot_history(history, os.path.join(run_dir, "loss_curves.png"))
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(history, f)

    # Diversity check: fresh noise, not the fixed noise used for progression grids.
    save_samples(generator, torch.randn(64, args.latent_dim, device=device),
                 os.path.join(run_dir, "random_samples_final.png"))
    return dict(generator_params=count_parameters(generator), critic_params=count_parameters(critic))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Train a generative model on ADNI or OASIS brain slices.")
    p.add_argument("--model", choices=["wgan-gp"], default="wgan-gp")
    p.add_argument("--dataset", choices=["adni", "oasis"], default="adni")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--run_name", required=True)
    p.add_argument("--iters", type=int, default=15100, help="generator iterations (Lab 2: 100 OASIS epochs = 15,100)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda_gp", type=float, default=10.0)
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

    torch.manual_seed(args.seed)
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

    images = load_training_images(args.dataset, args.resolution, args.cache_dir, args.num_workers)
    batches = GPUBatches(images, args.batch_size, device, args.seed)
    print(f"{args.dataset}: {len(images)} training slices at {args.resolution} px, "
          f"{len(images) // args.batch_size} iterations per epoch")

    t0 = time.time()
    info = train_wgan_gp(args, batches, device, run_dir)
    elapsed = time.time() - t0

    summary = dict(info, model=args.model, dataset=args.dataset, resolution=args.resolution,
                   iters=args.iters, epochs=batches.epoch, train_seconds=round(elapsed, 1),
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()