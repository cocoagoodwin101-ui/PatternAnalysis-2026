"""Model components, implemented in PyTorch only.

WGAN-GP baseline: ported from my COMP3710 Lab 2 Task 3 OASIS model
(github.com/cocoagoodwin101-ui/comp3710-lab2-pattern-recognition, task3_gan/train_gan_oasis.py),
which trained stably on OASIS at 64 x 64 without mode collapse. At 64 x 64 the layers are
identical to Lab 2. The only change is that the number of up/downsampling stages now follows
the resolution, so the same baseline can be compared with StyleGAN2 at 128 and 256.

StyleGAN2: a compact pure-PyTorch implementation of the generator and discriminator of
Karras et al. (2020), "Analyzing and improving the image quality of StyleGAN", written
for this project rather than adapted from NVIDIA's code (which relies on custom CUDA
kernels). Implemented as in the paper: mapping network, weight modulation and
demodulation, per-layer noise, skip-connection generator outputs, residual discriminator
with minibatch standard deviation, equalised learning rate, non-saturating logistic loss
and R1 regularisation. Simplifications, chosen for clarity: resampling uses bilinear
upsampling and average pooling instead of transposed convolutions with FIR blur filters,
and the network is narrower than the paper's (channels capped at 256 by default) to fit
the compute budget.

Adaptive discriminator augmentation (ADA; Karras et al., 2020, "Training generative
adversarial networks with limited data"): every image the discriminator sees, real or
generated, is randomly augmented with probability p, and p is tuned during training so
the discriminator cannot simply memorise the training set. Implemented here with the
subset of ADA's augmentations that suit single-channel, skull-stripped MRI slices:
integer translation, isotropic scaling and rotation (one affine warp), and brightness
and contrast. Flips are left out because a mirrored sagittal slice is anatomically
wrong (front and back swap), and colour augmentations do not apply to greyscale.

Class conditioning (optional, c_dim > 0), following StyleGAN2-ADA: the generator's mapping
network embeds the one-hot class, normalises it and concatenates it with the normalised z,
so every style vector depends on the class; the discriminator is a projection
discriminator (Miyato and Koyama, 2018), adding the inner product of a learned class
embedding with its final features to its score. With c_dim = 0 (the default) both
networks are unchanged, so unconditional snapshots still load.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_CHANNELS = 512


def _stage_channels(resolution, base):
    """Channels per stage from 4 x 4 up to `resolution`, halving at each upsampling.

    At 64 x 64 with base 64 this gives [512, 256, 128, 64], exactly the Lab 2 widths.
    Wider stages are capped at MAX_CHANNELS for higher resolutions.
    """
    n = int(math.log2(resolution)) - 2  # number of 2x stages from 4 x 4
    if resolution < 8 or 2 ** (n + 2) != resolution:
        raise ValueError(f"resolution must be a power of two >= 8, got {resolution}")
    return [min(MAX_CHANNELS, base * 2 ** (n - 1 - i)) for i in range(n)]


class WGANGenerator(nn.Module):
    """DCGAN-style generator: latent vector -> (1, R, R) image in [-1, 1].

    A 4 x 4 transposed convolution projects z to a feature map, then each stage doubles the
    spatial size with a stride-2 transposed convolution followed by BatchNorm and ReLU. The
    final stage outputs one channel through tanh, matching the [-1, 1] data scaling.
    """

    def __init__(self, latent_dim=100, resolution=64, base=64, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        ch = _stage_channels(resolution, base)
        layers = [nn.ConvTranspose2d(latent_dim, ch[0], 4, 1, 0, bias=False),  # 1 -> 4
                  nn.BatchNorm2d(ch[0]), nn.ReLU(True)]
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),  # doubles size
                       nn.BatchNorm2d(c_out), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(ch[-1], img_channels, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))


class WGANCritic(nn.Module):
    """Critic: (1, R, R) image -> one unbounded score per image (no sigmoid).

    Each stage halves the spatial size with a stride-2 convolution. InstanceNorm is used
    instead of BatchNorm because the gradient penalty is defined per sample, and batch
    statistics would couple the samples in a batch (Gulrajani et al., 2017).
    """

    def __init__(self, resolution=64, base=64, img_channels=1):
        super().__init__()
        ch = list(reversed(_stage_channels(resolution, base)))  # e.g. [64, 128, 256, 512]
        layers = [nn.Conv2d(img_channels, ch[0], 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(c_in, c_out, 4, 2, 1),
                       nn.InstanceNorm2d(c_out, affine=True), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ch[-1], 1, 4, 1, 0)]  # 4 -> 1
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def gradient_penalty(critic, real, fake):
    """WGAN-GP penalty: (||grad critic(x_hat)||_2 - 1)^2 at random points between real and fake.

    Keeps the critic approximately 1-Lipschitz, which the Wasserstein formulation requires.
    """
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x_hat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(x_hat)
    grads = torch.autograd.grad(outputs=scores, inputs=x_hat, grad_outputs=torch.ones_like(scores),
                                create_graph=True, retain_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def count_parameters(model):
    """Total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# StyleGAN2 building blocks
# --------------------------------------------------------------------------- #
SQRT2 = math.sqrt(2.0)


def lrelu(x):
    """Leaky ReLU scaled by sqrt(2) to keep activations at unit variance (StyleGAN convention)."""
    return F.leaky_relu(x, 0.2) * SQRT2


class EqualLinear(nn.Module):
    """Fully connected layer with equalised learning rate.

    Weights are stored as N(0, 1) / lr_mul and scaled at run time by 1 / sqrt(fan_in), so
    every layer learns at the same effective rate under Adam. lr_mul < 1 slows a layer
    down (used by the mapping network).
    """

    def __init__(self, in_dim, out_dim, bias_init=0.0, lr_mul=1.0, activation=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_dim,), float(bias_init)))
        self.scale = lr_mul / math.sqrt(in_dim)
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)
        return lrelu(out) if self.activation else out


class EqualConv2d(nn.Module):
    """Convolution with equalised learning rate (used in the discriminator)."""

    def __init__(self, in_ch, out_ch, kernel, bias=True, activation=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = 1.0 / math.sqrt(in_ch * kernel * kernel)
        self.padding = kernel // 2
        self.activation = activation

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)
        return lrelu(out) if self.activation else out


class MappingNetwork(nn.Module):
    """z -> w: pixel-normalised latent through an MLP with a reduced learning rate.

    Keeps a running average of w (w_avg) for the truncation trick at sampling time.
    """

    def __init__(self, z_dim=512, w_dim=512, n_layers=8, lr_mul=0.01, w_avg_beta=0.995, c_dim=0):
        super().__init__()
        self.c_dim = c_dim
        # Conditional: one-hot class -> w_dim embedding, concatenated with z before the MLP.
        self.embed = EqualLinear(c_dim, w_dim) if c_dim > 0 else None
        dims = [z_dim + (w_dim if c_dim > 0 else 0)] + [w_dim] * n_layers
        self.net = nn.Sequential(*[EqualLinear(a, b, lr_mul=lr_mul, activation=True)
                                   for a, b in zip(dims[:-1], dims[1:])])
        self.register_buffer("w_avg", torch.zeros(w_dim))
        self.w_avg_beta = w_avg_beta

    def forward(self, z, c=None, truncation=1.0, track_w_avg=False):
        z = z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)  # pixel norm
        if self.c_dim > 0:
            if c is None:
                raise ValueError("this mapping network is class-conditional: pass one-hot labels c")
            y = self.embed(c.float())
            z = torch.cat([z, y * torch.rsqrt(y.pow(2).mean(dim=1, keepdim=True) + 1e-8)], dim=1)
        w = self.net(z)
        if track_w_avg:
            self.w_avg.copy_(w.detach().mean(0).lerp(self.w_avg, self.w_avg_beta))
        if truncation != 1.0:
            w = self.w_avg.lerp(w, truncation)
        return w


class ModulatedConv2d(nn.Module):
    """The core StyleGAN2 operation: a convolution whose weights are scaled per sample.

    An affine layer maps w to one style per input channel; the weights are multiplied by
    the styles (modulation) and then rescaled so each output feature map has unit expected
    variance (demodulation), which replaces StyleGAN1's AdaIN and removes its droplet
    artefacts. Implemented as one grouped convolution with a separate kernel per sample.
    """

    def __init__(self, in_ch, out_ch, kernel, w_dim, demodulate=True, upsample=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.affine = EqualLinear(w_dim, in_ch, bias_init=1.0)  # styles start at 1
        self.scale = 1.0 / math.sqrt(in_ch * kernel * kernel)
        self.padding = kernel // 2
        self.demodulate = demodulate
        self.upsample = upsample
        self.in_ch, self.out_ch = in_ch, out_ch

    def forward(self, x, w):
        b = x.size(0)
        styles = self.affine(w).view(b, 1, self.in_ch, 1, 1)
        weight = self.scale * self.weight * styles  # (b, out, in, k, k)
        if self.demodulate:
            weight = weight * torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = x.reshape(1, b * self.in_ch, *x.shape[2:])
        out = F.conv2d(x, weight.view(b * self.out_ch, self.in_ch, *weight.shape[3:]),
                       padding=self.padding, groups=b)
        return out.view(b, self.out_ch, *out.shape[2:])


class StyledLayer(nn.Module):
    """Modulated 3x3 convolution + per-pixel noise (learned strength) + bias + activation."""

    def __init__(self, in_ch, out_ch, w_dim, upsample=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, 3, w_dim, upsample=upsample)
        self.noise_strength = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x, w, noise=None):
        x = self.conv(x, w)
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return lrelu(x + self.noise_strength * noise + self.bias)


class ToImage(nn.Module):
    """1x1 modulated convolution (no demodulation) to the image channels, added to the
    upsampled image from the previous resolution (skip-connection generator)."""

    def __init__(self, in_ch, w_dim, img_channels=1):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, img_channels, 1, w_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, img_channels, 1, 1))

    def forward(self, x, w, skip=None):
        out = self.conv(x, w) + self.bias
        if skip is not None:
            out = out + F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
        return out


def stylegan_channels(resolution, channel_base, max_channels):
    """Feature maps per resolution: channel_base / res, capped at max_channels."""
    return {2 ** i: min(max_channels, channel_base // 2 ** i) for i in range(2, int(math.log2(resolution)) + 1)}


class StyleGAN2Generator(nn.Module):
    """z -> image in roughly [-1, 1] (no tanh, as in StyleGAN2).

    A learned 4 x 4 constant is refined by two styled layers per resolution (the first
    upsamples); every resolution adds its ToImage output to the upsampled image so far.
    Each layer receives its own copy of w, so different layers can take different w
    vectors (style mixing): `ws` has shape (batch, num_ws, w_dim).
    """

    def __init__(self, resolution=64, z_dim=512, w_dim=512, mapping_layers=8, channel_base=8192,
                 max_channels=256, img_channels=1, c_dim=0):
        super().__init__()
        _stage_channels(resolution, 1)  # validates the resolution
        self.resolution, self.z_dim, self.w_dim = resolution, z_dim, w_dim
        self.latent_dim = z_dim  # common name used by train.py / predict.py
        ch = stylegan_channels(resolution, channel_base, max_channels)
        self.c_dim = c_dim
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers, c_dim=c_dim)
        self.const = nn.Parameter(torch.randn(1, ch[4], 4, 4))
        self.layers = nn.ModuleList([StyledLayer(ch[4], ch[4], w_dim)])
        self.to_images = nn.ModuleList([ToImage(ch[4], w_dim, img_channels)])
        res = 8
        while res <= resolution:
            self.layers.append(StyledLayer(ch[res // 2], ch[res], w_dim, upsample=True))
            self.layers.append(StyledLayer(ch[res], ch[res], w_dim))
            self.to_images.append(ToImage(ch[res], w_dim, img_channels))
            res *= 2
        # One w per styled layer, plus one for each ToImage (which shares the w of the
        # layer before it, as in the paper's indexing).
        self.num_ws = len(self.layers) + 1

    def synthesis(self, ws, noise_mode="random"):
        """Run the synthesis network on per-layer styles `ws` (B, n_ws, w_dim); noise_mode 'none' disables per-pixel noise."""
        b = ws.size(0)
        x = self.const.expand(b, -1, -1, -1)
        zero_noise = noise_mode == "none"  # "none" switches per-pixel noise off (deterministic output)
        x = self.layers[0](x, ws[:, 0], torch.zeros(b, 1, 4, 4, device=ws.device) if zero_noise else None)
        img = self.to_images[0](x, ws[:, 1])
        i = 1
        for to_image in self.to_images[1:]:
            for _ in range(2):
                layer = self.layers[i]
                size = x.size(2) * (2 if layer.conv.upsample else 1)
                n = torch.zeros(b, 1, size, size, device=ws.device) if zero_noise else None
                x = layer(x, ws[:, i], n)
                i += 1
            img = to_image(x, ws[:, i], skip=img)
        return img

    def forward(self, z, c=None, truncation=1.0, track_w_avg=False, return_ws=False, noise_mode="random"):
        w = self.mapping(z, c, truncation=truncation, track_w_avg=track_w_avg)
        ws = w.unsqueeze(1).expand(-1, self.num_ws, -1)
        img = self.synthesis(ws, noise_mode)
        return (img, ws) if return_ws else img


class DiscriminatorBlock(nn.Module):
    """Residual block: two 3x3 convolutions and 2x downsampling, plus a downsampled
    1x1 skip path; the sum is divided by sqrt(2) to keep the variance unchanged."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = EqualConv2d(in_ch, in_ch, 3)
        self.conv2 = EqualConv2d(in_ch, out_ch, 3)
        self.skip = EqualConv2d(in_ch, out_ch, 1, bias=False, activation=False)

    def forward(self, x):
        skip = self.skip(F.avg_pool2d(x, 2))
        x = F.avg_pool2d(self.conv2(self.conv1(x)), 2)
        return (x + skip) / SQRT2


class StyleGAN2Discriminator(nn.Module):
    """Residual discriminator: image -> one unbounded score per image.

    At 4 x 4, minibatch standard deviation appends one feature map holding the average
    spread of features across a small group of images, so the discriminator can detect a
    lack of variety (a guard against mode collapse).
    """

    def __init__(self, resolution=64, channel_base=8192, max_channels=256, img_channels=1, mbstd_group=4, c_dim=0):
        super().__init__()
        ch = stylegan_channels(resolution, channel_base, max_channels)
        self.from_image = EqualConv2d(img_channels, ch[resolution], 1)
        blocks, res = [], resolution
        while res > 4:
            blocks.append(DiscriminatorBlock(ch[res], ch[res // 2]))
            res //= 2
        self.blocks = nn.Sequential(*blocks)
        self.mbstd_group = mbstd_group
        self.conv = EqualConv2d(ch[4] + 1, ch[4], 3)
        self.fc = EqualLinear(ch[4] * 16, ch[4], activation=True)
        self.out = EqualLinear(ch[4], 1)
        self.c_dim = c_dim
        # Projection discriminator: score += <embed(c), features> / sqrt(features).
        self.embed = EqualLinear(c_dim, ch[4]) if c_dim > 0 else None

    def minibatch_std(self, x):
        """Append one feature map holding the mean standard deviation across each group of samples (exposes low variety)."""
        b, c, h, w = x.shape
        g = min(self.mbstd_group, b)
        while b % g:  # group size must divide the batch
            g -= 1
        y = x.view(g, -1, c, h, w)
        y = (y - y.mean(0)).pow(2).mean(0).add(1e-8).sqrt()  # std over each group
        y = y.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1).repeat(g, 1, h, w)  # one value per group
        return torch.cat([x, y], dim=1)

    def forward(self, img, c=None):
        x = self.blocks(self.from_image(img))
        x = self.conv(self.minibatch_std(x))
        h = self.fc(x.flatten(1))
        score = self.out(h).view(-1)
        if self.c_dim > 0:
            if c is None:
                raise ValueError("this discriminator is class-conditional: pass one-hot labels c")
            score = score + (self.embed(c.float()) * h).sum(dim=1) / math.sqrt(h.size(1))
        return score


# --------------------------------------------------------------------------- #
# StyleGAN2 losses
# --------------------------------------------------------------------------- #
def d_logistic_loss(real_scores, fake_scores):
    """Discriminator loss: softplus(D(fake)) + softplus(-D(real))."""
    return F.softplus(fake_scores).mean() + F.softplus(-real_scores).mean()


def g_nonsaturating_loss(fake_scores):
    """Non-saturating generator loss: softplus(-D(G(z)))."""
    return F.softplus(-fake_scores).mean()


def r1_penalty(discriminator, real, c=None):
    """R1 regularisation: squared gradient norm of D at real images (Mescheder et al., 2018).

    Returns the mean of ||grad D(real)||^2; the caller multiplies by gamma / 2.
    """
    real = real.detach().requires_grad_(True)
    scores = discriminator(real) if c is None else discriminator(real, c)
    grad, = torch.autograd.grad(scores.sum(), real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()


# --------------------------------------------------------------------------- #
# Adaptive discriminator augmentation (ADA)
# --------------------------------------------------------------------------- #
class AdaptiveAugment(nn.Module):
    """Differentiable augmentation applied to every discriminator input, with adaptive p.

    Each augmentation is applied to each image independently with probability p, so for
    p < 1 the discriminator still sees un-augmented images and the generator is not
    pushed to produce augmented-looking output (augmentations do not "leak").

    p adapts to r_t = E[sign(D(real))], the share of real images the discriminator is
    confidently right about: above `target` the discriminator is overfitting, so p rises;
    below it, p falls. p can move from 0 to 1 within `ada_kimg` thousand real images.
    """

    def __init__(self, target=0.6, ada_kimg=500, translate=0.125, scale_std=0.2, rotate_max=math.pi,
                 brightness_std=0.2, contrast_std=0.5, fixed_p=None):
        super().__init__()
        # fixed_p: hold p constant (plain differentiable augmentation, no adaptation); r_t is still measured.
        self.register_buffer("p", torch.tensor(float(fixed_p) if fixed_p is not None else 0.0))
        self.adaptive = fixed_p is None
        self.target, self.ada_kimg = target, ada_kimg
        self.translate, self.scale_std, self.rotate_max = translate, scale_std, rotate_max
        self.brightness_std, self.contrast_std = brightness_std, contrast_std
        self._signs, self._count = 0.0, 0

    def forward(self, x):
        p = float(self.p)
        if p <= 0:
            return x
        b, device = x.size(0), x.device

        def chosen():
            return (torch.rand(b, device=device) < p).float()

        # Geometric: one affine warp per image combining translation, scaling and rotation.
        tx = chosen() * torch.round((torch.rand(b, device=device) * 2 - 1) * self.translate * x.size(3)) * 2 / x.size(3)
        ty = chosen() * torch.round((torch.rand(b, device=device) * 2 - 1) * self.translate * x.size(2)) * 2 / x.size(2)
        log_s = chosen() * torch.randn(b, device=device) * self.scale_std * math.log(2)
        angle = chosen() * (torch.rand(b, device=device) * 2 - 1) * self.rotate_max
        inv_s = torch.exp(-log_s)  # affine_grid maps output to input coordinates
        cos, sin = torch.cos(angle) * inv_s, torch.sin(angle) * inv_s
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        # Shift so the black background (-1) maps to 0 before zero padding, then shift back.
        x = F.grid_sample(x + 1, grid, mode="bilinear", padding_mode="zeros", align_corners=False) - 1

        # Intensity: brightness offset, then contrast scaling about each image's mean.
        x = x + (chosen() * torch.randn(b, device=device) * self.brightness_std).view(b, 1, 1, 1)
        c = torch.exp(chosen() * torch.randn(b, device=device) * self.contrast_std * math.log(2)).view(b, 1, 1, 1)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return (x - mean) * c + mean

    def accumulate(self, real_scores):
        """Record sign(D(real)) for the next p update."""
        self._signs += float(torch.sign(real_scores.detach()).sum())
        self._count += real_scores.numel()

    def update(self, batch_size, interval):
        """Adjust p from r_t over the images seen since the last update; returns r_t."""
        if self._count == 0:
            return float("nan")
        r_t = self._signs / self._count
        if self.adaptive:
            step = batch_size * interval / (self.ada_kimg * 1000)
            self.p.copy_((self.p + math.copysign(step, r_t - self.target)).clamp(0.0, 1.0))
        self._signs, self._count = 0.0, 0
        return r_t