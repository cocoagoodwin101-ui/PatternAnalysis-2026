"""Model components, implemented in PyTorch only.

WGAN-GP baseline: ported from my COMP3710 Lab 2 Task 3 OASIS model
(github.com/cocoagoodwin101-ui/comp3710-lab2-pattern-recognition, task3_gan/train_gan_oasis.py),
which trained stably on OASIS at 64 x 64 without mode collapse. At 64 x 64 the layers are
identical to Lab 2. The only change is that the number of up/downsampling stages now follows
the resolution, so the same baseline can be compared with StyleGAN2 at 128 and 256.

StyleGAN2 components will be added here.
"""

import math

import torch
import torch.nn as nn

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
    return sum(p.numel() for p in model.parameters())