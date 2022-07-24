import math
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from source_model import *
from PIL import Image
from functools import partial

channel = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, M=4):
        super().__init__()
        self.M = M
        self.linear_1 = nn.Linear(128, M * M * 256)
        self.linear_2 = nn.Linear(10, M * M * 256)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh())
        res_arch_init(self)

    def forward(self, z, y):
        x = self.linear_1(z)
        y = self.linear_2(y)
        x = torch.cat([x, y], 1)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(
                channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1, bias=False)
        res_arch_init(self)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self):
        super().__init__()


class Discriminator32(Discriminator):
    def __init__(self):
        super().__init__()


def res_arch_init(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if 'residual' in name:
                init.xavier_uniform_(module.weight, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                init.zeros_(module.bias)


def show_result(num_epoch, fixed_z, fixed_y, G, show=False, save=False, path='result.png'):
    G.eval()
    test_images = G(fixed_z, fixed_y)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    if channel == 3:
        for k in range(10 * 10):
            i = k // 10
            j = k % 10
            ax[i, j].cla()
            img = test_images[k].cpu().data.permute(1, 2, 0).numpy()
            ax[i, j].imshow((img + 1) / 2)
    else:
        size = test_images[0].size(1)
        for k in range(10 * 10):
            i = k // 10
            j = k % 10
            ax[i, j].cla()
            ax[i, j].imshow(test_images[k].cpu().data.view(size, size).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']
    y3 = hist['C_losses']
    y4 = hist['C_src_losses']
    y5 = hist['C_tar_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')
    plt.plot(x, y3, label='C_loss')
    plt.plot(x, y4, label='C_src_loss')
    plt.plot(x, y5, label='C_tar_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def gradient_penalty(x, f):
    # interpolation
    alpha = torch.rand(x.size(0), 1, 1, 1).to(device)
    beta = torch.rand(x.size()).to(device)

    y = x + 0.5 * x.std() * beta
    z = x + alpha * (y - x)

    # gradient penalty
    z.requires_grad = True
    o = f(z)
    g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).to(device), create_graph=True)[0].view(z.size(0),
                                                                                                           -1)
    gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

    return gp


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss


class l2_norm(nn.Module):
    def forward(self, theta1, theta2):
        sum_all = 0
        for key in theta1.keys():
            sum_all += torch.sum(torch.square(theta1[key] - theta2[key]))

        return torch.sqrt(sum_all)


class condition_entropy(nn.Module):
    def forward(self, x):
        n = x.size(0)
        loss = torch.sum(-x * torch.log(x + 10 ** -10)) / n
        return loss


class feature_dis(nn.Module):
    def forward(self, x, y):
        n = x.size(0)
        loss = 0
        for i in range(n):
            loss += torch.norm(x[i] - y[i])
        loss = loss / n
        return loss
