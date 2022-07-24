import math
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from source_model import *
from target_model import Generator32
from PIL import Image

channel = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator32().to(device).eval()
G.load_state_dict(torch.load('./Target_model_parameter/generator_param_' + str(channel) + '.pkl'))

for i in range(10):
    z = torch.randn(1, 128).to(device)
    y = torch.zeros(1, 10).to(device)
    y[0][i] = 1
    img = G(z, y).detach()
    if channel == 3:
        img = img.cpu().data.view(3, 32, 32).permute(1, 2, 0).numpy()
        img = ((img + 1) / 2)
        plt.imshow(img)
        plt.show()
    else:
        img = img.cpu().data.view(32, 32).numpy()
        img = ((img + 1) / 2)
        plt.imshow(img, cmap='gray')
        plt.show()

