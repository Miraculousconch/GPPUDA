import math
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from target_model import Generator32
from source_model import *
from PIL import Image
import ssl

channel = 1
batch_size = 128
epochs = 20
step = 300
ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_1 = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])
transform_3 = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

if channel == 1:
    transform = transform_1
else:
    transform = transform_3

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, drop_last=True)


def T_train(G, C, optimizer, epoch, step, batch_size):
    G.train()
    C.train()
    for batch_idx in range(step):
        optimizer.zero_grad()
        with torch.no_grad():
            z_ = torch.randn(batch_size, 128).to(device)
            y_ = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).to(device)
            y_label_ = torch.zeros(batch_size, 10).to(device)
            y_label_.scatter_(1, y_.view(batch_size, 1), 1)
        G_result = G(z_, y_label_).detach()
        y_label_c_ = y_.view(batch_size).to(device)
        C_result = C(G_result)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(C_result, y_label_c_)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

    torch.save(C.state_dict(), 'Final_model_parameter/C_' + str(channel) + '.pkl')
    torch.save(optimizer.state_dict(), 'Final_model_parameter/optimizer_' + str(channel) + '.pkl')


def T_test(G, C, test_loader):
    G.eval()
    C.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = C(data)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            test_loss += criterion(output, label)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("Test set average loss: {:.4f}\tAccuracy: {}/{} ({:.2f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

if __name__ == '__main__':
    G = Generator32().to(device)
    G.load_state_dict(torch.load('./Target_model_parameter/generator_param_' + str(channel) + '.pkl'))

    C = ConvNet().to(device)
    C_optimizer = optim.Adam(C.parameters())

    for epoch in range(1, epochs + 1):
        T_train(G, C, C_optimizer, epoch, step, batch_size)
        T_test(G, C, test_loader)

