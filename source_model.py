import os
import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import ssl
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import warnings

channel = 1
batch_size = 128
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=UserWarning)


transform_1 = transforms.Compose([
    transforms.Resize((32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))])

transform_3 = transforms.Compose([
    transforms.Resize((32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

if channel == 1:
    transform = transform_1
else:
    transform = transform_3

train_loader = torch.utils.data.DataLoader(
    datasets.USPS('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.USPS('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, drop_last=True)

class ConditionalBatchNorm(nn.Module):

    def __init__(self, *args, n_domains=1, bn_func=nn.BatchNorm2d, **kwargs):
        super(ConditionalBatchNorm, self).__init__()

        self.n_domains = n_domains
        self.layers = [bn_func(*args, **kwargs) for i in range(n_domains)]

    def _apply(self, fn):
        super(ConditionalBatchNorm, self)._apply(fn)
        for layer in self.layers:
            layer._apply(fn)

    def parameters(self, d=0):
        return self.layers[d].parameters()

    def forward(self, x, d):
        layer = self.layers[d]
        return layer(x)


class ConvNet(nn.Module):

    def __init__(self, n_classes=10, n_domains=2):
        super(ConvNet, self).__init__()

        self.conditional_layers = []
        self.n_domains = n_domains

        self.norm = nn.InstanceNorm2d(channel, affine=False,
                                      momentum=0,
                                      track_running_stats=False)

        self.conv1_1 = nn.Conv2d(channel, 128, (3, 3), padding=1)
        self.conv1_1_bn = self._batch_norm(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = self._batch_norm(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = self._batch_norm(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = self._batch_norm(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = self._batch_norm(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = self._batch_norm(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = self._batch_norm(512)
        self.nin3_2 = nn.Conv2d(512, 256, (3, 3), padding=1)
        self.nin3_2_bn = self._batch_norm(256)
        self.nin3_3 = nn.Conv2d(256, 128, (3, 3), padding=1)
        self.nin3_3_bn = self._batch_norm(128)

        self.fc4 = nn.Linear(128, n_classes)

    def _batch_norm(self, *args, **kwargs):

        layer = ConditionalBatchNorm(*args, n_domains=self.n_domains, **kwargs)
        self.conditional_layers.append(layer)
        return layer

    def __call__(self, x, d=0):

        return self.forward(x, d)

    def forward(self, x, d=0):
        x = self.norm(x)

        x = F.relu(self.conv1_1_bn(self.conv1_1(x), d))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x), d))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x), d)))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x), d))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x), d))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x), d)))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x), d))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x), d))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x), d))

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)

        x = self.fc4(x)
        return x

    def conditional_params(self, d=0):
        for module in self.conditional_layers:
            for p in module.parameters(d):
                yield p

    def parameters(self, d=0, yield_shared=True, yield_conditional=True):

        if yield_shared:
            for param in super(ConvNet, self).parameters():
                yield param

        if yield_conditional:
            for param in self.conditional_params(d):
                yield param


class Extractor(nn.Module):
    def __init__(self, n_domains=2):
        super().__init__()
        self.conditional_layers = []
        self.n_domains = n_domains

        self.norm = nn.InstanceNorm2d(channel, affine=False,
                                      momentum=0,
                                      track_running_stats=False)

        self.conv1_1 = nn.Conv2d(channel, 128, (3, 3), padding=1)
        self.conv1_1_bn = self._batch_norm(128)
        self.conv1_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_2_bn = self._batch_norm(128)
        self.conv1_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv1_3_bn = self._batch_norm(128)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.conv2_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv2_1_bn = self._batch_norm(256)
        self.conv2_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_2_bn = self._batch_norm(256)
        self.conv2_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv2_3_bn = self._batch_norm(256)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.drop2 = nn.Dropout()

        self.conv3_1 = nn.Conv2d(256, 512, (3, 3), padding=0)
        self.conv3_1_bn = self._batch_norm(512)
        self.nin3_2 = nn.Conv2d(512, 256, (3, 3), padding=1)
        self.nin3_2_bn = self._batch_norm(256)
        self.nin3_3 = nn.Conv2d(256, 128, (3, 3), padding=1)
        self.nin3_3_bn = self._batch_norm(128)

    def _batch_norm(self, *args, **kwargs):
        layer = ConditionalBatchNorm(*args, n_domains=self.n_domains, **kwargs)
        self.conditional_layers.append(layer)
        return layer

    def __call__(self, x, d=0):
        return self.forward(x, d)

    def forward(self, x, d=0):
        x = self.norm(x)

        x = F.relu(self.conv1_1_bn(self.conv1_1(x), d))
        x = F.relu(self.conv1_2_bn(self.conv1_2(x), d))
        x = self.pool1(F.relu(self.conv1_3_bn(self.conv1_3(x), d)))
        x = self.drop1(x)

        x = F.relu(self.conv2_1_bn(self.conv2_1(x), d))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x), d))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x), d)))
        x = self.drop2(x)

        x = F.relu(self.conv3_1_bn(self.conv3_1(x), d))
        x = F.relu(self.nin3_2_bn(self.nin3_2(x), d))
        x = F.relu(self.nin3_3_bn(self.nin3_3(x), d))

        x = F.avg_pool2d(x, 6)
        x = x.view(-1, 128)

        return x

    def conditional_params(self, d=0):
        for module in self.conditional_layers:
            for p in module.parameters(d):
                yield p

    def parameters(self, d=0, yield_shared=True, yield_conditional=True):

        if yield_shared:
            for param in super(Extractor,self).parameters():
                yield param

        if yield_conditional:
            for param in self.conditional_params(d):
                yield param


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc4(x)

        return x


def M_train(extractor, classifier, device, train_loader, optimizer, epoch):
    extractor.train()
    classifier.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = classifier(extractor(data))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
    torch.save(extractor.state_dict(), 'Source_model_parameter/extractor_' + str(channel) + '.pkl')
    torch.save(classifier.state_dict(), 'Source_model_parameter/classifier_' + str(channel) + '.pkl')
    torch.save(optimizer.state_dict(), 'Source_model_parameter/optimizer_' + str(channel) + '.pkl')


def M_test(extractor, classifier, device, test_loader):
    extractor.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = classifier(extractor(data))
            criterion = nn.CrossEntropyLoss(reduction='mean')
            test_loss += criterion(output, label)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def M_test1(extractor, classifier, device, test_loader):
    extractor.eval()
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = classifier(extractor(data))
            criterion = nn.CrossEntropyLoss(reduction='mean')
            test_loss += criterion(output, label)  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    extractor = Extractor().to(device)
    classifier = Classifier().to(device)
    optimizer = optim.Adam([{'params': extractor.parameters()}, {'params': classifier.parameters()}])

    for epoch in range(1, epochs + 1):
        M_train(extractor, classifier, device, train_loader, optimizer, epoch)
        M_test(extractor, classifier, device, test_loader)

