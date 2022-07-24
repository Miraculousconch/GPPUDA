import os, time
import matplotlib.pyplot as plt
import pickle
# import imageio
import ssl
import numpy as np
import torch
import torch.optim as optim
from source_model import *
from target_model import *
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
channel = 1
batch_size = 64
lr = 0.0002
train_epoch = 20
train_step = 20000
alpha = 10
k_lap = 0.1
k_ce = 1
k_gen = 0.1
start_adaptation = 0
start_train_C = 0
epsilon = 5

# fixed noise & label
temp_z_ = torch.randn(10, 128)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_z_, fixed_y_label_ = fixed_z_.to(device), fixed_y_label_.to(device)

# data_loader
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

source_loader = torch.utils.data.DataLoader(
    datasets.USPS('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

# network
G = Generator32().to(device)
D = Discriminator32().to(device)
C_extractor = Extractor().to(device)
C_classifier = Classifier().to(device)

loss_fn = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
selfEntropy = condition_entropy()
feature_distance = feature_dis()
l2Norm = l2_norm()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))
C_optimizer = optim.Adam([{'params': C_extractor.parameters()}, {'params': C_classifier.parameters()}])

sched_G = optim.lr_scheduler.LambdaLR(
    G_optimizer, lambda step: 1 - step / train_step)
sched_D = optim.lr_scheduler.LambdaLR(
    D_optimizer, lambda step: 1 - step / train_step)

if os.path.exists('./Source_model_parameter/extractor_' + str(channel) + '.pkl'):
    C_extractor.load_state_dict(torch.load('./Source_model_parameter/extractor_' + str(channel) + '.pkl'))  # 加载模型参数
    C_classifier.load_state_dict(torch.load('./Source_model_parameter/classifier_' + str(channel) + '.pkl'))  # 加载模型参数
    C_optimizer.load_state_dict(torch.load('./Source_model_parameter/optimizer_' + str(channel) + '.pkl'))

# if os.path.exists('Target_model_parameter/discriminator_param_' + str(channel) + '.pkl'):
#     G.load_state_dict(torch.load('./Target_model_parameter/generator_param_' + str(channel) + '.pkl'))
#     D.load_state_dict(torch.load('./Target_model_parameter/discriminator_param_' + str(channel) + '.pkl'))

# results save folder
root = 'USPS_MNIST_results/'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results_' + str(channel)):
    os.mkdir(root + 'Fixed_results_' + str(channel))

train_hist = {}

train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['C_tar_losses'] = []
train_hist['C_losses'] = []
train_hist['C_src_losses'] = []

train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')

start_time = time.time()

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    C_losses = []
    C_src_losses = []
    C_tar_losses = []

    if (epoch + 1) == 2:
        start_adaptation = 1

    if (epoch + 1) == 8:
        start_train_C = 1

    epoch_start_time = time.time()
    target_looper = infiniteloop(train_loader)
    source_looper = infiniteloop(source_loader)
    for batch_idx in range(len(train_loader)):
        if batch_idx % 100 == 0:
            print("\nBatch index: ", batch_idx)
        for i in range(1):
            D_optimizer.zero_grad()

            with torch.no_grad():
                z_ = torch.randn(batch_size, 128).to(device)
                y_d = (torch.rand(batch_size, 1) * 10).type(torch.LongTensor).to(device)
                y_label_ = torch.zeros(batch_size, 10).to(device)
                y_label_.scatter_(1, y_d.view(batch_size, 1), 1)
                G_result = G(z_, y_label_).detach()

            x_, y_ = next(target_looper)
            x_, y_ = x_.to(device), y_.to(device)
            label_real = torch.ones(batch_size,1).to(device)
            label_fake = torch.zeros(batch_size,1).to(device)
            D_real = D(x_)
            D_fake = D(G_result)
            D_real_loss = loss_fn(D_real,label_real)
            D_fake_loss = loss_fn(D_fake,label_fake)
            gp = gradient_penalty(x_, D)
            loss = D_real_loss + D_fake_loss + gp * 10

            D_train_loss = loss

            D_train_loss.backward()
            D_optimizer.step()
            D_losses.append(D_train_loss.item())
        if batch_idx % 100 == 0:
            print("Discriminator Loss: ", D_train_loss.item())

        if start_train_C:
            for i in range(1):
                x_s, y_s = next(source_looper)
                x_s, y_s = x_s.to(device), y_s.to(device)
                C_optimizer.zero_grad()

                y_label_c_ = y_d.view(batch_size).to(device)
                C_src_result = C_classifier(C_extractor(x_s))
                C_src_loss = criterion(C_src_result, y_s)
                C_src_losses.append(C_src_loss.item())
                if batch_idx % 100 == 0:
                    print("C_source Loss: ", C_src_loss.item())
                G_result = G(z_, y_label_)
                G_result = G_result + torch.tensor(
                    np.random.laplace(0, 2 / epsilon, (batch_size, channel, 32, 32))).to(
                    device)
                G_result = G_result.type(torch.float)
                C_result = C_classifier(C_extractor(G_result))
                C_result_softmax = F.softmax(C_result, dim=1)
                C_cEntropy = selfEntropy(C_result_softmax)
                C_loss = criterion(C_result, y_label_c_)
                C_train_loss = C_src_loss + k_lap * C_loss + k_ce * C_cEntropy
                C_train_loss.backward()
                C_optimizer.step()
                C_losses.append(C_train_loss.item())
            if batch_idx % 100 == 0:
                print("C_train Loss: ", C_train_loss.item())

        for i in range(1):
            G_optimizer.zero_grad()

            x_s, y_s = next(source_looper)
            x_s, y_s = x_s.to(device), y_s.to(device)

            z_ = torch.randn(batch_size, 128).to(device)

            y_d = y_s.view(batch_size, 1)
            y_label_ = torch.zeros(batch_size, 10).to(device)
            y_label_.scatter_(1, y_d.view(batch_size, 1), 1)
            y_label_c_ = y_d.view(batch_size).to(device)

            G_result = G(z_, y_label_)
            D_result = D(G_result)
            label_real = torch.ones(batch_size, 1).to(device)
            G_loss = loss_fn(D_result, label_real)

            if start_adaptation:
                G_result = G(z_, y_label_)
                G_result = G_result + torch.tensor(
                    np.random.laplace(0, 2 / epsilon, (batch_size, channel, 32, 32))).to(
                    device)
                G_result = G_result.type(torch.float)
                C_tar_feat = C_extractor(G_result)
                C_result = C_classifier(C_tar_feat)
                C_src_feat = C_extractor(x_s)
                feat_dis = feature_distance(C_tar_feat, C_src_feat)
                C_train_loss = criterion(C_result, y_label_c_)
                C_tar_losses.append(C_train_loss.item())
                G_train_loss = G_loss + k_gen * (C_train_loss + feat_dis)
            else:
                G_train_loss = G_loss

            G_train_loss.backward()
            G_optimizer.step()
            G_losses.append(G_train_loss.item())
        if batch_idx % 100 == 0:
            if start_adaptation:
                print("C_target Loss: ", C_train_loss.item())
            print("Generator Loss: ", G_train_loss.item())

        sched_G.step()
        sched_D.step()

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('\n[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, loss_c: %.3f, loss_c_src: %.3f, loss_c_tar: %.3f' % (
        (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(C_losses)),
        torch.mean(torch.FloatTensor(C_src_losses)), torch.mean(torch.FloatTensor(C_tar_losses))))
    fixed_p = root + 'Fixed_results_' + str(channel) + '/' + str(epoch + 1) + '.png'
    show_result((epoch + 1), fixed_z_, fixed_y_label_, G, save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['C_losses'].append(torch.mean(torch.FloatTensor(C_losses)))
    train_hist['C_src_losses'].append(torch.mean(torch.FloatTensor(C_src_losses)))
    train_hist['C_tar_losses'].append(torch.mean(torch.FloatTensor(C_tar_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    torch.save(G.state_dict(), './Target_model_parameter/generator_param_' + str(channel) + '.pkl')
    torch.save(D.state_dict(), './Target_model_parameter/discriminator_param_' + str(channel) + '.pkl')
    torch.save(C_extractor.state_dict(),
               './Source_adaptation_parameter/source_extractor_param_' + str(channel) + '.pkl')
    torch.save(C_classifier.state_dict(),
               './Source_adaptation_parameter/source_classifier_param_' + str(channel) + '.pkl')
    with open(root + 'train_hist_' + str(channel) + '.pkl', 'wb') as f:
        pickle.dump(train_hist, f)
    print("Training results saved")

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
show_train_hist(train_hist, save=True, path=root + 'train_hist_' + str(channel) + '.png')

