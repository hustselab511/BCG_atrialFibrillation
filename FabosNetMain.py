import random

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

import ToolKit as tk

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# param init
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)


def train(net, train_dataloader, valid_dataloader, device, num_epoch, lr, init=True):
    if init:
        net.apply(init_xavier)

    print('training on:', device)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    train_losses = []
    train_acces = []
    eval_acces = []

    # train
    for epoch in range(num_epoch):
        print("——————Round {} training begins——————".format(epoch + 1))
        net.train()
        train_acc = 0
        for data, label in tqdm(train_dataloader):
            data, label = data.to(device).float(), label.to(device)
            predict = net(data)
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = predict.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / (data.shape[0])
            train_acc += acc

        scheduler.step()
        print("Train Loss: {} , Loss: {}, Acc:{}".format(epoch, loss.item(), train_acc / len(train_dataloader)))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(loss.item())

        # valid
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device).float(), label.to(device)
                predict = net(data)
                loss = criterion(predict, label)
                _, pred = predict.max(1)
                num_correct = (pred == label).sum().item()
                eval_loss += loss
                acc = num_correct / data.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / (len(valid_dataloader))
            eval_acc = eval_acc / len(valid_dataloader)
            if 'best_acc' not in locals() or eval_acc >= best_acc:
                best_acc = eval_acc
                # torch.save(net, 'save/UnitedResNet.pth')
            eval_acces.append(eval_acc)
            print("Valid Dataset Loss: {}".format(eval_losses))
            print("Valid Dataset Accuracy rate: {}".format(eval_acc))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, input):
        return self.conv(input)


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=16),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=16, out_channels=32),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=32, out_channels=64),
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=64, out_channels=128),
        )

    def forward(self, x):
        down1 = self.conv1(x)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        return down1, down2, down3, down4


class SpaceAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

    def forward(self, q_down, k_down, v_down):  # batch, in_channels, H, W (H = W)
        avg_pool_q = torch.mean(q_down, dim=1, keepdim=True)
        avg_pool_k = torch.mean(k_down, dim=1, keepdim=True)
        attention_map = torch.sigmoid(avg_pool_q + avg_pool_k)
        attention_map = torch.nn.functional.normalize(attention_map, p=1)
        return attention_map * v_down


class SpaceAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.att_low = SpaceAttentionBlock(self.in_channels)
        self.att_mid = SpaceAttentionBlock(self.in_channels)
        self.att_high = SpaceAttentionBlock(self.in_channels)

    def forward(self, low_down, mid_down, high_down):
        low = self.att_low(mid_down, high_down, low_down)
        mid = self.att_low(high_down, low_down, mid_down)
        high = self.att_low(low_down, mid_down, high_down)
        return low, mid, high


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool_16 = nn.Sequential(
            ConvBlock(in_channels=16, out_channels=16),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.avgpool_32 = nn.Sequential(
            ConvBlock(in_channels=32, out_channels=32),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.avgpool_64 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.avgpool_128 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
        x1 = self.avgpool_16(x1).squeeze(-1).squeeze(-1)
        x2 = self.avgpool_16(x2).squeeze(-1).squeeze(-1)
        x3 = self.avgpool_16(x3).squeeze(-1).squeeze(-1)
        x4 = self.avgpool_32(x4).squeeze(-1).squeeze(-1)
        x5 = self.avgpool_32(x5).squeeze(-1).squeeze(-1)
        x6 = self.avgpool_32(x6).squeeze(-1).squeeze(-1)
        x7 = self.avgpool_64(x7).squeeze(-1).squeeze(-1)
        x8 = self.avgpool_64(x8).squeeze(-1).squeeze(-1)
        x9 = self.avgpool_64(x9).squeeze(-1).squeeze(-1)
        x10 = self.avgpool_128(x10).squeeze(-1).squeeze(-1)
        x11 = self.avgpool_128(x11).squeeze(-1).squeeze(-1)
        x12 = self.avgpool_128(x12).squeeze(-1).squeeze(-1)
        return torch.cat([x1, x4, x7, x10], dim=-1), torch.cat([x2, x5, x8, x11], dim=-1), torch.cat([x3, x6, x9, x12], dim=-1)


class UnitedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.low_down_sample = DownSample(3)
        self.mid_down_sample = DownSample(3)
        self.high_down_sample = DownSample(3)

        self.down1_space_att = SpaceAttention(16)
        self.down2_space_att = SpaceAttention(16)
        self.down3_space_att = SpaceAttention(16)
        self.down4_space_att = SpaceAttention(16)

        self.channel_att = ChannelAttention()

        self.classify = nn.Sequential(
            nn.Linear(240, num_classes),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        assert x.shape[1] == 9
        low = x[:, :3, :, :]  # low
        mid = x[:, 3:6, :, :]  # mid
        high = x[:, 6:, :, :]  # high
        # down sample
        # torch.Size([16, 64, 224, 224]) torch.Size([16, 128, 112, 112]) torch.Size([16, 256, 56, 56]) torch.Size([16, 512, 28, 28])
        low_down1, low_down2, low_down3, low_down4 = self.low_down_sample(low)
        mid_down1, mid_down2, mid_down3, mid_down4 = self.mid_down_sample(mid)
        high_down1, high_down2, high_down3, high_down4 = self.high_down_sample(high)

        # cross attention
        low_down1, mid_down1, high_down1 = self.down1_space_att(low_down1, mid_down1, high_down1)  # torch.Size([16, 64, 224, 224])
        low_down2, mid_down2, high_down2 = self.down2_space_att(low_down2, mid_down2, high_down2)  # torch.Size([16, 128, 112, 112])
        low_down3, mid_down3, high_down3 = self.down3_space_att(low_down3, mid_down3, high_down3)  # torch.Size([16, 256, 56, 56])
        low_down4, mid_down4, high_down4 = self.down4_space_att(low_down4, mid_down4, high_down4)  # torch.Size([16, 512, 28, 28])

        # norm
        low_f, mid_f, high_f = self.channel_att(low_down1, mid_down1, high_down1, low_down2, mid_down2, high_down2, low_down3, mid_down3, high_down3, low_down4, mid_down4, high_down4)
        feature = low_f + mid_f + high_f
        ans = self.classify(feature)
        return ans


if __name__ == '__main__':
    # freq img
    naf_data_origin = torch.randn(5, 1, 1000)
    af_data_origin = torch.randn(5, 1, 1000)

    naf_data = torch.cat([
        tk.transferAllDataToCWT(tk.flatten_data(naf_data_origin), scale=128, wavelet='morl', fs=125, ylim=[1, 5], imageSize=224),
        tk.transferAllDataToCWT(tk.flatten_data(naf_data_origin), scale=128, wavelet='morl', fs=125, ylim=[5, 9], imageSize=224),
        tk.transferAllDataToCWT(tk.flatten_data(naf_data_origin), scale=128, wavelet='morl', fs=125, ylim=[9, 13], imageSize=224),
    ], dim=1).reshape(naf_data_origin.shape[0], naf_data_origin.shape[1], 9, 224, 224)
    af_data = torch.cat([
        tk.transferAllDataToCWT(tk.flatten_data(af_data_origin), scale=128, wavelet='morl', fs=125, ylim=[1, 5], imageSize=224),
        tk.transferAllDataToCWT(tk.flatten_data(af_data_origin), scale=128, wavelet='morl', fs=125, ylim=[5, 9], imageSize=224),
        tk.transferAllDataToCWT(tk.flatten_data(af_data_origin), scale=128, wavelet='morl', fs=125, ylim=[9, 13], imageSize=224),
    ], dim=1).reshape(af_data_origin.shape[0], af_data_origin.shape[1], 9, 224, 224)

    print(naf_data.shape)  # [2, 2, 9, 224, 224]
    label = torch.cat([tk.produceLabel(naf_data_origin, 0), tk.produceLabel(af_data, 1)], dim=0)

    # split dataset intra-patient
    # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    # split dataset inter-patient
    train_naf_index = random.sample(range(naf_data.shape[0]), int(naf_data.shape[0] * 0.8))
    train_af_index = random.sample(range(af_data.shape[0]), int(af_data.shape[0] * 0.8))
    test_naf_index = list(set(range(naf_data.shape[0])) - set(train_naf_index))
    test_af_index = list(set(range(af_data.shape[0])) - set(train_af_index))
    print("asdasd", torch.cat([naf_data[train_naf_index], af_data[train_af_index]], dim=0).shape)
    x_train = rearrange(torch.cat([naf_data[train_naf_index], af_data[train_af_index]], dim=0), 'a b c d e -> (a b) c d e')
    x_test = rearrange(torch.cat([naf_data[test_naf_index], af_data[test_af_index]], dim=0), 'a b c d e -> (a b) c d e')
    y_train = rearrange(torch.cat([label[train_naf_index], label[train_af_index]], dim=0), 'a b -> (a b)')
    y_test = rearrange(torch.cat([label[test_naf_index], label[test_af_index]], dim=0), 'a b -> (a b)')
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # dataset and DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # init
    net = UnitedResNet(num_classes=2)

    # Train
    train(net=net, train_dataloader=train_loader, valid_dataloader=test_loader, device=torch.device('cuda'), num_epoch=2, lr=1e-4, init=True)
