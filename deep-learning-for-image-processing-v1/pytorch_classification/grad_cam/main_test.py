# -*- coding: utf-8 -*-
import os

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from utils import GradCAM, show_cam_on_image, center_crop_img
import matplotlib.pyplot as plt




def Myloader(path):
    return Image.open(path).convert('RGB')


# 得到一个包含路径与标签的列表
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])

    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0


def load_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    path1 = 'data/newtrain/cat.%d.jpg'
    data1 = init_process(path1, [0, 500])
    path2 = 'data/newtrain/dog.%d.jpg'
    data2 = init_process(path2, [0, 500])
    path3 = 'data/newtest/cat.%d.jpg'
    data3 = init_process(path3, [12000, 12200])
    path4 = 'data/newtest/dog.%d.jpg'
    data4 = init_process(path4, [12000, 12200])
    # 1300个训练
    train_data = data1 + data2 + data3[0:150] + data4[0:150]

    train = MyDataset(train_data, transform=transform, loder=Myloader)
    # 100个测试
    test_data = data3[150:200] + data4[150:200]
    test = MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, batch_size=10, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0)

    return train_data, test_data


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()  # 继承__init__功能
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 第四层卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output = nn.Linear(in_features=256 * 14 * 14, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        temp = x.view(x.shape[0], -1)
        output = self.output(temp)
        print(x.shape)  # 打印卷积层输出形状
        return output, x


def train():
    train_loader, test_loader = load_data()
    epoch_num = 30
    # GPU计算
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()  # 梯度清0
            output = model(data)[0]  # 前向传播
            loss = criterion(output, target)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        torch.save(model, './MyModel/cnn.pkl')


def test():
    train_loader, test_loader = load_data()
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = torch.load('cnn.pkl')  # load model
    total = 0
    current = 0
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]

        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()

    print('Accuracy:%d%%' % (100 * current / total))

def main():
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = torch.load('/Users/luopeiyuan/Desktop/FYP/FYP_Codes/ImageClassifier/CNN/MyModel/cnn.pkl')  # load model
    model.forward()
    target_layers = [model.conv4]



    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/ImageClassifier/CNN/data/newtest/cat.10000.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()

if __name__ == '__main__':
    main()