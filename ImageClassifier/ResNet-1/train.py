import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import DogCatDataset


def main():
    # Step 0:查看torch版本、设置device
    print(torch.__version__)
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # Step 1:准备数据集
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = DogCatDataset.DogCatDataset(root_path=os.path.join(os.getcwd(), 'data/newtrain'),
                                             transform=train_transform)
    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

    # Step 2: 初始化模型
    model = models.resnet18()

    # 修改网络结构，将fc层1000个输出改为2个输出
    fc_input_feature = model.fc.in_features
    model.fc = nn.Linear(fc_input_feature, 2)

    # load除最后一层的预训练权重
    pretrained_weight = torch.hub.load_state_dict_from_url(
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth', progress=True)
    del pretrained_weight['fc.weight']
    del pretrained_weight['fc.bias']
    model.load_state_dict(pretrained_weight, strict=False)

    model.to(device)

    # Step 3:设置损失函数
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # Step 4:选择优化器
    LR = 0.01
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    # Step 5:设置学习率下降策略
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Step 6:训练网络
    model.train()
    MAX_EPOCH = 20  # 设置epoch=20

    for epoch in range(MAX_EPOCH):
        loss_log = 0
        total_sample = 0
        train_correct_sample = 0
        for data in tqdm(train_dataloader):
            img, label = data
            img, label = img.to(device), label.to(device)
            output = model(img)

            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

            _, predicted_label = torch.max(output, 1)

            total_sample += label.size(0)
            train_correct_sample += (predicted_label == label).cpu().sum().numpy()

            loss_log += loss.item()

            # if total_sample == 2400:
            #     print('mark!')

        # 打印信息
        print('epoch: ', epoch)
        print("accuracy:", train_correct_sample / total_sample)
        print('loss:', loss_log / total_sample)

        scheduler.step()  # 更新学习率
        torch.save(model.state_dict(), './MyModel/resnet18_Cat_Dog.pth')
    print('train finish!')
    # Step 7: 存储权重



if __name__ == '__main__':
    main()