import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img




def test_resnet_classification_result(model, image_path):
    # 加载待分类的图像
    image = Image.open(image_path)

    # 图像预处理
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    # 使用加载的模型进行图像分类
    with torch.no_grad():
        output = model(input_batch)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = predicted_idx.item()

    # 打印预测类别
    print("Predicted class:", predicted_class)

def main():
    # 定义图像预处理的转换函数
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载预训练的 ResNet 模型
    model = None
    model = models.resnet18()

    fc_input_feature = model.fc.in_features
    model.fc = torch.nn.Linear(fc_input_feature, 2)
    model.load_state_dict(
        torch.load('/Users/luopeiyuan/Desktop/FYP/FYP_Codes/ImageClassifier/ResNet-1/MyModel/resnet18_Cat_Dog.pth'))
    model.eval()
    target_layers = [model.layer4[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/ImageClassifier/CNN/data/newtest/cat.10000.jpg"  # "both.png"
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