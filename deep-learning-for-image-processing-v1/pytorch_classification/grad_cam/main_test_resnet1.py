import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from utils import GradCAM, show_cam_on_image, center_crop_img
import os
import glob


def test_resnet_classification_result(model, image_path):
    # 加载待分类的图像
    image = Image.open(image_path)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
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
    # 加载预训练的 ResNet 模型
    model = models.resnet18(pretrained=False)  # 不使用预训练权重
    fc_input_feature = model.fc.in_features
    model.fc = torch.nn.Linear(fc_input_feature, 2)
    model.load_state_dict(torch.load('/Users/luopeiyuan/Desktop/FYP/FYP_Codes/ImageClassifier/ResNet-1/MyModel/resnet18_Cat_Dog.pth'))
    model.eval()

    target_layers = [model.layer4[-1]]
    cat_image_folder = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/CatOriginalImage'
    dog_image_folder = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogOriginalImage'
    image_folder = dog_image_folder
    # 获取目录下所有图像文件的路径
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

    # 打印图像文件路径列表
    for image_path in tqdm(image_paths):
        # print(image_path)
        # 加载图像
        img_path = image_path
        assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        # 图像预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = 1  # 预测的类别索引（0 或 1）

        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]

        directory_path, filename = os.path.split(image_path)
        saved_cat_image_dir = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/CatImageWithMaps"
        saved_dog_image_dir = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogImageWithMaps"
        saved_image_dir = saved_dog_image_dir
        saved_img_path = saved_image_dir + "/" + filename
        print(filename)
        print(saved_image_dir)
        print(saved_img_path)
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
        #visualization_image = Image.fromarray((visualization * 255).astype(np.uint8))
        #visualization_image.save(saved_img_path)

        # plt.imshow(visualization)
        # plt.show()
        saved_dog_saliency_dir = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing/pytorch_classification/grad_cam/ResNet_IMAGE_Folder/DogSaliencyData"
        saliency_file_name = filename.split('.')[1]
        saved_saliency_path = saved_dog_saliency_dir + "/" + saliency_file_name
        np.save(saved_saliency_path, visualization)


if __name__ == '__main__':
    main()