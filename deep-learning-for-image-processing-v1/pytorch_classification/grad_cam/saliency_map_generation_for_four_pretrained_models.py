import glob
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from tqdm import tqdm

from utils import GradCAM, show_cam_on_image, center_crop_img


def main1():
    model = models.mobilenet_v3_large(pretrained=True)
    target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    target_layers = [model.features]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/ImageClassifier/CNN/data/newtest/cat.10000.jpg"# "both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    img_tensor = data_transform(img)

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

def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    model = models.regnet_y_800mf(pretrained=True)
    target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]


    cat_image_folder = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/RegNet_IMAGE_Folder/CatOriginalImage'
    dog_image_folder = '/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/RegNet_IMAGE_Folder/DogOriginalImage'
    image_folder = cat_image_folder
    # 获取目录下所有图像文件的路径
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

    for image_path in tqdm(image_paths):
        img_path = image_path
        assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

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
        saved_cat_image_dir = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/RegNet_IMAGE_Folder/CatImageWithMaps"
        saved_dog_image_dir = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/RegNet_IMAGE_Folder/DogImageWithMaps"
        saved_image_dir = saved_cat_image_dir
        saved_img_path = saved_image_dir + "/" + filename
        print(filename)
        print(saved_image_dir)
        print(saved_img_path)
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
        visualization_image = Image.fromarray((visualization * 255).astype(np.uint8))
        visualization_image.save(saved_img_path)

        # plt.imshow(visualization)
        # plt.show()
        saved_cat_saliency_dir = "/Users/luopeiyuan/Desktop/FYP/FYP_Codes/deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/RegNet_IMAGE_Folder/CatSaliencyData"
        saliency_file_name = filename.split('.')[1]
        saved_saliency_path = saved_cat_saliency_dir + "/" + saliency_file_name
        np.save(saved_saliency_path, visualization)



if __name__ == '__main__':
    main()
