# FYP_Codes
Final Year Project Codes
My GitHub Link for my FYP Code: https://github.com/alfreddLUO/FYP_Codes

## Model Training

E.g Trainning ResNet Model

```shell
python ./ImageClassifier/ResNet-1/train.py 
```

## Grad-CAM Saleincy Map Generation

### GradCAM Saliecny Map For ResNet

```shell
python ./deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/saliency_map_generation_for_resnet.py
```

### GradCAM Saliecny Map For MobileNet-v3, VGG16 RegNet

```shell
python ./deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/saliency_map_generation_for_four_pretrained_models.py
```

## Data Folders
### MobileNet_IMage_Folder, ResNet_IMAGE_Folder, RegNet_IMAGE_Folder, VGG_IMAGE_Folder
Stores the data for the four models.
#### 1. CatOriginalImage: Stores the original cat images
#### 2. CatImageWithMaps: Stores the cat images with saliency map
#### 3. CatSaliencyData: Stores the saliency data for cat image

#### 4. DogOriginalImage: Stores the original dog images
#### 5. DogImageWithMaps: Stores the dog images with saliency map
#### 6. DogSaliencyData: Stores the saliency data for dog image

## GMM Clustering and HMM Conversion

### The code for GMM clustering and HMM conversion is stored in the path: 
```shell
FYP_Codes/deep-learning-for-image-processing-v1/pytorch_classification/grad_cam/Clustering, GMM, and HMM Generation.ipynb
```





