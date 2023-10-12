import os
import cv2
from torch.utils.data import Dataset


class DogCatDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.label_name = {"Cat": 0, "Dog": 1}
        self.root_path = root_path
        self.transform = transform
        self.get_train_img_info()

    def __getitem__(self, index):
        self.img = cv2.imread(os.path.join(self.root_path, self.train_img_name[index]))
        if self.transform is not None:
            self.img = self.transform(self.img)
        self.label = self.train_img_label[index]
        return self.img, self.label

    def __len__(self):
        return len(self.train_img_name)

    def get_train_img_info(self):
        self.train_img_name = os.listdir(self.root_path)
        self.train_img_label = [0 if 'cat' in imgname else 1 for imgname in self.train_img_name]

