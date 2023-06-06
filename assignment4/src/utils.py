import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TF

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_value = np.inf
        self.max_value = 0
        self.mode = mode

    def __call__(self, value): 
        if self.mode == 'min':
            if value < self.min_value:
                self.min_value = value
                self.counter = 0     
            elif value > (self.min_value + self.delta):
                self.counter += 1
                print(f'Early stopping: {self.counter}/{self.patience}')
                
        elif self.mode == 'max':
            if value > self.max_value:
                self.max_value = value
                self.counter = 0     
            elif value < (self.max_value - self.delta):
                self.counter += 1
                print(f'Early stopping: {self.counter}/{self.patience}')
                
        if self.counter >= self.patience:
            return True
        
        # print((self.min_value + self.min_delta))
        
        return False
    

class SignDataset(Dataset):
    def __init__(self, imgs, data_dir, labels_map, transform=None, augment=False):
        self.labels_map = labels_map
        self.imgs = imgs
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.imgs)
    
    def __augment(self, image):
        if random.random() > 0.4:
            angle = random.randint(-10, 10)
            image = TF.rotate(image, angle)
            
        return image

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.imgs[idx])
        image = read_image(img_path, ImageReadMode.RGB).float()
        image = image / 255.0
        label = self.labels_map[self.imgs[idx].split('/')[0]]

        if self.transform:
            image = self.transform(image)
        
        if self.augment:
            image = self.__augment(image)
        
        return image, label
    

def preprocess_img(img):
    image = torch.tensor(img).float().permute(2,0,1)
    image = image / 255.0

    transform = transforms.Compose([transforms.Resize((64+4, 64+4), antialias=False), transforms.CenterCrop((64, 64))])
    image = transform(image)

    return image.unsqueeze(0)


if __name__ == '__main__':
    img = cv2.imread('data/A.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_img(img)
    print(img.shape)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(img[0].permute(1, 2, 0))
    plt.show()