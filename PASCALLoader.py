from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
import torch
import cv2
import numpy as np

# https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

class PascalVOCLoader(Dataset):
    def getBlurredOutput(self, img):
        return img
        
    def __init__(self, inputDf: pd.DataFrame, transform=None) -> None:
        super(PascalVOCLoader, self).__init__()
        self.inputDf = inputDf
        self.transforms = transform
        self.blur_transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize((256, 256))])
        
    def motion_blur_horizontal(self,img_path,kernel_size):
        img = cv2.imread(img_path)

        kernel_h = np.zeros((kernel_size, kernel_size))
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        kernel_h /= kernel_size

        horizontal_mb = cv2.filter2D(img, -1, kernel_h)
        horizontal_mb = cv2.cvtColor(horizontal_mb, cv2.COLOR_RGB2BGR)
        return horizontal_mb
    
    def motion_blur_vertical(self,img_path,kernel_size):
        img = cv2.imread(img_path)

        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_v /= kernel_size

        vertical_mb = cv2.filter2D(img, -1, kernel_v)
        vertical_mb = cv2.cvtColor(vertical_mb, cv2.COLOR_RGB2BGR)
        return vertical_mb
    
    def gaussian_blur(self,img_path,k_size):
        img = cv2.imread(img_path)
        img = cv2.GaussianBlur(img,(k_size,k_size),0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def avg_blur(self,img_path,k_size):
        img = cv2.imread(img_path)
        img = cv2.blur(img, (k_size,k_size))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
        
    def getBlurredOutput(self,img_path):
        #img_path = os.path.join(self.root, img_path)
        p = torch.rand(1)
        k_size = 15
        
        if p<=0.25:
            #print("Vertical")
            return self.motion_blur_vertical(img_path,k_size)
        if p<=0.5:
            #print("Horizontal")
            return self.motion_blur_horizontal(img_path,k_size)
        if p<=0.75:
            #print("Average")
            return self.avg_blur(img_path,k_size)
        else:
            #print("Gaussian")
            return self.gaussian_blur(img_path,k_size)

    def __len__(self) -> int:
        return self.inputDf.shape[0]

    def __getitem__(self, index: int):
        img_name = self.inputDf.iloc[index, 0]
        img = Image.open(img_name)
        blurredImg = self.getBlurredOutput(img_name)
        outputClass = self.inputDf.iloc[index, 1]
        sample = {'inputImg':blurredImg, 'image': img, 'class': outputClass}

        if(self.transforms):
            sample['image'] = self.transforms(sample['image'])
            sample['inputImg'] = self.blur_transforms(sample['inputImg'])
        return sample

def getPascalLoader(inputFileName: str, batch_size: int, shuffle: bool, inputTransform) -> DataLoader:
    df = pd.read_csv(inputFileName)
    dataset = PascalVOCLoader(df, transform=inputTransform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if(__name__=='__main__'):
    dl = getPascalLoader('pascalvoc.csv', 16, True, transform)
    data = next(iter(dl))
    print(f"Feature batch shape: {data['image'].size()}")
    print(f"Labels batch shape: {data['class'].size()}")