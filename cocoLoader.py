from collections import defaultdict
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from pycocotools.coco import COCO
import cv2
import torch

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

category_dict = {'person': 1,
                 'bicycle': 2,
                 'car': 3,
                 'motorcycle': 4,
                 'airplane': 5,
                 'bus': 6,
                 'train': 7,
                 # 'truck': 8,
                 'boat': 9,
                 'bird': 16,
                 'bottle': 44,
                 'cat': 17,
                 'chair': 62,
                 'cow': 21,
                 'dining table': 67,
                 'dog': 18,
                 'horse': 19,
                 'potted plant': 64,
                 'sheep': 20,
                 'couch': 63,
                 'tv': 72}

temp_dict = {1: 0,
             2: 1,
             3: 2,
             4: 3,
             5: 4,
             6: 5,
             7: 6,
             # 8: 7,
             9: 7,
             16: 8,
             44: 9,
             17: 10,
             62: 11,
             21: 12,
             67: 13,
             18: 14,
             19: 15,
             64: 16,
             20: 17,
             63: 18,
             72: 19}

class CocoDataset(Dataset):
    
    def __init__(self, rootDir: str, annFile: str, transform = None) -> None:
        super(CocoDataset, self).__init__()
        self.coco = COCO(annFile)
        self.ids = list()
        for _, val in category_dict.items():
            img_ids = self.coco.getImgIds(catIds=[val])
            requiredID = list()
            for ids in img_ids:
                anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[ids], iscrowd=None))
                target_classes = defaultdict()
                for i in range(len(anns)):
                    if(anns[i]['category_id'] in category_dict.values()):
                        if(anns[i]['category_id'] in target_classes.keys()):
                            target_classes[anns[i]['category_id']] += 1
                        else:
                            target_classes[anns[i]['category_id']] = 1
                    else:
                        target_classes = defaultdict()
                        break
                if(len(target_classes.keys()) == 1):
                    requiredID.append(ids)
            self.ids.extend(requiredID)
            
        self.ids = list(set(sorted(self.ids)))
        print(len(self.ids))
        self.root = rootDir
        self.transform = transform
        self.blur_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256, 256))])
        
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
        img_path = os.path.join(self.root, img_path)
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
        return len(self.ids)

    def _load_image(self, ids: int) -> Image.Image:
        path = self.coco.loadImgs(ids)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"),path

    def _load_target(self, ids: int):
        target = self.coco.loadAnns(self.coco.getAnnIds(ids, iscrowd=None))
#         for i in range(len(target)):
#             print(target[i]['category_id'])
#         print('********')
        return temp_dict[target[0]['category_id']]
#         target_classes = defaultdict()
#         # not sure on this
#         for i in range(len(target)):
#             print(target[i]['category_id'], target[i]['iscrowd'])
#             if(target[i]['category_id'] in category_dict.values()):
#                 if(target[i]['category_id'] in target_classes.keys()):
#                     target_classes[target[i]['category_id']] += 1
#                 else:
#                     target_classes[target[i]['category_id']] = 1
#         print('**********')
#         max_val = -np.inf
#         max_category = 0
#         # print(target_classes)
#         for key, val in target_classes.items():
#             if val > max_val:
#                 max_category = key
#                 max_val = val
#         return max_category

    def __getitem__(self, index: int):
        ids = self.ids[index]
        image,path = self._load_image(ids)
        blurredImage = self.getBlurredOutput(path)
        target = self._load_target(ids)

        if(self.transform is not None):
            image = self.transform(image)
            blurredImage = self.blur_transform(blurredImage)

        return {'image': image, 'inputImg': blurredImage, 'class': target}

if(__name__ == '__main__'):
    cocoData = CocoDataset('train2014', 'annotations/instances_train2014.json', transform=transform)
    dl = DataLoader(cocoData, batch_size=2, shuffle=True)
    for step, (data) in enumerate(dl):
        a = step