from collections import defaultdict
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from pycocotools.coco import COCO

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
                 'truck': 8,
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

class CocoDataset(Dataset):
    def getBlurredOutput(self, img):
        return img

    def __init__(self, rootDir: str, annFile: str, transform = None) -> None:
        super(CocoDataset, self).__init__()
        self.coco = COCO(annFile)
        # self.ids = list(sorted(self.coco.imgs.keys()))
        self.ids = list()
        for _, val in category_dict.items():
            img_ids = self.coco.getImgIds(catIds=[val])
            self.ids.extend(img_ids)
        self.ids = list(sorted(self.ids))
        self.root = rootDir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        print(path)
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        target = self.coco.loadAnns(self.coco.getAnnIds(id))
        target_classes = defaultdict()
        # not sure on this
        # print(target)
        for i in range(len(target)):
            if(target[i]['category_id'] in target_classes.keys()):
                target_classes[target[i]['category_id']] += 1
            else:
                target_classes[target[i]['category_id']] = 1
        
        max_val = -np.inf
        max_category = 0
        for key, val in target_classes.items():
            if val > max_val:
                max_category = key
        return max_category

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        blurredImage = self.getBlurredOutput(image)
        target = self._load_target(id)

        if(self.transform is not None):
            image = self.transform(image)
            blurredImage = self.transform(blurredImage)

        return {'inputImg': image, 'blurredImage': blurredImage, 'class': target}

if(__name__ == '__main__'):
    cocoData = CocoDataset('../../train2014', '../../annotations/instances_train2014.json', transform=transform)
    dl = DataLoader(cocoData, batch_size=2, shuffle=True)
    for step, (data) in enumerate(dl):
        a = step