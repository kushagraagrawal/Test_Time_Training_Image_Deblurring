import enum
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd

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

    def __len__(self) -> int:
        return self.inputDf.shape[0]

    def __getitem__(self, index: int):
        img_name = self.inputDf.iloc[index, 0]
        inputImg = Image.open(img_name)
        blurredImg = self.getBlurredOutput(inputImg)
        outputClass = self.inputDf.iloc[index, 1]
        sample = {'inputImg':blurredImg, 'image': inputImg, 'class': outputClass}

        if(self.transforms):
            sample['image'] = self.transforms(sample['image'])
            sample['inputImg'] = self.transforms(sample['inputImg'])
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