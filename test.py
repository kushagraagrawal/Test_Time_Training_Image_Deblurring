# ============= imports =============
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PASCALLoader import getPascalLoader
from cocoLoader import CocoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Models import Unet_encoder, Unet_decoder, Classifier
import numpy as np
import os
import sewar
import cv2
import copy
from tqdm import tqdm

# ============= argparser =============
parser = argparse.ArgumentParser(description='testing Params')
parser.add_argument('--LR', default=1e-4, help='Learning Rate', type=float)
parser.add_argument('--nEpoch', default=10, help='epochs', type=int)
parser.add_argument('--experiment', default='experiment', help='result dir', type=str)
parser.add_argument('--BatchSize', default=1, help='test batch size', type=int)
parser.add_argument('--pascalCSV', default='pascalvoc.csv', help='location of pascal voc annotations', type=str)


args = parser.parse_args()


# ============= torch cuda =============
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'

# ============= DataLoader =============
datasetTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

pascalLoader = getPascalLoader(args.pascalCSV,1, shuffle=True, inputTransform=datasetTransform)

# ============= model =============
encoder = Unet_encoder(in_channels=3).to(device)
decoder = Unet_decoder().to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(torch.load(args.experiment + '/encoder.pth'))
decoder.load_state_dict(torch.load(args.experiment + '/decoder.pth'))
classifier.load_state_dict(torch.load(args.experiment + '/classifier.pth'))

# ============= loss func =============
criterionDeblur = nn.MSELoss() # for now
criterionClassification = nn.CrossEntropyLoss()

# ============= training/val loop =============
trainIter = 0
valIter = 0

psnr_before = []
psnr_after = []
ssim_before = []
ssim_after = []
uqi_before = []
uqi_after = []

it = 0
accuracy = 0

def normalize_img(img):
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

def get_metrics(orig_img,pred_img):
    
    #converting to numpy array
    orig_img = orig_img.permute(1,2,0).detach().cpu().numpy()
    pred_img = pred_img.permute(1,2,0).detach().cpu().numpy()
    
    #normalization
    orig_img = normalize_img(orig_img)
    pred_img = normalize_img(pred_img)
    
    #psnr
    psnr_score = sewar.psnr(pred_img,orig_img)
    #ssim
    ssim_score = sewar.ssim(pred_img,orig_img)[0]
    #uqi
    uqi_score = sewar.uqi(pred_img,orig_img)
    
    return psnr_score,ssim_score,uqi_score

for step, (data) in tqdm(enumerate(pascalLoader)):
    
        encoder_copy = copy.deepcopy(encoder).to(device)
        decoder_copy = copy.deepcopy(decoder).to(device)
        classifier_copy = copy.deepcopy(classifier).to(device)
        
        params = list(encoder_copy.parameters()) + list(classifier_copy.parameters())
        optimizer = optim.Adam(params, lr=args.LR)
    
        encoder_copy.train()
        decoder_copy.eval()
        classifier_copy.train()

        gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)
        #print("classImg",classImg)
        
        it += 1
        
        for e in range(args.nEpoch):
            encoder_copy.train()
            decoder_copy.eval()
            classifier_copy.train()
            
            optimizer.zero_grad()
            x, skip_connections = encoder_copy(blurImg)
            
            # ===== self-supervised task =====
            predictions = classifier_copy(x)
            loss_classification = criterionClassification(predictions, classImg)
            
            loss_classification.backward()
            optimizer.step()
            
            #Computing metrics before test time training
            if e==0:
                with torch.no_grad():
                    pred_img = decoder_copy(x, skip_connections)
                    psnr_score,ssim_score,uqi_score = get_metrics(gtImg[0],pred_img[0])
                    psnr_before.append(psnr_score)
                    ssim_before.append(ssim_score)
                    uqi_before.append(uqi_score)
                    
            #Computing metrics after test time training
            if e==args.nEpoch-1:
                with torch.no_grad():
                    pred_img = decoder_copy(x, skip_connections)
                    psnr_score,ssim_score,uqi_score = get_metrics(gtImg[0],pred_img[0])
                    psnr_after.append(psnr_score)
                    ssim_after.append(ssim_score)
                    uqi_after.append(uqi_score)
                    first_epoch = False
            
        if it%100 == 0:
            print('instances: %d \n Average Before Test Time Training: SSIM %f PSNR %f UQI %f \n Average After Test Time SSIM %f PSNR %f UQI %f' %(it,np.mean(ssim_before),np.mean(psnr_before),np.mean(uqi_before),np.mean(ssim_after),np.mean(psnr_after),np.mean(uqi_after)))

print('Total instances: %d \n Average Before Test Time Training: SSIM %f PSNR %f UQI %f \n Average After Test Time SSIM %f PSNR %f UQI %f' %(it,np.mean(ssim_before),np.mean(psnr_before),np.mean(uqi_before),np.mean(ssim_after),np.mean(psnr_after),np.mean(uqi_after)))
