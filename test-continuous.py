# ============= imports =============
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PASCALLoader import getPascalLoader
from cocoLoader import CocoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Model_architectures import Unet,SPP,Resnet,Unet2
import numpy as np
import os
import sewar
import cv2
import copy
from tqdm import tqdm

# ============= argparser =============
parser = argparse.ArgumentParser(description='testing Params')
parser.add_argument('--LR', default=1e-5, help='Learning Rate', type=float)
parser.add_argument('--nEpoch', default=10, help='epochs', type=int)
parser.add_argument('--experiment', default='experiment', help='result dir', type=str)
parser.add_argument('--BatchSize', default=1, help='test batch size', type=int)
parser.add_argument('--pascalCSV', default='pascalvoc_test.csv', help='location of pascal voc annotations', type=str)
parser.add_argument("--SPP", action="store_true")
parser.add_argument('--model', default='Unet',choices=['Unet', 'SPP', 'Resnet','Unet2'])



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

pascalLoader = getPascalLoader(args.pascalCSV,8, shuffle=True, inputTransform=datasetTransform)

# ============= model =============

if args.model == 'Unet':
    encoder = Unet.Encoder(in_channels=3).to(device)
    decoder = Unet.Decoder().to(device)
    classifier = Unet.Classifier().to(device)
    #loadPretrainedWeight(encoder)
    
elif args.model == 'SPP':
    encoder = SPP.Encoder().to(device)
    decoder = SPP.Decoder().to(device)
    classifier = SPP.Classifier().to(device)
    
elif args.model == 'Resnet':
    encoder = Resnet.Encoder().to(device)
    decoder = Resnet.Decoder().to(device)
    classifier = Resnet.Classifier().to(device)

elif args.model == 'Unet2':
    encoder = Unet2.Encoder(in_channels=3).to(device)
    decoder = Unet2.Decoder().to(device)
    classifier = Unet2.Classifier().to(device)
    
else:
    print("Invlaid Model choice")

encoder.load_state_dict(torch.load(args.experiment + '/encoder.pth'))
decoder.load_state_dict(torch.load(args.experiment + '/decoder.pth'))
classifier.load_state_dict(torch.load(args.experiment + '/classifier.pth'))

# ============= loss func =============
criterionDeblur = nn.MSELoss() # for now
criterionClassification = nn.CrossEntropyLoss()

# ============= training/val loop =============
trainIter = 0
valIter = 0

#Before training scores
psnr_before = []
uqi_before = []
ssim_before = []

#After training scores
psnr_after = []
uqi_after = []
ssim_after = []

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

encoder_new = copy.deepcopy(encoder).to(device)
decoder_new = decoder #Since decoder does not train
classifier_new = copy.deepcopy(classifier).to(device)
        
encoder_new.train()
decoder_new.eval()
classifier_new.train()


params = list(encoder_new.parameters()) + list(classifier_new.parameters())
optimizer = optim.Adam(params, lr=args.LR)

print("Starting Training")
#Training on PASCAL
for e in tqdm(range(args.nEpoch)):
    for step, (data) in enumerate(pascalLoader):
        
        optimizer.zero_grad()
        gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)
        x, skip_connections = encoder_new(blurImg)
        output1 = decoder_new(x, skip_connections)

        # ===== self-supervised task =====
        predictions = classifier_new(x)
        loss_classification = criterionClassification(predictions, classImg)

        loss_classification.backward()
        optimizer.step()
print("Finished Training")     

#Evaluate on PASCAL
encoder_new.eval()
decoder_new.eval()
classifier_new.eval()
pascalLoader = getPascalLoader(args.pascalCSV,1, shuffle=True, inputTransform=datasetTransform)

print("Starting Evaluation")
it = 0
for step, (data) in tqdm(enumerate(pascalLoader)):
    
        gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)
        it += 1
        
        #Evaluate before test time training
        with torch.no_grad():
            x, skip_connections = encoder(blurImg)
            pred_img = decoder(x, skip_connections)
            psnr_score,ssim_score,uqi_score = get_metrics(gtImg[0],pred_img[0])
            psnr_before.append(psnr_score)
            ssim_before.append(ssim_score)
            uqi_before.append(uqi_score)

        #Computing metrics after test time training
        with torch.no_grad():
            x, skip_connections = encoder_new(blurImg)
            pred_img = decoder_new(x, skip_connections)
            psnr_score,ssim_score,uqi_score = get_metrics(gtImg[0],pred_img[0])
            psnr_after.append(psnr_score)
            ssim_after.append(ssim_score)
            uqi_after.append(uqi_score)

        if it%100 == 0:
            print('instances: %d \n Average Before Test Time Training: SSIM %f PSNR %f UQI %f \n Average After Test Time SSIM %f PSNR %f UQI %f' %(it,np.mean(ssim_before),np.mean(psnr_before),np.mean(uqi_before),np.mean(ssim_after),np.mean(psnr_after),np.mean(uqi_after)))

print('Total instances: %d \n Average Before Test Time Training: SSIM %f PSNR %f UQI %f \n Average After Test Time SSIM %f PSNR %f UQI %f' %(it,np.mean(ssim_before),np.mean(psnr_before),np.mean(uqi_before),np.mean(ssim_after),np.mean(psnr_after),np.mean(uqi_after)))

print('Saving model')
PATH = args.experiment + '/continuous_ttt_' + str(args.nEpoch) + '.ckpt'
torch.save({
               'encoder': encoder_new.state_dict(),
               'decoder': decoder_new.state_dict(),
               'classifier': classifier_new.state_dict(),
               }, PATH)
        