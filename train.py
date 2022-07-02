# ============= imports =============
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from PASCALLoader import getPascalLoader
from cocoLoader import CocoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Model_architectures import Unet,SPP,Resnet,Unet2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torchvision.models.resnet as resnet
from torchvision.utils import save_image
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

# ============= argparser =============
parser = argparse.ArgumentParser(description='training Params')
parser.add_argument('--imageRoot', default='./train2014', help='location of the training images', type=str)
parser.add_argument('--trainLabelRoot', default='./annotations/instances_train2014.json', help='location of the train labels', type=str)
parser.add_argument('--valImageRoot', default='./val2014', help='location of the training images', type=str)
parser.add_argument('--valLabelRoot', default='./annotations/instances_val2014.json', help='location of the val labels', type=str)
parser.add_argument('--pascalCSV', default='pascalvoc.csv', help='location of pascal voc annotations', type=str)
parser.add_argument('--initLR', default=1e-4, help='initial Learning Rate', type=float)
parser.add_argument('--trainBatchSize', default=16, help='train batch size', type=int)
parser.add_argument('--valBatchSize', default=8, help='val batch size', type=int)
parser.add_argument('--nEpoch', default=10, help='epochs', type=int)
parser.add_argument('--experiment', default='train_result', help='result dir', type=str)
parser.add_argument('--checkpoint', default=None, help='restore training from checkpoint', type=str)
parser.add_argument('--lr_milestones', help='LR milestones', default=[5,10,15,20,25,30])
parser.add_argument('--gamma', default=0.2, help='gamma value', type=float)
parser.add_argument('--weight_decay', help='regularization weight decay', default=1e-4, type=float)
parser.add_argument('--loss_weightage',help='weightage to deblur loss', default=0.5, type=float)
parser.add_argument("--SPP", action="store_true")
parser.add_argument("--SGD", action="store_true")
parser.add_argument("--SSIM", action="store_true")
parser.add_argument('--model', default='Unet',choices=['Unet', 'SPP', 'Resnet','Unet2'])


args = parser.parse_args()

os.system('mkdir %s'%(args.experiment))

# ============= tensorboard visualisation =============
writer = SummaryWriter()

# ============= torch cuda =============
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'

# ============= DataLoader =============
datasetTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

trainCocoDataset = CocoDataset(args.imageRoot, args.trainLabelRoot, transform=datasetTransform)
trainCocoDL = DataLoader(dataset=trainCocoDataset, batch_size=args.trainBatchSize, shuffle=True)

valCocoDataset = CocoDataset(args.valImageRoot, args.valLabelRoot, transform=datasetTransform)
valCocoDL = DataLoader(dataset=valCocoDataset, batch_size=args.valBatchSize, shuffle=True)

# ============= model =============

if args.model == 'Unet':
    encoder = Unet.Encoder(in_channels=3).to(device)
    decoder = Unet.Decoder().to(device)
    classifier = Unet.Classifier().to(device)
    # loadPretrainedWeight(encoder)
    
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

# ============= optimizer, loss func =============
params = list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters())

optimizer = optim.Adam(params,lr=args.initLR, weight_decay=args.weight_decay)

criterionDeblur = nn.L1Loss()
criterionClassification = nn.CrossEntropyLoss()
criterion_ssim = SSIM()

# ============= loss arrays =============
trainIter = 0
valIter = 0
best_train_loss = np.inf
best_val_loss = np.inf
train_loss_epoch = []
val_loss_epoch = []
train_accuracy_epoch = []
val_accuracy_epoch = []
e = 0

# ============= load checkpoint =============
if(args.checkpoint is not None):
    checkpoint = torch.load(args.checkpoint)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    e = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['loss']
    
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 256, 256)
    save_image(img, name)

# ============= train/val loop =============
while e < args.nEpoch:
    encoder.train()
    decoder.train()
    classifier.train()
    train_loss = 0
    val_loss = 0
    
    correct = 0
    total = 0
    
    # train loop
    for step, (data) in enumerate(trainCocoDL):
        trainIter += 1
        optimizer.zero_grad()
        gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)

        x, skip_connections = encoder(blurImg)
        if args.SPP:
            output1 = decoder(blurImg, skip_connections)
        else:
            output1 = decoder(x, skip_connections)
        loss = criterionDeblur(output1, gtImg)

        # ===== self-supervised task =====
        predictions = classifier(x)
        loss_classification = criterionClassification(predictions, classImg)
        
        loss_final = (args.loss_weightage * loss) + ((1 - args.loss_weightage) * loss_classification)
        
        writer.add_scalar("Loss_classification/train_iteration", loss_classification, trainIter)
        writer.add_scalar("Loss_deblur/train_iteration", loss, trainIter)
        
        if args.SSIM:
            loss_ssim = 1-criterion_ssim(output1, gtImg)
            loss_final += loss_ssim
            writer.add_scalar("Loss_SSIM/train_iteration", loss_ssim, trainIter)
        
        # ===== accuracy calculation =====
        _, predict = predictions.max(1)
        total += classImg.size(0)
        correct += (predict==classImg).float().sum().item()
        accuracy = correct / total
        
        # ===== visualise the predictions =====
        if(step == int((len(trainCocoDataset)/trainCocoDL.batch_size)-1)):
            os.system('mkdir %s/%s'%(args.experiment, 'epoch_' + str(e)))
            save_decoded_image(gtImg.cpu().data, name=f'{args.experiment}/epoch_{str(e)}/trainGroundTruth_{trainIter}_{e}.png')
            save_decoded_image(blurImg.cpu().data, name=f'{args.experiment}/epoch_{str(e)}/trainInputImage_{trainIter}_{e}.png')
            save_decoded_image(output1.cpu().data, name=f'{args.experiment}/epoch_{str(e)}/trainVisualization_{trainIter}_{e}.png')

        train_loss += loss_final.item()

        loss_final.backward()
        optimizer.step()

        print('Train - Epoch: %d, Iteration: %d, deblur loss: %0.3f, Classification loss: %0.3f, Classification accuracy: %0.2f'%(e, trainIter, loss, loss_classification, accuracy))
        if args.SSIM:
            print('SSIM loss %0.4f'%(loss_ssim))
            criterion_ssim.reset()
    
    train_loss_epoch.append(train_loss/total)
    writer.add_scalar("Loss_epoch/train", train_loss/total, e+1)
    writer.add_scalar("Accuracy_epoch/train", accuracy, e+1)
    train_accuracy_epoch.append(accuracy)

    # val loop
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        classifier.eval()

        correct = 0
        total = 0
        for step, (data) in enumerate(valCocoDL):
            valIter += 1
            gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)

            x, skip_connections = encoder(blurImg)
            if args.SPP:
                output1 = decoder(blurImg, skip_connections)
            else:
                output1 = decoder(x, skip_connections)
            loss = criterionDeblur(output1, gtImg)

            # ===== self-supervised task =====
            predictions = classifier(x)
            loss_classification = criterionClassification(predictions, classImg)
            
            loss_final = (args.loss_weightage * loss) + ((1 - args.loss_weightage) * loss_classification)
            
            writer.add_scalar("Loss_classification/val_iteration", loss_classification, valIter)
            writer.add_scalar("Loss_deblur/val_iteration", loss, valIter)
            
            if args.SSIM:
                loss_ssim = 1-criterion_ssim(output1, gtImg)
                loss_final += loss_ssim
                writer.add_scalar("Loss_SSIM/val_iteration", loss_ssim, valIter)
            
            # ===== accuracy calculation =====
            _, predict = predictions.max(1)
            total += classImg.size(0)
            correct += (predict==classImg).float().sum().item()
            accuracy = correct / total
            
            #  ===== visualise the predictions =====
            if(step == int((len(valCocoDataset)/valCocoDL.batch_size)-1)):
                save_decoded_image(gtImg.cpu().data, name=f'{args.experiment}/epoch_{str(e)}/valGroundTruth_{trainIter}_{e}.png')
                save_decoded_image(blurImg.cpu().data, name=f'{args.experiment}/epoch_{str(e)}/valInputImage_{trainIter}_{e}.png')
                save_decoded_image(output1.cpu().data, name=f'{args.experiment}/epoch_{str(e)}/valVisualization_{trainIter}_{e}.png')

            val_loss += loss_final.item()

            print('Val - Epoch: %d, Iteration: %d, deblur loss: %0.3f, Classification loss: %0.3f, Classification accuracy: %0.2f'%(e, valIter, loss, loss_classification, accuracy))
        
            if args.SSIM:
                print('SSIM loss %0.4f'%(loss_ssim))
                criterion_ssim.reset()
        
        val_loss_epoch.append(val_loss/total)
        val_accuracy_epoch.append(accuracy)
        writer.add_scalar("Loss_epoch/val", val_loss/total, e+1)
        writer.add_scalar("Accuracy_epoch/val", accuracy, e+1)
        
        if(val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), args.experiment + '/encoder.pth')
            torch.save(decoder.state_dict(), args.experiment + '/decoder.pth')
            torch.save(classifier.state_dict(), args.experiment + '/classifier.pth')
            
            print('************ saving checkpoint at epoch: %d ************'%(e))
            best_train_loss = train_loss
            PATH = args.experiment + '/checkpoint_' + str(e) + '.ckpt'
            torch.save({
                       'encoder': encoder.state_dict(),
                       'decoder': decoder.state_dict(),
                       'classifier': classifier.state_dict(),
                       'optimizer':optimizer.state_dict(),
                       'epoch': e,
                       'loss': best_val_loss
                       }, PATH)
        
    e += 1
    scheduler.step(val_loss/total)

    np.save(args.experiment + '/train_loss.npy', np.array(train_loss_epoch))
    np.save(args.experiment + '/val_loss.npy', np.array(val_loss_epoch))