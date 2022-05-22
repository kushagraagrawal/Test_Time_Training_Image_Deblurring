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
import random
import matplotlib.pyplot as plt

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

args = parser.parse_args()

os.system('mkdir %s'%(args.experiment))

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

# pascalLoader = getPascalLoader(args.pascalCSV, args.trainBatchSize, shuffle=True, inputTransform=datasetTransform)
cocoDataset = CocoDataset(args.imageRoot, args.trainLabelRoot, transform=datasetTransform)
trainCocoDL = DataLoader(dataset=cocoDataset, batch_size=args.trainBatchSize, shuffle=True)

cocoDataset = CocoDataset(args.valImageRoot, args.valLabelRoot, transform=datasetTransform)
valCocoDL = DataLoader(dataset=cocoDataset, batch_size=args.valBatchSize, shuffle=True)

# ============= model =============
encoder = Unet_encoder(in_channels=3).to(device)
decoder = Unet_decoder().to(device)
classifier = Classifier().to(device)

# ============= optimizer, loss func =============
params = list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params, lr=args.initLR)
criterionDeblur = nn.MSELoss() # for now
criterionClassification = nn.CrossEntropyLoss()

# ============= training/val loop =============
trainIter = 0
valIter = 0
best_train_loss = np.inf
best_val_loss = np.inf
train_loss_epoch = []
val_loss_epoch = []
train_accuracy_epoch = []
val_accuracy_epoch = []
e = 0

if(args.checkpoint is not None):
    checkpoint = torch.load(args.checkpoint)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    e = checkpoint['epoch']
    best_train_loss = checkpoint['loss']

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
        output1 = decoder(x, skip_connections)
        loss = criterionDeblur(output1, gtImg)

        # ===== self-supervised task =====
        predictions = classifier(x)
        loss_classification = criterionClassification(predictions, classImg) # potential shape mismatch
        loss_final = loss + loss_classification
        _, predict = predictions.max(1)
        total += classImg.size(0)
        correct += (predict==classImg).float().sum().item()
        accuracy = correct / total
        
        
        if(((step + 1) % 100) == 0):
            idx = random.randint(0, data['image'].shape[0] - 1)
            img = data['image'][idx].detach().cpu().squeeze()
            blurred_img = data['inputImg'][idx].detach().cpu().squeeze()
        
            prediction_output = output1[idx].detach().cpu().squeeze().numpy()
            prediction_output = (prediction_output - np.min(prediction_output)) / np.max(prediction_output)
            prediction_output = np.uint8(255 * prediction_output)
            fig,axs = plt.subplots(1,3)
            axs[0].imshow(img.permute(1,2,0))
            axs[1].imshow(blurred_img.permute(1,2,0))
            axs[2].imshow(prediction_output.transpose([1,2,0]))
            plt.savefig('%s/trainVisualization_%d_%d.png'%(args.experiment, step, e))
            fig.clf()
            plt.close()

        train_loss += loss_final

        loss_final.backward()
        optimizer.step()

        print('Train - Epoch: %d, Iteration: %d, deblur loss: %f, Classification loss: %f, Classification accuracy: %f'%(e, trainIter, loss, loss_classification, accuracy))
    
    train_loss_epoch.append(train_loss)
    train_accuracy_epoch.append(accuracy)
    if(train_loss < best_train_loss):
        print('************ saving checkpoint at epoch: %d ************'%(e))
        best_train_loss = train_loss
        PATH = args.experiment + '/checkpoint.ckpt'
        torch.save({
                   'encoder': encoder.state_dict(),
                   'decoder': decoder.state_dict(),
                   'classifier': classifier.state_dict(),
                   'epoch': e,
                   'loss': best_train_loss
                   }, PATH)

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
            output1 = decoder(x, skip_connections)
            loss = criterionDeblur(output1, gtImg)

            # ===== self-supervised task =====
            predictions = classifier(x)
            loss_classification = criterionClassification(predictions, classImg)
            loss_final = loss + loss_classification

            _, predict = predictions.max(1)
            total += classImg.size(0)
            correct += (predict==classImg).float().sum().item()
            accuracy = correct / total

            if(((step + 1) % 100) == 0):
                idx = random.randint(0, data['image'].shape[0] - 1)
                img = data['image'][idx].detach().cpu().squeeze()
                blurred_img = data['inputImg'][idx].detach().cpu().squeeze()
                prediction_output = output1[idx].detach().cpu().squeeze().numpy()
                prediction_output = (prediction_output - np.min(prediction_output)) / np.max(prediction_output)
                prediction_output = np.uint8(255 * prediction_output)
                fig,axs = plt.subplots(1,3)
                axs[0].imshow(img.permute(1,2,0))
                axs[1].imshow(blurred_img.permute(1,2,0))
                axs[2].imshow(prediction_output.transpose([1,2,0]))
                plt.savefig('%s/valVisualization_%d_%d.png'%(args.experiment, step, e))
                fig.clf()
                plt.close()

            val_loss += loss_final

            print('Val - Epoch: %d, Iteration: %d, deblur loss: %f, Classification loss: %f, Classification accuracy: %f'%(e, valIter, loss, loss_classification, accuracy))
        
        val_loss_epoch.append(val_loss)
        val_accuracy_epoch.append(accuracy)
        if(val_loss < best_val_loss):
            best_val_loss = loss
            torch.save(encoder.state_dict(), args.experiment + '/encoder.pth')
            torch.save(decoder.state_dict(), args.experiment + '/decoder.pth')
            torch.save(classifier.state_dict(), args.experiment + '/classifier.pth')
    
    e += 1
    train_loss_epoch.append(train_loss)
    val_loss_epoch.append(val_loss)

np.save(args.experiment + '/train_loss.npy', train_loss_epoch)
np.save(args.experiment + '/val_loss.npy', val_loss_epoch)