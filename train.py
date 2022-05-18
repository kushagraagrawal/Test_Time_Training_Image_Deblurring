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

# ============= argparser =============
parser = argparse.ArgumentParser(description='training Params')
parser.add_argument('--imageRoot', default='./train2014', help='location of the training images', type=str)
parser.add_argument('--trainLabelRoot', default='./annotations/instances_train2014.json', help='location of the train labels', type=str)
parser.add_argument('--valImageRoot', default='./val2014', help='location of the training images', type=str)
parser.add_argument('--valLabelRoot', default='./annotations/instances_val2014.json', help='location of the val labels', type=str)
parser.add_argument('--pascalCSV', default='pascalvoc.csv', help='location of pascal voc annotations', type=str)
parser.add_argument('--initLR', default=1e-4, help='initial Learning Rate', type=float)
parser.add_argument('--batchSize', default=16, help='batch size', type=float)
parser.add_argument('--nEpoch', default=10, help='epochs', type=float)

args = parser.parse_args()

# ============= torch cuda =============
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ============= DataLoader =============
datasetTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

pascalLoader = getPascalLoader(args.pascalCSV, args.batchSize, shuffle=True, inputTransform=datasetTransform)
cocoDataset = CocoDataset(args.imageRoot, args.trainLabelRoot, transform=datasetTransform)
trainCocoDL = DataLoader(dataset=cocoDataset, batch_size=args.batchSize, shuffle=True)

cocoDataset = CocoDataset(args.valImageRoot, args.valLabelRoot, transform=datasetTransform)
valCocoDL = DataLoader(dataset=cocoDataset, batch_size=args.batchSize, shuffle=True)

# ============= model =============
encoder = Unet_encoder(in_channels=3)
decoder = Unet_decoder()
classifier = Classifier()

# ============= optimizer, loss func =============
params = list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params, lr=args.initLR)
criterionDeblur = nn.MSELoss() # for now
criterionClassification = nn.CrossEntropyLoss()

# ============= training/val loop =============
for e in range(args.nEpoch):
    encoder.train()
    decoder.train()
    classifier.train()
    # train loop
    for step, (data) in enumerate(trainCocoDL):
        optimizer.zero_grad()
        gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)

        x, skip_connections = encoder(blurImg)
        output1 = decoder(x, skip_connections)
        loss = criterionDeblur(output1, gtImg)

        # ===== self-supervised task =====
        predictions = classifier(x)
        loss_classification = criterionClassification(predictions, classImg) # potential shape mismatch
        loss += loss_classification

        loss.backward()
        optimizer.step()

    # val loop
    encoder.eval()
    decoder.eval()
    classifier.eval()
    for step, (data) in enumerate(valCocoDL):
        gtImg, blurImg, classImg = data['image'].to(device), data['inputImg'].to(device), data['class'].to(device)

        x, skip_connections = encoder(blurImg)
        output1 = decoder(x, skip_connections)
        loss = criterionDeblur(output1, gtImg)

        # ===== self-supervised task =====
        predictions = classifier(x)
        loss_classification = criterionClassification(predictions, classImg) # potential shape mismatch
        loss += loss_classification
