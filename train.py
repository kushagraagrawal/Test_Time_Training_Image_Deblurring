import argparse
import torch.nn as nn
import torch.optim as optim
from PASCALLoader import getPascalLoader
from cocoLoader import CocoDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Models import Unet_encoder, Unet_decoder, Classifier

parser = argparse.ArgumentParser(description='training Params')
parser.add_argument('--imageRoot', default='./train2014', help='location of the training images', type=str)
parser.add_argument('--trainLabelRoot', default='./annotations/instances_train2014.json', help='location of the labels', type=str)
parser.add_argument('--pascalCSV', default='pascalvoc.csv', help='location of pascal voc annotations', type=str)
parser.add_argument('--initLR', default=1e-4, help='initial Learning Rate', type=float)
parser.add_argument('--batchSize', default=16, help='batch size', type=float)
parser.add_argument('--nEpoch', default=10, help='epochs', type=float)

args = parser.parse_args()

device = 'cuda'

# ============= DataLoader =============
datasetTransform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

pascalLoader = getPascalLoader(args.pascalCSV, args.batchSize, shuffle=True, inputTransform=datasetTransform)
cocoDataset = CocoDataset(args.imageRoot, args.trainLabelRoot, transform=datasetTransform)
cocoDL = DataLoader(dataset=cocoDataset, batch_size=args.batchSize, shuffle=True)

# ============= model =============
encoder = Unet_encoder(in_channels=3)
decoder = Unet_decoder()
classifier = Classifier()

# ============= optimizer, loss func =============
params = list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(params, lr=args.initLR)
criterion = nn.MSELoss() # for now

