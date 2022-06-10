# Test Time Training Image Deblurring
The project uses Test-Time Training for Image Deblurring.

## Dataset download
Simply run `bash datasetDownload.sh`

## Run Demo
- Download pretrained model from here - [link](https://drive.google.com/drive/folders/15cPC01SkFCAD3yAdjPY6v2JCl24Wo4o5?usp=sharing)
- Place it in a folder `Unet_all_blur`
- Install all dependencies using Conda and open jupyterlab
    - `conda create --name TTT python=3.8 -y`
    - `conda activate TTT`
    - `python -m pip install -r requirements.txt`
    - `jupyter lab`
- Open `demo.ipynb` and run all the cells
- Deactivate and remove the environment
    - `conda deactivate`
    - `conda env remove --name TTT`

## Files
- cocoLoader.py -> Dataset class for MS COCO
- PASCALLoader.py -> Dataset class for PASCAL VOC 2012
- Model_architectures -> Folders containing model implementations for ResNet, UNet and SPP
- train.py -> Training script
- test.py -> Testing script
- test-continuous.py -> Testing script for online Test-Time Training

## Command to train the model
- Download the dataset using the shell script above
- run `python train.py` with the following arguments
```
usage: train.py [-h] [--imageRoot IMAGEROOT] [--trainLabelRoot TRAINLABELROOT] [--valImageRoot VALIMAGEROOT]
                [--valLabelRoot VALLABELROOT] [--pascalCSV PASCALCSV] [--initLR INITLR] [--trainBatchSize TRAINBATCHSIZE]
                [--valBatchSize VALBATCHSIZE] [--nEpoch NEPOCH] [--experiment EXPERIMENT] [--checkpoint CHECKPOINT]
                [--lr_milestones LR_MILESTONES] [--gamma GAMMA] [--weight_decay WEIGHT_DECAY]
                [--loss_weightage LOSS_WEIGHTAGE] [--SPP] [--SGD] [--SSIM] [--model {Unet,SPP,Resnet,Unet2}]

training Params

optional arguments:
  -h, --help            show this help message and exit
  --imageRoot IMAGEROOT
                        location of the training images
  --trainLabelRoot TRAINLABELROOT
                        location of the train labels
  --valImageRoot VALIMAGEROOT
                        location of the training images
  --valLabelRoot VALLABELROOT
                        location of the val labels
  --pascalCSV PASCALCSV
                        location of pascal voc annotations
  --initLR INITLR       initial Learning Rate
  --trainBatchSize TRAINBATCHSIZE
                        train batch size
  --valBatchSize VALBATCHSIZE
                        val batch size
  --nEpoch NEPOCH       epochs
  --experiment EXPERIMENT
                        result dir
  --checkpoint CHECKPOINT
                        restore training from checkpoint
  --lr_milestones LR_MILESTONES
                        LR milestones
  --gamma GAMMA         gamma value
  --weight_decay WEIGHT_DECAY
                        regularization weight decay
  --loss_weightage LOSS_WEIGHTAGE
                        weightage to deblur loss
  --SPP
  --SGD
  --SSIM
  --model {Unet,SPP,Resnet,Unet2}
```

## Command to test the model
- Run `python test.py`
- Use the following arguments
```
usage: test.py [-h] [--LR LR] [--nEpoch NEPOCH] [--experiment EXPERIMENT]
               [--BatchSize BATCHSIZE] [--pascalCSV PASCALCSV] [--SPP]
               [--model {Unet,SPP,Resnet,Unet2}]

testing Params

optional arguments:
  -h, --help            show this help message and exit
  --LR LR               Learning Rate
  --nEpoch NEPOCH       epochs
  --experiment EXPERIMENT
                        result dir
  --BatchSize BATCHSIZE
                        test batch size
  --pascalCSV PASCALCSV
                        location of pascal voc annotations
  --SPP
  --model {Unet,SPP,Resnet,Unet2}
```

## Command to test the model (Online TTT)
- Run `python test-continuous.py`
- Use the following arguments
```
usage: test-continuous.py [-h] [--LR LR] [--nEpoch NEPOCH]
                          [--experiment EXPERIMENT] [--BatchSize BATCHSIZE]
                          [--pascalCSV PASCALCSV] [--SPP]
                          [--model {Unet,SPP,Resnet,Unet2}]

testing Params

optional arguments:
  -h, --help            show this help message and exit
  --LR LR               Learning Rate
  --nEpoch NEPOCH       epochs
  --experiment EXPERIMENT
                        result dir
  --BatchSize BATCHSIZE
                        test batch size
  --pascalCSV PASCALCSV
                        location of pascal voc annotations
  --SPP
  --model {Unet,SPP,Resnet,Unet2}
```
