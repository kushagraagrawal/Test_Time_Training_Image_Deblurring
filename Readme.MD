# Test Time Training Image Deblurring
Project for ECE 285

## Files
- cocoLoader.py -> Dataset class for MS COCO
- PASCALLoader.py -> Dataset class for PASCAL VOC 2012
- Models.py -> model classes
- train.py -> Training script

## Links to download Datasets
- PASCAL VOC 2012 - http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- MS COCO 2014 - https://archive.org/details/MSCoco2014

## Useful References
- Coco API - https://github.com/cocodataset/cocoapi

## Notes
05/31/2022
- Switched from L2 to L1 Loss for the deblur task, supposedly more robust and should deal with overfitting issues. Also taking weighted sum of the losses instead of simple addition

05/17/2022
- In CocoLoader, tried picking training images with just 1 annotation (commented out code in `cocoLoader.py`), resulted in only 3745 images.
