#!/bin/sh

# Copy the following script command.
start = 'date + %s'

echo "Download COCO, PASCAL VOC 2012 dataset"

echo "Prepare to download train-val2014 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm -f annotations_trainval2014.zip

echo "Prepare to download train2014 image zip file..."
wget -c http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm -f train2014.zip

echo "Prepare to download test2014 image zip file..."
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm -f val2014.zip

echo "Download PASCAL VOC 2012 dataset"
wget https://data.deepai.org/PascalVOC2012.zip
unzip PascalVOC2012.zip
rm -f PascalVOC2012.zip

end = 'date + %s'
runtime = $((end - start))

echo "Download completed in " $runtime  " second"
