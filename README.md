# D-MIL.pyorch
**This is the official implementation of D-MIL in pytorch**

This implementation is based on ruotianluo's [https://github.com/ruotianluo/pytorch-faster-rcnn](pytorch-faster-rcnn).

**Using vgg16 as backbone, the trained model has detection mAP 53.7 on PASCAL VOC 2007 and 49.8 on PASCAL VOC 2012**

# Performances
  1). On PASCAL VOC 2007 dataset
  model    | #GPUs | batch size | lr | lr_decay | max iteration size | mAP | CorLoc
---------|--------|-----|--------|-----|-----|-------|--------|-----
VGG-16     | 1 | 1 | 1e-3 | 30000 | 40000 |  -  |  53.7  |  71.2


  2). On PASCAL VOC 2012 dataset
  model    | #GPUs | batch size | lr        | lr_decay | max iteration size | mAP | CorLoc
---------|--------|-----|--------|-----|-----|--------|-----
VGG-16     | 1 | 1 | 1e-3 | 80000   | 110000   |  49.8  |  71.9

# Prerequisites
* Nvidia GPU Tesla V100
* Ubuntu 16.04 LTS
* python **3.6**
* pytorch version in **0.4** is required. 
* tensorflow, tensorboard and [tensorboardX](https://github.com/lanpa/tensorboardX) for visualizing training and validation curve.

# Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/vasgaowei/D-MIL.pyorch.git
  ```
2. Compile the modules(nms, roi_pooling, roi_ring_pooling and roi_align)
  ```
  cd fast-rcnn-retrain-07/lib
  bash make.sh
  ```
3. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```
# Setup the data

1. Download the training, validation, test data and the VOCdevkit
  ```
  cd fast-rcnn-retrain-07/
  mkdir data
  cd data/
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
  ```
  
  
2. Extract all of these tars into one directory named VOCdevkit
  ```
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_08-Jun-2007.tar
  ```
3. Create symlinks for PASCAL VOC dataset or just rename the VOCdevkit to VOCdevkit2007
  ```
  cd fast-rcnn-retrain-07/data
  ln -s VOCdevkit VOCdevkit2007
  ```
4. It should have this basic structure
  ```
  $VOCdevkit2007/                     # development kit
  $VOCdevkit2007/VOC2007/             # VOC utility code
  $VOCdevkit2007/VOCcode/             # image sets, annodations, etc
  ```
  And for PASCAL VOC 2010 and PASCAL VOC 2012, just following the similar steps.
  
# Download the pre-trained ImageNet models
  VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth) and put it in the data/pretrained_model and rename it vgg16_caffe.pth. The folder has the following form.
  ```
  $ data/imagenet_weights/vgg16.pth
  ```
# Download the Selective Search proposals for PASCAL VOC 2007
  Download it from: https://dl.dropboxusercontent.com/s/orrt7o6bp6ae0tc/selective_search_data.tgz
  and unzip it and the final folder has the following form
  ```
  $ data/selective_search_data/voc_2007_test.mat
  $ data/selective_search_data/voc_2007_trainval.mat
  $ data/selective_search_data/voc_2012_test.mat
  $ data/selective_search_data/voc_2012_trainval.mat
  ```
# Train your own model
  ```Shell
  ./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
  ```
# Test and evaluate for detection performance mAP
  ```Shell
  ./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
  ```
# Test for localization performance CorLoc
  ```Shell
  ./experiments/scripts/test_faster_rcnn_corloc.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_faster_rcnn.sh
  # Examples:
  ./experiments/scripts/test_faster_rcnn_corloc.sh 0 pascal_voc vgg16
  ```
