# D-MIL.pyorch
**This is the official implementation of D-MIL in pytorch**

This implementation is based on jwyang's [pytorch-faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) and ppengtang's [pcl.pytorch](https://github.com/ppengtang/pcl.pytorch).

**Using vgg16 as backbone, the trained model has detection mAP 53.5 on PASCAL VOC 2007 and 49.6 on PASCAL VOC 2012**

# Prerequisites
* Nvidia GPU Tesla V100
* Ubuntu 16.04 LTS
* python **3.6**
* pytorch version in **1.0 ~ 1.4** is required. 
* tensorflow, tensorboard and [tensorboardX](https://github.com/lanpa/tensorboardX) for visualizing training and validation curve.

# Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/vasgaowei/D-MIL.pyorch.git
  ```
2. Compile the modules(nms, roi_pooling, roi_ring_pooling and roi_align)
  ```
  cd D-MIL.pytorch/lib
  bash make_cuda.sh
  ```
# Setup the data

1. Download the training, validation, test data and the VOCdevkit
  ```
  cd D-MIL.pyorch/
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
  cd D-MIL.pyorch/data
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
  $ data/pretrained_model/vgg16.pth
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
  For vgg16 backbone, we can train and evaluate the model using the following commands
  ```
  bash both_2007.sh prefix GPU_ID
  ```
  And for evaluation on detection mAP, we can using the following commands
  ```
  bash test_test_2007.sh prefix
  ```
  And for evaluation on CorLoc, we can using the following commands
  ```
  bash test_corloc_2007.sh prefix
  ```
# Retrain using Faster RCNN
  First, run the following commands to get the pseudo ground-truths
  ```
  bash .sh prefix
  ```
  
