# D-MIL.pyorch for COCO
**This is the official implementation of D-MIL in pytorch**
This codebase is for tranining and testing on COCO 2014 dataset.
This implementation is based on jwyang's [pytorch-faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) and ppengtang's [pcl.pytorch](https://github.com/ppengtang/pcl.pytorch).

**Using vgg16 as backbone, the trained model has detection mAP@0.5 24.7 and mAP@[.5,.95] 11.3**

The model is trained on COCO 2014 train and tested on COCO 2014 val. 

# Performances
  1). On COCO 2014dataset
  model    | #GPUs | batch size | lr        | lr_decay | max_epoch     |  time/epoch | mAP@0.5 | mAP@[.5,.95]
---------|--------|-----|--------|-----|-----|-------|--------|-----
VGG-16     | 4 | 4 | 1e-3 | 3   | 5   |  53 hr |  24.7  |  11.3


  2). Detection performance with different numbers of MIL Learners
  #num    |    mAP@0.5    |   mAP@[.5,.95]
  --------|---------------|-----
  2       |    23.4       |  10.7
  3       |    23.6       |  10.8
  4       |    24.7       |  11.3
  5       |    23.8       |  10.8
  

# Prerequisites
* Nvidia GPU RTX 2080Ti
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
  cd D-MIL-COCO.pytorch/lib
  bash make_cuda.sh
  ```
# Setup the data

1. Download the training, validation, test data for COCO 2014
  Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.
2. It should have this basic structure
  ```
  $data/coco                     # coco
  $data/coco/images             # coco images 
  $data/coco/images/train2014   # training
  $data/coco/images/val2014     # testing
  $data/coco/annotations             # annotations
  $data/coco/PythonAPI               # coco pycocotools
  ```
  
# Download the pre-trained ImageNet models
  VGG16: [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth) and put it in the data/pretrained_model and rename it vgg16_caffe.pth. The folder has the following form.
  ```
  $ data/pretrained_model/vgg16_caffe.pth
  ```
# Download the precomputed proposals for COCO 2014
  Download it from [https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/](MCG)
  and unzip it and the final folder has the following form
  ```
  $ data/mcg/MCG-COCO-train2014-boxes
  $ data/mcg/MCG-COCO-val2014-boxes
  ```
# Train your own model
  For vgg16 backbone, we can train and evaluate the model using the following commands
  ```
  bash train.sh $prefix
  ```
  And for evaluation on detection mAP, we can using the following commands
  ```
  bash test.sh $prefix
  ```
  We can modify ```train.sh``` and ```test.sh``` for specifying the GPU IDs.
  When using ```batch_size``` 8, we can using 8 GPUs with each image on each GPU. Meanwhile, the learning rate should be doubled as ```2e-3```.
