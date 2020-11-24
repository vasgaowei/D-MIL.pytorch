# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.dmil.dmil import DMIL
import pdb

class vgg16(DMIL):
  def __init__(self, classes, pretrained=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained

    DMIL.__init__(self, classes)

  def _init_modules(self):
    vgg = models.vgg16()
    vgg.features[24] = nn.Conv2d(in_channels=512, out_channels=512, 
                                 kernel_size=3, dilation=2 ,stride=1,
                                 padding=2)
    vgg.features[26] = nn.Conv2d(in_channels=512, out_channels=512, 
                                 kernel_size=3, dilation=2 ,stride=1,
                                 padding=2)
    vgg.features[28] = nn.Conv2d(in_channels=512, out_channels=512, 
                                 kernel_size=3, dilation=2 ,stride=1,
                                 padding=2)
    del vgg.features[23]
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls0_score0 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls0_score1 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls1_score0 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls1_score1 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls2_score0 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls2_score1 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls3_score0 = nn.Linear(4096, self.n_classes)
    self.RCNN_cls3_score1 = nn.Linear(4096, self.n_classes)
    self.RCNN_pcl_score0 = nn.Linear(4096, self.n_classes + 1)
    self.RCNN_pcl_score1 = nn.Linear(4096, self.n_classes + 1)
    self.RCNN_pcl_score2 = nn.Linear(4096, self.n_classes + 1)

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

