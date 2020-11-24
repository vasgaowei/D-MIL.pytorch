#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:30:59 2019

@author: vasgaoweithu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
# from model.utils.cython_nms import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob
import nn as mynn
from xml.etree import ElementTree as ET
from xml.dom import minidom

from torch.nn import functional as F

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        im_shapes: the list of image shapes
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_list_to_blob([im]))

    blob = processed_ims
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois_blob_real = []

    for i in xrange(len(im_scale_factors)):
        rois, levels = _project_im_rois(im_rois, np.array([im_scale_factors[i]]))
        rois_blob = np.hstack((levels, rois))
        rois_blob_real.append(rois_blob.astype(np.float32, copy=False))

    return rois_blob_real

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1
        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    
    return blobs, im_scale_factors

def iou_other_self_mutual(bb, bbgt):
    bb = bb.astype(np.float32)
    bbgt = bbgt.astype(np.float32)
    bi = np.concatenate([np.maximum(bb[:,0:1], bbgt[0:1]), np.maximum(bb[:,1:2], bbgt[1:2]),
                         np.minimum(bb[:,2:3], bbgt[2:3]), np.minimum(bb[:,3:4], bbgt[3:4])], axis=1)
    iw = bi[:,2] - bi[:,0] + 1
    ih = bi[:,3] - bi[:,1] + 1
    other_area = (bb[:,2] - bb[:,0] + 1) * (bb[:,3] - bb[:,1] + 1)
    self_area = (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
    mask = (np.greater(iw,0)*np.greater(ih,0)).astype(np.float32)
    cross_area = iw * ih * mask
    return cross_area/other_area, cross_area/self_area, cross_area/(other_area+self_area-cross_area)

def im_detect(data, rois, labels, model):
  inputs = {'data': [Variable(torch.from_numpy(data))],
            'rois': [Variable(torch.from_numpy(rois))],
            'labels': [Variable(torch.from_numpy(labels))],
            'seg_map': [Variable(torch.from_numpy(np.zeros((1,1))))]}

  pcl_prob0, pcl_prob1, pcl_prob2 = model(**inputs)

  scores = pcl_prob2

  scores = scores.data.cpu().numpy()

  return scores[:, 1:].copy()

def im_detect_cls(data, rois, labels, net):
    data_tensor = Variable(torch.from_numpy(data)).cuda()
    rois_tensor = Variable(torch.from_numpy(rois)).cuda()
    
    base_feat = fasterRCNN.RCNN_base(data_tensor)
    pooled_feat = fasterRCNN.RCNN_roi_pool(base_feat, rois_tensor.view(-1,5).type(base_feat.dtype))
    fc_feat = fasterRCNN._head_to_tail(pooled_feat)
    
    cls0_score0 = fasterRCNN.RCNN_cls0_score0(fc_feat)
    cls0_score1 = fasterRCNN.RCNN_cls0_score1(fc_feat)
    cls0_prob = F.softmax(cls0_score0,1)*F.softmax(cls0_score1,0)
    
    cls1_score0 = fasterRCNN.RCNN_cls1_score0(fc_feat)
    cls1_score1 = fasterRCNN.RCNN_cls1_score1(fc_feat)
    cls1_prob = F.softmax(cls1_score0,1)*F.softmax(cls1_score1,0)
    
    return cls0_prob.data.cpu().numpy(), cls1_prob.data.cpu().numpy()

def parse_xml_12(xml_file, re_xml_file, gt_truth, image_name):
    tree = ET.parse(xml_file)
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = tree.find('folder').text
    ET.SubElement(root, 'filename').text = tree.find('filename').text
    
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = tree.find('source').find('database').text
    ET.SubElement(source, 'annotation').text = tree.find('source').find('annotation').text
    ET.SubElement(source, 'image').text = tree.find('source').find('image').text
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = tree.find('size').find('width').text
    ET.SubElement(size, 'height').text = tree.find('size').find('height').text
    ET.SubElement(size, 'depth').text = tree.find('size').find('depth').text
    
    ET.SubElement(root, 'segmented').text = tree.find('segmented').text
    
    for obj in gt_truth:
        obj_struct = ET.SubElement(root, 'object')
        ET.SubElement(obj_struct, 'name').text = obj[0]
        bndbox = ET.SubElement(obj_struct, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj[1])
        ET.SubElement(bndbox, 'ymin').text = str(obj[2])
        ET.SubElement(bndbox, 'xmax').text = str(obj[3])
        ET.SubElement(bndbox, 'ymax').text = str(obj[4])
    xmltsr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=6*' ')
    
    open(re_xml_file, 'w').close()
    
    with open(re_xml_file, 'w') as f:
        f.write(xmltsr)

def parse_xml_07(xml_file, re_xml_file, gt_truth, image_name):
    tree = ET.parse(xml_file)
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = tree.find('folder').text
    ET.SubElement(root, 'filename').text = tree.find('filename').text
    
    source = ET.SubElement(root, 'source')
    ET.SubElement(source, 'database').text = tree.find('source').find('database').text
    ET.SubElement(source, 'annotation').text = tree.find('source').find('annotation').text
    ET.SubElement(source, 'image').text = tree.find('source').find('image').text
    ET.SubElement(source, 'flickrid').text = tree.find('source').find('flickrid').text
    
    owner = ET.SubElement(root, 'owner')
    ET.SubElement(owner, 'flickrid').text = tree.find('owner').find('flickrid').text
    ET.SubElement(owner, 'name').text = tree.find('owner').find('name').text
    
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = tree.find('size').find('width').text
    ET.SubElement(size, 'height').text = tree.find('size').find('height').text
    ET.SubElement(size, 'depth').text = tree.find('size').find('depth').text
    
    ET.SubElement(root, 'segmented').text = tree.find('segmented').text
    
    for obj in gt_truth:
        obj_struct = ET.SubElement(root, 'object')
        ET.SubElement(obj_struct, 'name').text = obj[0]
        bndbox = ET.SubElement(obj_struct, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj[1])
        ET.SubElement(bndbox, 'ymin').text = str(obj[2])
        ET.SubElement(bndbox, 'xmax').text = str(obj[3])
        ET.SubElement(bndbox, 'ymax').text = str(obj[4])
    xmltsr = minidom.parseString(ET.tostring(root)).toprettyxml(indent=6*' ')
    
    open(re_xml_file, 'w').close()
    
    with open(re_xml_file, 'w') as f:
        f.write(xmltsr)
    
    

def _get_seg_map(seg_map_path, im_scale_factors):
    seg_map_path = seg_map_path.replace('JPEGImages','SEG_MAP')
    seg_map_path = seg_map_path.replace('jpg','png')

    seg_map = cv2.imread(seg_map_path)
    seg_map = seg_map[:,:,0]
    if(roidb[0]['flipped']):
        seg_map = np.flip(seg_map, axis=1)

    seg_maps = []
    for im_scale in im_scale_factors:
        seg_map = cv2.resize(seg_map, None, None, fx=im_scale, fy=im_scale,
                             interpolation=cv2.INTER_NEAREST)
        seg_maps.append(seg_map.astype('float32'))
    return seg_maps


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--imdb', dest='imdbval_name',
                      help='tesing imdb',
                      default='voc_2007_test', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--gcn', dest='gcn',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)

  args.cfg_file = "cfgs/{}_gcn.yml".format(args.net) if args.gcn else "cfgs/{}_scale.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = os.path.join(args.load_dir, args.net, args.dataset)
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  tmp_state_dict = checkpoint['model']
  correct_state_dict = {k:tmp_state_dict['module.'+k] for k in fasterRCNN.state_dict()}
  fasterRCNN.load_state_dict(correct_state_dict)
  # fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  #fasterRCNN = mynn.DataParallel(fasterRCNN, minibatch=True)

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0
  num_images = len(imdb.image_index)


  _t = {'im_detect': time.time(), 'misc': time.time()}

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  all_proposal_num = 0.
  consistent_proposal_num_1 = 0.
  consistent_proposal_num_2 = 0.
  num_1 = 0.
  num_2 = 0.
  
  for i in range(num_images):
      image_name = imdb.image_path_at(i)
      image_name = os.path.basename(image_name).split('.')[0]
      im = cv2.imread(imdb.image_path_at(i))
      boxes = roidb[i]['boxes']
      labels = roidb[i]['labels']
      det_tic = time.time()
      blobs, unused_im_scale_factors = _get_blobs(im, boxes)
      for j in range(len(blobs['data'])):
        scores_tmp_1, scores_tmp_2 = im_detect_cls(blobs['data'][j], blobs['rois'][j], roidb[i]['labels'], fasterRCNN)
        scores_tmp = im_detect(blobs['data'][j], blobs['rois'][j], roidb[i]['labels'], fasterRCNN)
        if j == 0:
          scores_1 = scores_tmp_1.copy()
          scores_2 = scores_tmp_2.copy()
          scores = scores_tmp.copy()
        else:
          scores_1 += scores_tmp_1
          scores_2 += scores_tmp_2
          scores += scores
      pred_boxes = boxes.copy()
      for j in range(imdb.num_classes):
          if labels[0,j] > 0:
              all_proposal_num += 1
              cls_name = imdb._classes[j]
              ind_1 = np.argmax(scores_1[:, j])
              ind_2 = np.argmax(scores_2[:, j])
              ind = np.argmax(scores[:, j])
              gt_box_1 = boxes[ind_1, :].copy()
              gt_box_2 = boxes[ind_2, :].copy()
              gt_box = boxes[ind, :].copy()
              
              _, _, mutual_iou_1 = iou_other_self_mutual(np.expand_dims(gt_box_1,0), gt_box)
              _, _, mutual_iou_2 = iou_other_self_mutual(np.expand_dims(gt_box_2,0), gt_box)
              if mutual_iou_1[0] > 0.8:
                  consistent_proposal_num_1 += 1
              if mutual_iou_2[0] > 0.8:
                  consistent_proposal_num_2 += 1
              if ind_1 ==ind:
                  num_1 += 1
              if ind_2 == ind:
                  num_2 += 1

      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

  end = time.time()
  rate_1 = consistent_proposal_num_1 / all_proposal_num
  rate_2 = consistent_proposal_num_2 / all_proposal_num
  if not os.path.exists('consistent_rate_0.8.txt'):
      open('consistent_rate_0.8.txt', 'w').close()
  with open('consistent_rate_0.8.txt', 'a+') as f:
      f.write(str(rate_1)+'\t')
      f.write(str(rate_2)+'\n')
  if not os.path.exists('consistent_rate_1.txt'):
      open('consistent_rate_1.txt', 'w').close()
  with open('consistent_rate_1.txt', 'a+') as f:
      f.write(str(num_1 / all_proposal_num)+'\t')
      f.write(str(num_2 / all_proposal_num)+'\n')
  print("test time: %0.4fs" % (end - start))
  print('all proposal:{:.1f}, inconsistent proposal:{:.1f}, rate is {:.6f}'\
        .format(all_proposal_num, consistent_proposal_num_1, rate_1))
  print('all proposal:{:.1f}, inconsistent proposal:{:.1f}, rate is {:.6f}'\
        .format(all_proposal_num, consistent_proposal_num_2, rate_2))