# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
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
assert torch.cuda.device_count() == 1
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
#from model.nms.nms_wrapper import nms
from model.ops.nms.nms_wrapper import nms
# from model.utils.cython_nms import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.dmil.vgg16 import vgg16
from model.utils.blob import im_list_to_blob
#import nn as mynn

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

def im_detect(data, rois, labels, model):
  inputs = {'data': Variable(torch.from_numpy(data)),
            'rois': Variable(torch.from_numpy(rois)),
            'labels': Variable(torch.from_numpy(labels))}

  pcl_prob0, pcl_prob1, pcl_prob2 = model(**inputs)

  scores = pcl_prob0 + pcl_prob1 + pcl_prob2

  scores = scores.data.cpu().numpy()

  return scores[:, 1:].copy()

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

  parser.add_argument('--num_worker',
                      help='checksession to load model',
                      type=int)
  parser.add_argument('--worker_id',
                      help='checksession to load model',
                      type=int)

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

  args.cfg_file = "cfgs/{}_gcn.yml".format(args.net) if args.gcn else "cfgs/{}.yml".format(args.net)
  args.worker_id = args.worker_id - 0

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = os.path.join(args.load_dir, args.net, args.dataset)
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    dmil_model = vgg16(imdb.classes, pretrained=False)
  else:
    print("network is not defined")
    pdb.set_trace()

  dmil_model.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  tmp_state_dict = checkpoint['model']
  correct_state_dict = {k:tmp_state_dict['module.'+k] for k in dmil_model.state_dict()}
  dmil_model.load_state_dict(correct_state_dict)
  # fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    dmil_model.cuda()

  dmil_model = nn.DataParallel(dmil_model)

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_{}_{}_{}'.format(args.checksession, args.checkepoch, args.checkpoint)
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  boxes_all = []
  scores_all = []

  output_dir = os.path.join(args.load_dir, args.net, args.imdbval_name, save_name)
  if not os.path.exists(output_dir) and args.worker_id == 0:
      os.makedirs(output_dir)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections_{}.pkl'.format(args.worker_id))
  #args.worker_id = args.worker_id - 4
  dmil_model.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  test_list = range(num_images//args.num_worker*args.worker_id, num_images//args.num_worker*(args.worker_id+1))
  if args.worker_id == args.num_worker - 1:
      test_list = range(num_images//args.num_worker*args.worker_id, num_images)
  for i in test_list:
      im = cv2.imread(imdb.image_path_at(i))
      boxes = roidb[i]['boxes']
      det_tic = time.time()
      blobs, unused_im_scale_factors = _get_blobs(im, boxes)
      for j in range(len(blobs['data'])):
        scores_tmp = im_detect(blobs['data'][j], blobs['rois'][j], roidb[i]['labels'], dmil_model)

        if cfg.TEST.USE_FLIPPED:
          blobs['data'][j] = blobs['data'][j][:, :, :, ::-1].copy()
          width = blobs['data'][j].shape[3]
          oldx1 = blobs['rois'][j][:, 1].copy()
          oldx2 = blobs['rois'][j][:, 3].copy()
          blobs['rois'][j][:, 1] = width - oldx2 - 1
          blobs['rois'][j][:, 3] = width - oldx1 - 1

          scores_tmp += im_detect(blobs['data'][j], blobs['rois'][j], roidb[i]['labels'], dmil_model)

        if j == 0:
          scores = scores_tmp.copy()
        else:
          scores += scores_tmp

      scores /= (len(blobs['data']) * (1. + cfg.TEST.USE_FLIPPED))
      pred_boxes = boxes.copy()

      # scores_all.append(scores)
      # boxes_all.append(pred_boxes)

      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(imdb.num_classes):
          inds = np.where((scores[:, j] > thresh))[0]
          # if there is det
          if len(inds) > 0:
            cls_scores = scores[inds, j]
            top_inds = np.argsort(-cls_scores)
            cls_boxes = pred_boxes[inds, :]

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            cls_dets = cls_dets[top_inds, :]

            _, keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets, 0.3)
            all_boxes[j][i] = cls_dets.copy()
          else:
            all_boxes[j][i] = empty_array

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(imdb.num_classes)])
          if len(image_scores) > max_per_image:
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)

  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, protocol=0)
  with open(det_file+'.complete', 'wb') as f:
      pickle.dump(0, f, protocol=0)

  if args.worker_id == 0:
      comp_flag = False
      while not comp_flag:
          comp_flag = True
          for worker_id in range(1, args.num_worker):
              det_file = os.path.join(output_dir, 'detections_{}.pkl'.format(worker_id)+'.complete')
              if not os.path.exists(det_file):
                  comp_flag = False
                  break
          if not comp_flag:
              time.sleep(60)

      for worker_id in range(1, args.num_worker):
        det_file = os.path.join(output_dir, 'detections_{}.pkl'.format(worker_id))
        with open(det_file, 'rb') as f:
            tmp_boxes = pickle.load(f)
            test_list = range(num_images // args.num_worker * worker_id,
                              num_images // args.num_worker * (worker_id + 1))
            if worker_id == args.num_worker - 1:
                test_list = range(num_images // args.num_worker * worker_id, num_images)
            for cls_index in range(imdb.num_classes):
                for im_index in test_list:
                    all_boxes[cls_index][im_index] = tmp_boxes[cls_index][im_index]

      print('length of concated all boxes: {}'.format(len(all_boxes[0])))

      print('Evaluating detections')
      imdb.evaluate_detections(all_boxes, output_dir)

      end = time.time()
      print("test time: %0.4fs" % (end - start))
