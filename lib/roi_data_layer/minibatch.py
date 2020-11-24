# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
from model.utils.cython_bbox import bbox_overlaps
import pdb
import cv2
import os

def _get_seg_map(roidb,im_scale):
  seg_map_path = roidb[0]['image']
  seg_map_path = seg_map_path.replace('JPEGImages','SegMap')
  seg_map_path = seg_map_path.replace('jpg','png')

  seg_map = cv2.imread(seg_map_path)
  seg_map = seg_map[:,:,0:1]
  if(roidb[0]['flipped']):
    seg_map = np.flip(seg_map, axis=1)

  seg_map = cv2.resize(seg_map, None, None, fx=im_scale, fy=im_scale,
                      interpolation=cv2.INTER_NEAREST)
  #print(np.shape(seg_map))
  return seg_map.astype('float32')

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

  #seg_map = _get_seg_map(roidb,im_scales[0])

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"

  rois_blob = np.zeros((0, 5), dtype=np.float32)
  labels_blob = np.zeros((0, num_classes), dtype=np.float32)

  for im_i in range(num_images):
    labels, im_rois = _sample_rois(roidb[im_i], num_classes)

    # Add to RoIs blob
    rois = _project_im_rois(im_rois, im_scales[im_i])
    batch_ind = im_i * np.ones((rois.shape[0], 1))
    rois_blob_this_image = np.hstack((batch_ind, rois))

    if cfg.DEDUP_BOXES > 0:
      v = np.array([1, 1e3, 1e6, 1e9, 1e12])
      hashes = np.round(rois_blob_this_image * cfg.DEDUP_BOXES).dot(v)
      _, index, inv_index = np.unique(hashes, return_index=True,
                                      return_inverse=True)
      rois_blob_this_image = rois_blob_this_image[index, :]

    rois_blob = np.vstack((rois_blob, rois_blob_this_image))

    # Add to labels blob
    labels_blob = np.vstack((labels_blob, labels))

  blobs = {'data' : im_blob,
           'rois' : rois_blob,
           'labels' : labels_blob}

  return blobs

def _sample_rois(roidb, num_classes):
  """Generate a random sample of RoIs"""
  labels = roidb['labels']
  rois = roidb['boxes']

  if cfg.TRAIN.BATCH_SIZE > 0:
    batch_size = cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.IMS_PER_BATCH
    if batch_size < rois.shape[0]:
      rois_inds = npr.permutation(rois.shape[0])[:batch_size]
      rois = rois[rois_inds, :]

  return labels.reshape(1, -1), rois

def _get_image_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
  """Project image RoIs into the rescaled training image."""
  rois = im_rois * im_scale_factor
  return rois
