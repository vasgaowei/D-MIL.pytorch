from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.cython_bbox import bbox_overlaps
from model.utils.config import cfg

import numpy as np
from sklearn.cluster import KMeans

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def PCL(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob = cls_prob.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    eps = 1e-9
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = _get_highest_score_proposals(boxes.copy(), cls_prob.copy(), 
        im_labels.copy())

    labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights = _get_proposal_clusters(boxes.copy(), 
            proposals, im_labels.copy(), cls_prob_new.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.float32).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'pc_labels' : pc_labels.reshape(1, -1).astype(np.float32).copy(),
            'pc_probs' : pc_probs.reshape(1, -1).astype(np.float32).copy(),
            'pc_count' : pc_count.reshape(1, -1).astype(np.float32).copy(),
            'img_cls_loss_weights' : img_cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'im_labels_real' : np.hstack((np.array([[1]]), im_labels)).astype(np.float32).copy()}

def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        if len(po_index) > 0:
            img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
            pc_labels[i] = gt_labels[i, 0]
            pc_count[i] = len(po_index)
            pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])
        else:
            img_cls_loss_weights[i] = 0
            pc_labels[i] = gt_labels[i, 0]
            pc_count[i] = 0
            pc_probs[i] = 0

    return labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, pc_count, img_cls_loss_weights

def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)

            gt_boxes = np.vstack((gt_boxes, boxes[max_index, :].reshape(1, -1)))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
            gt_scores = np.vstack((gt_scores,
                                   cls_prob_tmp[max_index] * np.ones((1, 1), dtype=np.float32)))
            cls_prob[max_index, :] = 0

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals
