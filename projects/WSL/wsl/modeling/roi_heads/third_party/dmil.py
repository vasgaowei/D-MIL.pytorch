from __future__ import absolute_import
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn


from detectron2.structures import Boxes, pairwise_iou

# import utils.boxes as box_utils
# from core.config import cfg

cfg_TRAIN_NUM_KMEANS_CLUSTER = 3
cfg_RNG_SEED = 3
cfg_TRAIN_GRAPH_IOU_THRESHOLD = 0.4
cfg_TRAIN_MAX_PC_NUM = 5
cfg_TRAIN_FG_THRESH = 0.5
cfg_TRAIN_BG_THRESH = 0.1


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def DMIL(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob_0, cls_prob_1 = cls_prob
    
    cls_prob_0 = cls_prob_0.data.cpu().numpy()
    cls_prob_1 = cls_prob_1.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    # print(cls_prob.shape, cls_prob_new.shape, im_labels.shape)
    if cls_prob_0.shape[1] != im_labels.shape[1]:
        cls_prob_0 = cls_prob_0[:, 1:]
        cls_prob_1 = cls_prob_1[:, 1:]
    eps = 1e-9
    cls_prob_0[cls_prob_0 < eps] = eps
    cls_prob_0[cls_prob_0 > 1 - eps] = 1 - eps
    cls_prob_1[cls_prob_1 < eps] = eps
    cls_prob_1[cls_prob_1 > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = instane_selector(cls_prob_0.copy(), cls_prob_1.copy(), boxes.copy(), im_labels.copy())

    (
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
    ) = _get_proposal_clusters(boxes.copy(), proposals, im_labels.copy(), cls_prob_new.copy())

    return {
        "labels": labels.reshape(1, -1).astype(np.float32).copy(),
        "cls_loss_weights": cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        "gt_assignment": gt_assignment.reshape(1, -1).astype(np.float32).copy(),
        "pc_labels": pc_labels.reshape(1, -1).astype(np.float32).copy(),
        "pc_probs": pc_probs.reshape(1, -1).astype(np.float32).copy(),
        "pc_count": pc_count.reshape(1, -1).astype(np.float32).copy(),
        "img_cls_loss_weights": img_cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        "im_labels_real": np.hstack((np.array([[1]]), im_labels)).astype(np.float32).copy(),
    }

def DMILV1(boxes, cls_prob, im_labels, cls_prob_new):

    cls_prob_0, cls_prob_1 = cls_prob
    
    cls_prob_0 = cls_prob_0.data.cpu().numpy()
    cls_prob_1 = cls_prob_1.data.cpu().numpy()
    cls_prob_new = cls_prob_new.data.cpu().numpy()
    # print(cls_prob.shape, cls_prob_new.shape, im_labels.shape)
    if cls_prob_0.shape[1] != im_labels.shape[1]:
        cls_prob_0 = cls_prob_0[:, 1:]
        cls_prob_1 = cls_prob_1[:, 1:]
    eps = 1e-9
    cls_prob_0[cls_prob_0 < eps] = eps
    cls_prob_0[cls_prob_0 > 1 - eps] = 1 - eps
    cls_prob_1[cls_prob_1 < eps] = eps
    cls_prob_1[cls_prob_1 > 1 - eps] = 1 - eps
    cls_prob_new[cls_prob_new < eps] = eps
    cls_prob_new[cls_prob_new > 1 - eps] = 1 - eps

    proposals = instane_selector(cls_prob_0.copy(), cls_prob_1.copy(), boxes.copy(), im_labels.copy())

    return proposals
def PCLV1(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""
    eps = 1e-9
    cls_prob = cls_prob.data.cpu().numpy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]
    cls_prob[cls_prob < eps] = eps
    cls_prob[cls_prob > 1 - eps] = 1 - eps

    num_images, num_classes = im_labels.shape
    assert num_images == 1
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] > 0:
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

def instane_selector(cls0_prob, cls1_prob, rois, im_labels):
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1), dtype=np.float32)
                 }
    for cls_index in range(20):
        if im_labels[0, cls_index] > 0:
            cls0_prob_cls = cls0_prob[:, cls_index].copy()
            cls0_top_index = np.argmax(cls0_prob_cls)
            cls0_top_box = rois[cls0_top_index]
            cls0_top_scores = cls0_prob_cls[cls0_top_index]
           
            cls1_prob_cls = cls1_prob[:, cls_index].copy()
            cls1_top_index = np.argmax(cls1_prob_cls)
            cls1_top_box = rois[cls1_top_index]
            cls1_top_scores = cls1_prob_cls[cls1_top_index]
            
            
            top_proposals = np.vstack([cls0_top_box, cls1_top_box])
            top_scores = np.vstack([cls0_top_scores, cls1_top_scores])
            
            proposal_area = cal_proposal_area(top_proposals)
            proposal_inds = [0,1]
            
            while len(proposal_inds) > 0:
                remove_inds = []
                keep_inds = []
                tmp_proposals = top_proposals[list(proposal_inds)].copy()
                tmp_scores = top_scores[list(proposal_inds)].copy()
                tmp_area = proposal_area[list(proposal_inds)].copy()
                
                
                max_area_ind = np.argmax(tmp_area)
                proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], tmp_proposals[[max_area_ind]]))
                proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]])))
                proposals['gt_scores'] = np.vstack((proposals['gt_scores'], tmp_scores[[max_area_ind]]))
                
                su_ids = is_enclosing_surround(tmp_proposals, tmp_proposals[max_area_ind])
                su_ids = list(np.array(proposal_inds)[su_ids])
                
                _,_,mutual_iou = iou_other_self_mutual(tmp_proposals, tmp_proposals[max_area_ind])
                
                remove_inds.extend(list(np.array(proposal_inds)[mutual_iou>0.3]))
                keep_inds.extend(list(np.array(proposal_inds)[mutual_iou<0.1]))
                
                for t_id in list(su_ids):
                    if t_id in keep_inds:
                        keep_inds.remove(t_id)
                
                remove_inds.extend(keep_inds)
                remove_inds.extend(list(su_ids))
                
                proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], top_proposals[list(keep_inds)]))
                proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]*len(keep_inds)]).reshape((-1,1))))
                proposals['gt_scores'] = np.vstack((proposals['gt_scores'], top_scores[list(keep_inds)]))
                
                remove_inds = list(set(remove_inds))
                
                for ind in remove_inds:
                    proposal_inds.remove(ind)
    return proposals

def cal_proposal_area(bbox):
    bbox = bbox.astype(np.float32)
    area = (bbox[:, 2] - bbox[:,0] + 1)*(bbox[:,3] - bbox[:,1] + 1)
    return area

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

def is_enclosing_surround(bb, bbgt):
    x1_bool = bb[:, 0] > bbgt[0]
    y1_bool = bb[:, 1] > bbgt[1]
    x2_bool = bb[:, 2] < bbgt[2]
    y2_bool = bb[:, 3] < bbgt[3]
    
    return (x1_bool&y1_bool&x2_bool&y2_bool).nonzero()[0]

class PCL_Losses(nn.Module):
    def __init__(self):
        super(PCL_Losses, self).__init__()

    def forward(self, boxes, im_labels, cls_prob_new, proposals):
        eps = 1e-9
        cls_prob_new = cls_prob_new.clamp(eps, 1 - eps)

        num_images, num_classes = im_labels.shape
        assert num_images == 1, 'batch size shoud be equal to 1'
        # overlaps: (rois x gt_boxes)
        gt_boxes = proposals['gt_boxes']
        gt_labels = proposals['gt_classes'].astype(np.long)
        gt_scores = proposals['gt_scores']
        overlaps = pairwise_iou(Boxes(boxes), Boxes(gt_boxes))
        overlaps = overlaps.data.cpu().numpy()
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_labels[gt_assignment, 0]
        cls_loss_weights = gt_scores[gt_assignment, 0]

        # Select background RoIs as those with < FG_THRESH overlap
        bg_inds = np.where(max_overlaps < cfg_TRAIN_FG_THRESH)[0]
        labels[bg_inds] = 0
        gt_assignment[bg_inds] = -1

        ig_inds = np.where(max_overlaps < cfg_TRAIN_BG_THRESH)[0]
        cls_loss_weights[ig_inds] = 0.0

        device_id = cls_prob_new.get_device()
        cls_loss_weights = torch.from_numpy(cls_loss_weights)
        labels = torch.from_numpy(labels)
        gt_assignment = torch.from_numpy(gt_assignment)
        gt_labels = torch.from_numpy(gt_labels)
        gt_scores = torch.from_numpy(gt_scores).cuda(device_id)

        loss = torch.tensor(0.).cuda(device_id)

        for i in range(len(gt_boxes)):
            p_mask = torch.where(gt_assignment==i,
                                 torch.ones_like(gt_assignment, dtype=torch.float),
                                 torch.zeros_like(gt_assignment, dtype=torch.float)).cuda(device_id)
            p_count = torch.sum(p_mask)
            if p_count > 0:
                mean_prob = torch.sum(cls_prob_new[:,gt_labels[i,0]]*p_mask)/p_count
                loss = loss - torch.log(mean_prob)*p_count*gt_scores[i,0]
        n_mask = torch.where(labels==0, cls_loss_weights, torch.zeros_like(labels, dtype=torch.float)).cuda(device_id)
        loss = loss - torch.sum(torch.log(cls_prob_new[:,0])*n_mask)
        return loss / cls_prob_new.shape[0]

def _get_top_ranking_propoals(probs):
    """Get top ranking proposals by k-means"""
    n_clusters = min(cfg_TRAIN_NUM_KMEANS_CLUSTER, probs.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg_RNG_SEED).fit(probs)
    high_score_label = np.argmax(kmeans.cluster_centers_)

    index = np.where(kmeans.labels_ == high_score_label)[0]

    if len(index) == 0:
        index = np.array([np.argmax(probs)])

    return index


def _build_graph(boxes, iou_threshold):
    """Build graph based on box IoU"""
    # overlaps = box_utils.bbox_overlaps(
    # boxes.astype(dtype=np.float32, copy=False),
    # boxes.astype(dtype=np.float32, copy=False))
    overlaps = pairwise_iou(Boxes(boxes), Boxes(boxes))
    overlaps = overlaps.data.cpu().numpy()

    return (overlaps > iou_threshold).astype(np.float32)


def _get_graph_centers(boxes, cls_prob, im_labels):
    """Get graph centers."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, "batch size shoud be equal to 1"
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            idxs = np.where(cls_prob_tmp >= 0)[0]
            idxs_tmp = _get_top_ranking_propoals(cls_prob_tmp[idxs].reshape(-1, 1))
            idxs = idxs[idxs_tmp]
            boxes_tmp = boxes[idxs, :].copy()
            cls_prob_tmp = cls_prob_tmp[idxs]

            graph = _build_graph(boxes_tmp, cfg_TRAIN_GRAPH_IOU_THRESHOLD)

            keep_idxs = []
            gt_scores_tmp = []
            count = cls_prob_tmp.size
            while True:
                order = np.sum(graph, axis=1).argsort()[::-1]
                tmp = order[0]
                keep_idxs.append(tmp)
                inds = np.where(graph[tmp, :] > 0)[0]
                gt_scores_tmp.append(np.max(cls_prob_tmp[inds]))

                graph[:, inds] = 0
                graph[inds, :] = 0
                count = count - len(inds)
                if count <= 5:
                    break

            gt_boxes_tmp = boxes_tmp[keep_idxs, :].copy()
            gt_scores_tmp = np.array(gt_scores_tmp).copy()

            keep_idxs_new = np.argsort(gt_scores_tmp)[
                -1 : (-1 - min(len(gt_scores_tmp), cfg_TRAIN_MAX_PC_NUM)) : -1
            ]

            gt_boxes = np.vstack((gt_boxes, gt_boxes_tmp[keep_idxs_new, :]))
            gt_scores = np.vstack((gt_scores, gt_scores_tmp[keep_idxs_new].reshape(-1, 1)))
            gt_classes = np.vstack(
                (gt_classes, (i + 1) * np.ones((len(keep_idxs_new), 1), dtype=np.int32))
            )

            # If a proposal is chosen as a cluster center,
            # we simply delete a proposal from the candidata proposal pool,
            # because we found that the results of different strategies are similar and this strategy is more efficient
            cls_prob = np.delete(cls_prob.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)
            boxes = np.delete(boxes.copy(), idxs[keep_idxs][keep_idxs_new], axis=0)

    proposals = {"gt_boxes": gt_boxes, "gt_classes": gt_classes, "gt_scores": gt_scores}

    return proposals


def _get_proposal_clusters(all_rois, proposals, im_labels, cls_prob):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    num_images, num_classes = im_labels.shape
    assert num_images == 1, "batch size shoud be equal to 1"
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals["gt_boxes"]
    gt_labels = proposals["gt_classes"]
    gt_scores = proposals["gt_scores"]
    # overlaps = box_utils.bbox_overlaps(
    # all_rois.astype(dtype=np.float32, copy=False),
    # gt_boxes.astype(dtype=np.float32, copy=False))

    overlaps = pairwise_iou(Boxes(all_rois), Boxes(gt_boxes))
    overlaps = overlaps.data.cpu().numpy()

    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(max_overlaps >= cfg_TRAIN_FG_THRESH)[0]

    # Select background RoIs as those with < FG_THRESH overlap
    bg_inds = np.where(max_overlaps < cfg_TRAIN_FG_THRESH)[0]

    ig_inds = np.where(max_overlaps < cfg_TRAIN_BG_THRESH)[0]
    cls_loss_weights[ig_inds] = 0.0

    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    img_cls_loss_weights = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_probs = np.zeros(gt_boxes.shape[0], dtype=np.float32)
    pc_labels = np.zeros(gt_boxes.shape[0], dtype=np.int32)
    pc_count = np.zeros(gt_boxes.shape[0], dtype=np.int32)

    for i in xrange(gt_boxes.shape[0]):
        po_index = np.where(gt_assignment == i)[0]
        img_cls_loss_weights[i] = np.sum(cls_loss_weights[po_index])
        pc_labels[i] = gt_labels[i, 0]
        pc_count[i] = len(po_index)
        pc_probs[i] = np.average(cls_prob[po_index, pc_labels[i]])

    return (
        labels,
        cls_loss_weights,
        gt_assignment,
        pc_labels,
        pc_probs,
        pc_count,
        img_cls_loss_weights,
    )

