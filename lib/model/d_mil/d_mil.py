import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.pcl.pcl import PCL, _get_proposal_clusters
import time
import pdb
from model.utils.cython_bbox import bbox_overlaps
from model.ops.roi_align import RoIAlign
from torchvision.ops import RoIPool
import cv2
import os

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class DMIL(nn.Module):
    """ faster RCNN """
    def __init__(self, classes):
        super(DMIL, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_pcl_loss0 = 0
        self.RCNN_pcl_loss1 = 0
        self.RCNN_pcl_loss2 = 0

        self.RCNN_roi_pool = RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/8.0)
        self.RCNN_roi_align = RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/8.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        
        self.pcl_losses0 = PCL_Losses()
        self.pcl_losses1 = PCL_Losses()
        self.pcl_losses2 = PCL_Losses()

    def forward(self, data, rois, labels):
        batch_size = data[0].size(0)
        if self.training:
            rois = rois[0].squeeze(dim=0)
            labels = labels[0].squeeze(dim=0)

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(data[0])

        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5).type(base_feat.dtype))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute object classification probability
        pcl_score0 = self.RCNN_pcl_score0(pooled_feat)
        pcl_prob0 = F.softmax(pcl_score0, 1)
        pcl_score1 = self.RCNN_pcl_score1(pooled_feat)
        pcl_prob1 = F.softmax(pcl_score1, 1)
        pcl_score2 = self.RCNN_pcl_score2(pooled_feat)
        pcl_prob2 = F.softmax(pcl_score2, 1)

        if not self.training:
            return pcl_prob0, pcl_prob1, pcl_prob2
        else:
            eps = 1e-9
            device_id = pcl_prob0.get_device()
            epoch = cfg.TRAIN.EPOCH
            alpha = (epoch - 1) / 17. * 2./3 * 0.1

            cls0_score0 = self.RCNN_cls0_score0(pooled_feat)
            cls0_score1 = self.RCNN_cls0_score1(pooled_feat)
            cls0_prob = F.softmax(cls0_score0, 1) * F.softmax(cls0_score1, 0)
            
            cls1_score0 = self.RCNN_cls1_score0(pooled_feat)
            cls1_score1 = self.RCNN_cls1_score1(pooled_feat)
            cls1_prob = F.softmax(cls1_score0, 1) * F.softmax(cls1_score1, 0)
            
            cls_score1 = cls0_score1 + cls1_score1
            
            cls0_prob_cl = F.softmax(cls0_score0, 1) * F.softmax(cls_score1, 0)
            cls1_prob_cl = F.softmax(cls1_score0, 1) * F.softmax(cls_score1, 0)
             
            im_cls_prob_0 = cls0_prob.sum(dim=0, keepdim=True)
            im_cls_prob_1 = cls1_prob.sum(dim=0, keepdim=True)
            
            im_cls_prob_cl_0 = cls0_prob_cl.sum(dim=0, keepdim=True)
            im_cls_prob_cl_1 = cls1_prob_cl.sum(dim=0, keepdim=True)
            
            
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]
            
            cls_loss_0 = 0.00001 * cross_entropy_losses(im_cls_prob_0, labels.type(im_cls_prob_0.dtype))
            cls_loss_1 = 0.00001 * cross_entropy_losses(im_cls_prob_1, labels.type(im_cls_prob_1.dtype))
            
            cls_loss_cl_0 = cross_entropy_losses(im_cls_prob_cl_0, labels.type(im_cls_prob_cl_0.dtype))
            cls_loss_cl_1 = cross_entropy_losses(im_cls_prob_cl_1, labels.type(im_cls_prob_cl_1.dtype))
            
            dc_loss = discrepancy_l1_loss(F.softmax(cls0_score1,0), F.softmax(cls1_score1,0), im_labels.copy())
            
            kl_loss_0 = alpha*learner_detector_collaboration(F.softmax(cls0_score0, 1), pcl_prob2.detach().clone(), boxes, im_labels)
            kl_loss_1 = alpha*learner_detector_collaboration(F.softmax(cls1_score0, 1), pcl_prob2.detach().clone(), boxes, im_labels)
            
            proposals0 = instane_selector(cls0_prob.detach().clone().data.cpu().numpy().copy(), cls1_prob.clone().detach().data.cpu().numpy().copy(),boxes.copy(), im_labels.copy())
            pcl_loss0 = self.pcl_losses0(boxes, im_labels, pcl_prob0, proposals0)
            
            proposals1 = get_highest_score_proposals(boxes.copy(), pcl_prob0.detach().data.cpu().numpy().copy(), im_labels.copy())
            pcl_loss1 = self.pcl_losses1(boxes, im_labels, pcl_prob1, proposals1)
            
            proposals2 = get_highest_score_proposals(boxes.copy(), pcl_prob1.detach().data.cpu().numpy().copy(), im_labels.copy())
            pcl_loss2 = self.pcl_losses2(boxes, im_labels, pcl_prob2, proposals2)
            
            return cls_loss_0.unsqueeze(0), cls_loss_1.unsqueeze(0), cls_loss_cl_0.unsqueeze(0), cls_loss_cl_1.unsqueeze(0), kl_loss_0.unsqueeze(0), kl_loss_1.unsqueeze(0), dc_loss.unsqueeze(0), pcl_loss0.unsqueeze(0), pcl_loss1.unsqueeze(0), pcl_loss2.unsqueeze(0)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_cls0_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls0_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls1_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls1_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
        #orthgonal_init(self.RCNN_cls0_score0, self.RCNN_cls1_score0)
        #orthgonal_init(self.RCNN_cls0_score1, self.RCNN_cls1_score1)
        
        normal_init(self.RCNN_pcl_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_pcl_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_pcl_score2, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

def cross_entropy_losses(probs, labels):
    probs = probs.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(probs) - (1 - labels) * torch.log(1 - probs)

    return loss.mean()

def discrepancy_l1_loss(probs_1, probs_2, image_label):
    loss = torch.tensor([0],dtype=probs_1.dtype).to(probs_2.device)
    num_classes = image_label.shape[1]
    for i in np.arange(num_classes):
        if image_label[0,i] > 0:
            prob_minus = -torch.abs(probs_1[:,i] - probs_2[:,i])
            #prob_alpha = torch.max(probs_1[:,i], probs_2[:,i])
            loss += (prob_minus).mean()
    return loss / np.sum(image_label)

def orthgonal_init(fc1,fc2): 
    out_dim, in_dim = fc1.weight.data.shape
    device = fc1.weight.data.device
    for i in np.arange(out_dim):
        weight_random = torch.randn([in_dim, 2]).to(device)
        Q, R = torch.qr(weight_random)
        fc1.weight.data[i,:] = Q[:,0]
        fc2.weight.data[i,:] = Q[:,1]
    fc1.bias.data.zero_()
    fc2.bias.data.zero_()
    
def get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""
    eps = 1e-9
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

def learner_detector_collaboration(prob1, pcl2, rois, im_labels):
    loss = torch.tensor([0], dtype=prob1.dtype).to(prob1.device)
    num_class = im_labels.shape[1]
    for id_cls in range(num_class):
        if im_labels[0, id_cls] > 0:
            id_box = torch.argmax(pcl2[:, id_cls+1])
            _, _, ious = iou_other_self_mutual(rois, rois[id_box])
            id_cluster = np.where(ious>0.7)[0]
            cl_score = prob1[id_cluster, id_cls]
            loss += - torch.log(cl_score.clamp(1e-6)).mean()
    return loss / np.sum(im_labels)
    
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
        overlaps = bbox_overlaps(
            np.ascontiguousarray(boxes, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_labels[gt_assignment, 0]
        cls_loss_weights = gt_scores[gt_assignment, 0]

        # Select background RoIs as those with < FG_THRESH overlap
        bg_inds = np.where(max_overlaps < cfg.TRAIN.FG_THRESH)[0]
        labels[bg_inds] = 0
        gt_assignment[bg_inds] = -1

        ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
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
    
def instane_selector(cls0_prob, cls1_prob, rois, im_labels):
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1), dtype=np.float32),
                 'wsddn_indices': []}
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
                proposals['wsddn_indices'].append(max_area_ind)
                
                su_ids = is_enclosing_surround(tmp_proposals, tmp_proposals[max_area_ind])
                su_ids = list(np.array(proposal_inds)[su_ids])
                
                _,_,mutual_iou = iou_other_self_mutual(tmp_proposals, tmp_proposals[max_area_ind])
                
                remove_inds.extend(list(np.array(proposal_inds)[mutual_iou>0.4]))
                keep_inds.extend(list(np.array(proposal_inds)[mutual_iou<0.1]))
                
                for t_id in list(su_ids):
                    if t_id in keep_inds:
                        keep_inds.remove(t_id)
                
                remove_inds.extend(keep_inds)
                remove_inds.extend(list(su_ids))
                
                proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], top_proposals[list(keep_inds)]))
                proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]*len(keep_inds)]).reshape((-1,1))))
                proposals['gt_scores'] = np.vstack((proposals['gt_scores'], top_scores[list(keep_inds)]))
                proposals['wsddn_indices'].extend(list(keep_inds))
                
                remove_inds = list(set(remove_inds))
                
                for ind in remove_inds:
                    proposal_inds.remove(ind)
    return proposals

def check_cover_baseline(cls0_prob, cls1_prob, rois, seg_map, im_labels):
    mask = np.ones_like(cls1_prob)
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1)),
                 'wsddn_indices': []}
    for cls_index in range(20):
        if im_labels[0, cls_index] > 0:
            cls0_prob_cls = cls0_prob[:, cls_index]
            cls0_top_index = np.argmax(cls0_prob_cls)
            cls0_top_box = rois[cls0_top_index]
            cls0_top_scores = cls0_prob_cls[cls0_top_index]
            cls1_prob_cls = cls0_prob[:, cls_index]
            cls1_top_index = np.argmax(cls1_prob_cls)
            cls1_top_box = rois[cls1_top_index]
            cls1_top_scores = cls0_prob_cls[cls1_top_index]
            proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], np.expand_dims(cls1_top_box, 0)))
            proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]])))
            proposals['gt_scores'] = np.vstack((proposals['gt_scores'], np.expand_dims(cls1_top_scores, 0)))
            proposals['wsddn_indices'].append(1)

            if cls1_top_index != cls0_top_index:
                proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], np.expand_dims(cls0_top_box, 0)))
                proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]])))
                proposals['gt_scores'] = np.vstack((proposals['gt_scores'], np.expand_dims(cls0_top_scores, 0)))
                proposals['wsddn_indices'].append(0)

    return mask, proposals

def check_cover(cls0_prob, cls1_prob, rois, seg_map, im_labels):
    mask = np.ones_like(cls1_prob)
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1)),
                 'wsddn_indices': []}
    for cls_index in range(20):
        if im_labels[0, cls_index] > 0:
            cls0_prob_cls = cls0_prob[:, cls_index]
            cls0_top_index = np.argmax(cls0_prob_cls)
            cls0_top_box = rois[cls0_top_index]
            cls0_top_scores = cls0_prob_cls[cls0_top_index]

            cls_map = (seg_map == cls_index + 1).astype('float32')
            cover_rate = get_cover_rate(cls0_top_box, cls_map)

            if cover_rate > 0 and cover_rate < 0.3:
                other_iou, _, mutual_iou = iou_other_self_mutual(rois, cls0_top_box)
                mask[mutual_iou>0.3, cls_index] = 0

            masked_cls1_prob_cls = cls1_prob[:, cls_index].copy()*mask[:,cls_index]
            cls1_top_index = np.argmax(masked_cls1_prob_cls)
            cls1_top_box = rois[cls1_top_index]
            cls1_top_scores = masked_cls1_prob_cls[cls1_top_index]
            proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], np.expand_dims(cls1_top_box, 0)))
            proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]])))
            proposals['gt_scores'] = np.vstack((proposals['gt_scores'], np.expand_dims(cls1_top_scores, 0)))
            proposals['wsddn_indices'].append(1)

            other_iou, _, mutual_iou = iou_other_self_mutual(np.expand_dims(cls0_top_box,0), cls1_top_box)
            if mutual_iou[0] < 0.1:
                proposals['gt_boxes'] = np.vstack((proposals['gt_boxes'], np.expand_dims(cls0_top_box, 0)))
                proposals['gt_classes'] = np.vstack((proposals['gt_classes'], np.array([[cls_index + 1]])))
                proposals['gt_scores'] = np.vstack((proposals['gt_scores'], np.expand_dims(cls0_top_scores, 0)))
                proposals['wsddn_indices'].append(0)

    return mask, proposals

def get_cover_rate(box, cls_map):
    x,y,x1,y1 = box.astype(np.int32)
    inter_area = np.sum(cls_map[y:y1+1, x:x1+1])
    seg_area = np.sum(cls_map)

    if(seg_area==0):
        cover_rate = 0
    else:
        cover_rate = inter_area/seg_area
    return cover_rate

def get_seg_output(boxes, proposals, im_labels, cls_prob_new):
    eps = 1e-9
    cls_prob_new_np = cls_prob_new.data.cpu().numpy()
    cls_prob_new_np[cls_prob_new_np < eps] = eps
    cls_prob_new_np[cls_prob_new_np > 1 - eps] = 1 - eps
    labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
    pc_count, img_cls_loss_weights = _get_proposal_clusters(boxes.copy(), proposals, im_labels.copy(), cls_prob_new_np.copy())

    return {'labels' : labels.reshape(1, -1).astype(np.float32).copy(),
            'cls_loss_weights' : cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'gt_assignment' : gt_assignment.reshape(1, -1).astype(np.float32).copy(),
            'pc_labels' : pc_labels.reshape(1, -1).astype(np.float32).copy(),
            'pc_probs' : pc_probs.reshape(1, -1).astype(np.float32).copy(),
            'pc_count' : pc_count.reshape(1, -1).astype(np.float32).copy(),
            'img_cls_loss_weights' : img_cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
            'im_labels_real' : np.hstack((np.array([[1]]), im_labels)).astype(np.float32).copy()}

def masked_softmax(input, mask, axis):
    input_max = torch.max(input, dim=axis)[0]
    input_exp = torch.exp(input - input_max)

    input_masked_exp = input_exp * mask
    input_masked_exp_sum = torch.sum(input_masked_exp,  dim=axis)
    input_masked_softmax = input_masked_exp / input_masked_exp_sum
    return input_masked_softmax

def cal_proposal_area(bbox):
    bbox = bbox.astype(np.float32)
    area = (bbox[:, 2] - bbox[:,0] + 1)*(bbox[:,3] - bbox[:,1] + 1)
    return area

def is_enclosing_surround(bb, bbgt):
    x1_bool = bb[:, 0] > bbgt[0]
    y1_bool = bb[:, 1] > bbgt[1]
    x2_bool = bb[:, 2] < bbgt[2]
    y2_bool = bb[:, 3] < bbgt[3]
    
    return (x1_bool&y1_bool&x2_bool&y2_bool).nonzero()[0]


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
