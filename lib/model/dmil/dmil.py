import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
#from model.rpn.rpn import _RPN
#from model.roi_pooling.modules.roi_pool import _RoIPooling
#from model.roi_crop.modules.roi_crop import _RoICrop
#from model.roi_align.modules.roi_align import RoIAlignAvg

#from model.ops.roi_pool import RoIPool
from model.ops.roi_align import RoIAlign
from model.ops.roi_crop import RoICrop

from torchvision.ops import RoIPool

#from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb as pdb
from model.utils.net_utils import _affine_grid_gen

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3
from model.utils.cython_bbox import bbox_overlaps

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
        self.RCNN_roi_crop = RoICrop()

        self.pcl_losses0 = PCL_Losses()
        self.pcl_losses1 = PCL_Losses()
        self.pcl_losses2 = PCL_Losses()

    def forward(self, data, rois, labels, data_shape=None):
        if not data.is_cuda:
            data = data.cuda()
        if not rois.is_cuda:
            rois = rois.cuda()
        if not labels.is_cuda:
            labels = labels.cuda()
        
        if not data_shape is None and not data_shape.is_cuda:
            data_shape = data_shape.cuda()
        
        '''For D-MIL
        if not seg_map is None and not seg_map.is_cuda:
            seg_map = seg_map.cuda()
        '''
        batch_size = data.size(0)
        if self.training:
            rois = rois.squeeze(dim=0)
            labels = labels.squeeze(dim=0)
            #seg_map = seg_map.squeeze(dim=0)
        
        if not data_shape is None:
            data_shape = data_shape.squeeze(dim=0)
            data = data[:, :, :data_shape[0], :data_shape[1]]
            rois = rois[:data_shape[2]]
            #if self.training:
            #    seg_map = seg_map[:data_shape[0], :data_shape[1]]

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(data)

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5).type(base_feat.dtype), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5).type(base_feat.dtype))
        elif cfg.POOLING_MODE == 'pool':
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

        if self.training:
            eps = 1e-9
            device_id = pcl_prob0.get_device()

            cls0_score0 = self.RCNN_cls0_score0(pooled_feat)
            cls0_score1 = self.RCNN_cls0_score1(pooled_feat)
            cls0_prob = F.softmax(cls0_score0, 1) * F.softmax(cls0_score1, 0)

            cls1_score0 = self.RCNN_cls1_score0(pooled_feat)
            cls1_score1 = self.RCNN_cls1_score1(pooled_feat)
            cls1_prob = F.softmax(cls1_score0, 1) * F.softmax(cls1_score1, 0)
            
            cls2_score0 = self.RCNN_cls2_score0(pooled_feat)
            cls2_score1 = self.RCNN_cls2_score1(pooled_feat)
            cls2_prob = F.softmax(cls2_score0, 1) * F.softmax(cls2_score1, 0)
            
            cls3_score0 = self.RCNN_cls3_score0(pooled_feat)
            cls3_score1 = self.RCNN_cls3_score1(pooled_feat)
            cls3_prob = F.softmax(cls3_score0, 1) * F.softmax(cls3_score1, 0)
            
            cls_score1 = cls0_score1 + cls1_score1 + cls2_score1 + cls3_score1
            cls0_prob_cls = F.softmax(cls0_score0, 1) * F.softmax(cls_score1, 0)
            cls1_prob_cls = F.softmax(cls1_score0, 1) * F.softmax(cls_score1, 0)
            cls2_prob_cls = F.softmax(cls2_score0, 1) * F.softmax(cls_score1, 0)
            cls3_prob_cls = F.softmax(cls3_score0, 1) * F.softmax(cls_score1, 0)

            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            #seg_map = seg_map.data.cpu().numpy()
            boxes = boxes[:, 1:]
            mask, proposals0 = instance_selector(cls0_prob.data.cpu().numpy(), cls1_prob.data.cpu().numpy(),
                                                           cls2_prob.data.cpu().numpy(), cls3_prob.data.cpu().numpy(), boxes, im_labels)
            im_cls_prob_0 = cls0_prob_cls.sum(dim=0, keepdim=True)
            im_cls_prob_1 = cls1_prob_cls.sum(dim=0, keepdim=True)
            im_cls_prob_2 = cls2_prob_cls.sum(dim=0, keepdim=True)
            im_cls_prob_3 = cls3_prob_cls.sum(dim=0, keepdim=True)
            
            im_cls_cl_prob_0 = cls0_prob.sum(dim=0, keepdim=True)
            im_cls_cl_prob_1 = cls1_prob.sum(dim=0, keepdim=True)
            im_cls_cl_prob_2 = cls2_prob.sum(dim=0, keepdim=True)
            im_cls_cl_prob_3 = cls3_prob.sum(dim=0, keepdim=True)
            
            cls_loss_0 = cross_entropy_losses(im_cls_prob_0, labels.type(im_cls_prob_0.dtype))
            cls_loss_1 = cross_entropy_losses(im_cls_prob_1, labels.type(im_cls_prob_1.dtype))
            cls_loss_2 = cross_entropy_losses(im_cls_prob_2, labels.type(im_cls_prob_2.dtype))   
            cls_loss_3 = cross_entropy_losses(im_cls_prob_3, labels.type(im_cls_prob_3.dtype))
            
            cls_cl_loss_0 = 0.00001*cross_entropy_losses(im_cls_cl_prob_0, labels.type(im_cls_cl_prob_0.dtype))
            cls_cl_loss_1 = 0.00001*cross_entropy_losses(im_cls_cl_prob_1, labels.type(im_cls_cl_prob_1.dtype))
            cls_cl_loss_2 = 0.00001*cross_entropy_losses(im_cls_cl_prob_2, labels.type(im_cls_cl_prob_2.dtype)) 
            cls_cl_loss_3 = 0.00001*cross_entropy_losses(im_cls_cl_prob_3, labels.type(im_cls_cl_prob_3.dtype)) 
            
            dc_loss01 = discrepancy_l1_loss(F.softmax(cls0_score1, 0), F.softmax(cls1_score1, 0), im_labels)
            dc_loss02 = discrepancy_l1_loss(F.softmax(cls0_score1, 0), F.softmax(cls2_score1, 0), im_labels)
            dc_loss03 = discrepancy_l1_loss(F.softmax(cls0_score1, 0), F.softmax(cls3_score1, 0), im_labels)
            dc_loss12 = discrepancy_l1_loss(F.softmax(cls1_score1, 0), F.softmax(cls2_score1, 0), im_labels)
            dc_loss13 = discrepancy_l1_loss(F.softmax(cls1_score1, 0), F.softmax(cls3_score1, 0), im_labels)
            dc_loss23 = discrepancy_l1_loss(F.softmax(cls2_score1, 0), F.softmax(cls3_score1, 0), im_labels)

            pcl_loss0 = self.pcl_losses0(boxes, im_labels, pcl_prob0, proposals0)

            proposals1 = get_highest_score_proposals(boxes.copy(), pcl_prob0, im_labels.copy())
            pcl_loss1 = self.pcl_losses1(boxes, im_labels, pcl_prob1, proposals1)

            proposals2 = get_highest_score_proposals(boxes.copy(), pcl_prob1, im_labels.copy())
            pcl_loss2 = self.pcl_losses2(boxes, im_labels, pcl_prob2, proposals2)
            dc_loss = dc_loss01 + dc_loss02 + dc_loss03 + dc_loss12 + dc_loss13 + dc_loss23
            if np.random.rand() < 0.01:
                print("\t\t\tloss_dc: %.6f, loss_cls0: %.4f, loss_cls1: %.4f, loss_cls2: %.4f, loss_cls2: %.4f, loss_cl_cls0: %.6f, loss_cl_cls1: %.6f, loss_cl_cls2: %.6f, loss_cl_cls3: %.6f, loss_pcl0: %.4f, loss_pcl1: %.4f, loss_pcl2 %.4f" \
                      % (dc_loss.data.cpu().numpy(), cls_loss_0.data.cpu().numpy(), cls_loss_1.data.cpu().numpy(),cls_loss_2.data.cpu().numpy(), cls_loss_3.data.cpu().numpy(), cls_cl_loss_0.data.cpu().numpy(),cls_cl_loss_1.data.cpu().numpy(), cls_cl_loss_2.data.cpu().numpy(), cls_cl_loss_3.data.cpu().numpy(), pcl_loss0.data.cpu().numpy(), pcl_loss1.data.cpu().numpy(), pcl_loss2.data.cpu().numpy()))
            return dc_loss.unsqueeze(0) + cls_loss_0.unsqueeze(0)+ cls_loss_1.unsqueeze(0) + cls_loss_2.unsqueeze(0) + cls_loss_3.unsqueeze(0) + cls_cl_loss_0.unsqueeze(0) + cls_cl_loss_1.unsqueeze(0) + cls_cl_loss_2.unsqueeze(0) + cls_cl_loss_3.unsqueeze(0) + pcl_loss0.unsqueeze(0)+ pcl_loss1.unsqueeze(0)+ pcl_loss2.unsqueeze(0)
        else:
            return pcl_prob0, pcl_prob1, pcl_prob2

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
        normal_init(self.RCNN_cls2_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls2_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls3_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls3_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        
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
        #print('gt', gt_boxes.shape)
        #print('box ', boxes.shape)
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
        return loss / max(800, cls_prob_new.shape[0])

def get_highest_score_proposals(boxes, cls_prob, im_labels):
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

def instance_selector(cls0_prob, cls1_prob, cls2_prob, cls3_prob, rois, im_labels):
    mask = np.ones_like(cls1_prob)
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1), dtype=np.float32),
                 'wsddn_indices': []}
    num_images, num_classes = im_labels.shape
    for cls_index in range(num_classes):
        if im_labels[0, cls_index] > 0:
            cls0_prob_cls = cls0_prob[:, cls_index].copy()
            cls0_top_index = np.argmax(cls0_prob_cls)
            cls0_top_box = rois[cls0_top_index]
            cls0_top_scores = cls0_prob_cls[cls0_top_index]
           
            cls1_prob_cls = cls1_prob[:, cls_index].copy()
            cls1_top_index = np.argmax(cls1_prob_cls)
            cls1_top_box = rois[cls1_top_index]
            cls1_top_scores = cls1_prob_cls[cls1_top_index]
            
            cls2_prob_cls = cls2_prob[:, cls_index].copy()
            cls2_top_index = np.argmax(cls2_prob_cls)
            cls2_top_box = rois[cls2_top_index]
            cls2_top_scores = cls2_prob_cls[cls2_top_index]
            
            cls3_prob_cls = cls3_prob[:, cls_index].copy()
            cls3_top_index = np.argmax(cls3_prob_cls)
            cls3_top_box = rois[cls3_top_index]
            cls3_top_scores = cls3_prob_cls[cls3_top_index]
            
            top_proposals = np.vstack([cls0_top_box, cls1_top_box, cls2_top_box, cls3_top_box])
            top_scores = np.vstack([cls0_top_scores, cls1_top_scores, cls2_top_scores, cls3_top_scores])
            proposal_inds = [0,1,2,3]
            
            
            proposal_area = cal_proposal_area(top_proposals)
            
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
                proposals['wsddn_indices'].extend(list(keep_inds))
                
                remove_inds = list(set(remove_inds))
                
                for ind in remove_inds:
                    proposal_inds.remove(ind)
    return mask, proposals

def check_cover_CL_M_max_area(cls0_prob, cls1_prob, rois, im_labels):
    mask = np.ones_like(cls1_prob)
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1), dtype=np.float32),
                 'wsddn_indices': []}
    num_images, num_classes = im_labels.shape
    for cls_index in range(num_classes):
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
                proposals['wsddn_indices'].extend(list(keep_inds))
                
                remove_inds = list(set(remove_inds))
                
                for ind in remove_inds:
                    proposal_inds.remove(ind)
    return mask, proposals

def check_cover(cls0_prob, cls1_prob, rois, seg_map, im_labels):
    mask = np.ones_like(cls1_prob).astype(np.float32)
    proposals = {'gt_boxes' : np.zeros((0,4)),
                 'gt_classes': np.zeros((0,1)),
                 'gt_scores': np.zeros((0,1), dtype=np.float32),
                 'wsddn_indices': []}
    num_images, num_classes = im_labels.shape
    for cls_index in range(num_classes):
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
