#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:25:57 2019

@author: vasgaoweithu
"""

from torch.nn.modules.module import Module
import torch

class _PCL_Losses(Module):
    def forward(self, pcl_prob, labels, cls_loss_weights, gt_assignment,
                pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                im_labels_real):
        loss = torch.tensor([0], dtype=pcl_prob.dtype).to(pcl_prob.device)
        eps = 1e-6
        num_class = im_labels_real.shape[1]
        for c in range(num_class):
            if im_labels_real[0, c] > 0:
                if c == 0:
                    bg_inds = (labels==0).nonzero()[:,1]
                    loss += - (cls_loss_weights[0, bg_inds] * torch.log(pcl_prob[bg_inds, 0].clamp(eps))).sum()
                else:
                    pc_ind = (pc_labels==c).nonzero()[:, 1]
                    loss += - (pc_count[0, pc_ind] * img_cls_loss_weights[0, pc_ind] * torch.log(pc_probs[0, pc_ind].clamp(eps))).sum()
        return loss / pcl_prob.size(0)
        