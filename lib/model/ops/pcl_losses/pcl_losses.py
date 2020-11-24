#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:48:44 2019

@author: vasgaoweithu
"""
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.nn.modules.module import Module

from . import pcl_losses_cpu
from . import pcl_losses_cuda

class PCLLosses(Function):
    @staticmethod
    def forward(ctx, pcl_probs, labels, cls_loss_weights, gt_assignment,
                pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                im_labels):
        device_id = pcl_probs.get_device()
        output = pcl_probs.new(1, pcl_probs.shape[1]).zero_()
        if not pcl_probs.is_cuda:
            ctx.save_for_backward(pcl_probs, labels, cls_loss_weights,
                              gt_assignment, pc_labels, pc_probs,
                              pc_count, img_cls_loss_weights, im_labels,
                              torch.tensor(device_id))
            pcl_losses_cpu.forward(pcl_probs, labels, cls_loss_weights,
                                      pc_labels, pc_probs, img_cls_loss_weights, 
                                      im_labels, output)
        else:
            ctx.save_for_backward(pcl_probs, labels, cls_loss_weights,
                              gt_assignment, pc_labels, pc_probs,
                              pc_count, img_cls_loss_weights, im_labels,
                              torch.tensor(device_id))
            pcl_losses_cuda.forward(pcl_probs, labels, cls_loss_weights,
                                      pc_labels, pc_probs, img_cls_loss_weights, 
                                      im_labels, output)
        return output.sum() / pcl_probs.size(0)

    @staticmethod
    def backward(ctx, grad_output):
        pcl_probs, labels, cls_loss_weights, gt_assignment, pc_labels, pc_probs, \
        pc_count, img_cls_loss_weights, im_labels, device_id = ctx.saved_tensors
        
        grad_input = grad_output.new(pcl_probs.size()).zero_()
        if not grad_output.is_cuda:
            pcl_losses_cpu.backward(pcl_probs, labels, cls_loss_weights,
                                       gt_assignment, pc_labels, pc_probs, 
                                       pc_count, img_cls_loss_weights, im_labels,
                                       grad_output, grad_input)
        else:
            pcl_losses_cuda.backward(pcl_probs, labels, cls_loss_weights,
                                       gt_assignment, pc_labels, pc_probs, 
                                       pc_count, img_cls_loss_weights, im_labels,
                                       grad_output, grad_input)
        grad_input /= pcl_probs.size(0)

        return grad_input, None, None, None, None, None, None, None, None

pcl_losses = PCLLosses.apply

class  _PCL_Losses(Module):

    def forward(self, pcl_prob, labels, cls_loss_weights, gt_assignment,
                pc_labels, pc_probs, pc_count, img_cls_loss_weights,
                im_labels_real):
        return pcl_losses(pcl_prob, labels, cls_loss_weights, gt_assignment,
                           pc_labels, pc_probs, pc_count, img_cls_loss_weights, im_labels_real)