# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from nms.pth_nms import pth_nms
from nms.cpu_soft_nms import cpu_soft_nms

def nms(dets, thresh):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  return pth_nms(dets, thresh)

def soft_nms(dets, iou_thr, method='gaussian', sigma=0.25, min_score=1e-4):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = cpu_soft_nms(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
