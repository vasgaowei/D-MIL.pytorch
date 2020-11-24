from __future__ import absolute_import
import torch
import numpy as np
from ._ext import nms
import pdb

def nms_gpu(dets, thresh):
    dets = torch.from_numpy(dets).float().cuda()
    keep = torch.IntTensor(dets.size(0)).zero_().cuda()
    num_out = torch.IntTensor(1).zero_().cuda()
    nms.nms_cuda(keep, dets, num_out, thresh)
    keep = keep[:num_out[0]]
    return keep
