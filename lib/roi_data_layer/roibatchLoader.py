
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch

import numpy as np
import numpy.random as npr
import pdb


class roibatchLoader(data.Dataset):
  def __init__(self, roidb, batch_size, num_classes, training=True):
    self._roidb = roidb
    self._num_classes = num_classes
    self.training = training
    self.batch_size = batch_size

  def __getitem__(self, index_tuple):
    index = index_tuple

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    blobs['data'] = blobs['data'].squeeze(axis=0)

    return blobs

  def __len__(self):
    return len(self._roidb)



class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_index):
        self.ratio_index = ratio_index
        self.num_data = len(ratio_index)

    def __iter__(self):
        rand_perm = npr.permutation(self.num_data)
        ratio_index = self.ratio_index[rand_perm]
        # re-calculate minibatch ratio list

        return iter(ratio_index.tolist())

    def __len__(self):
        return self.num_data


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

def collate_minibatch_for(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    lists = []
    for blobs in list_of_blobs:
        lists.append({'data' : blobs.pop('data'),
                      'rois' : blobs.pop('rois'),
                      'labels' : blobs.pop('labels')})
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = lists[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        minibatch = default_collate(mini_list)
        for key in minibatch:
            if i == 0:
                Batch[key] = minibatch[key]
            else:
                Batch[key] = torch.cat((Batch[key], minibatch[key]), dim=0)

    return Batch

def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    lists = []
    for blobs in list_of_blobs:
        data = blobs.pop('data')
        data_shape = list(data.shape[1:])
        if np.max(data_shape) > 1200:
            print('*'* 50)
            print('Error: max width or hight is larger than 1200!')
            data = data[:, :1200, :1200]
        data = np.pad(data, ((0, 0), (0, 1200-data.shape[1]), (0, 1200-data.shape[2])),
                      'constant', constant_values=0)

        rois = blobs.pop('rois')
        data_shape.append(rois.shape[0])
        if np.max(rois.shape) > 4000:
            print('*'* 50)
            print('Error: max rois is larger than 4000!')
            rois = rois[:4000]
        rois = np.pad(rois, ((0, 4000-rois.shape[0]), (0, 0)), 'constant', constant_values=0)

        '''For D-MIL
        seg_map = blobs.pop('seg_map')
        assert seg_map.shape[0] == data_shape[0]
        assert seg_map.shape[1] == data_shape[1]
        if np.max(seg_map.shape) > 1200:
            print('*'* 50)
            print('Error: max width or hight is larger than 1200!')
            seg_map = seg_map[:1200, :1200]
        seg_map = np.pad(seg_map, ((0, 1200-seg_map.shape[0]), (0, 1200-seg_map.shape[1])),
                         'constant', constant_values=0)
        '''
        
        data_shape = np.array(data_shape)

        lists.append({'data' : data,
                      'data_shape' : data_shape,
                      'rois' : rois,
                      'labels' : blobs.pop('labels')})
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = lists[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        minibatch = default_collate(mini_list)
        for key in minibatch:
            if i == 0:
                Batch[key] = minibatch[key]
            else:
                Batch[key] = torch.cat((Batch[key], minibatch[key]), dim=0)

    return Batch