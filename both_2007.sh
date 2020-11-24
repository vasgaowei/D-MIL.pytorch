#!/bin/bash

prefix=$1
GPU_ID=$2
export CUDA_VISIBLE_DEVICES=${GPU_ID}
python -u ./tools/trainval_net.py \
  --dataset pascal_voc_2007 \
  --net vgg16 \
  --bs 1 \
  --lr 0.0005 \
  --lr_decay_step 10 \
  --epochs 18 \
  --cuda \
  --o sgd \
  --itersize 2 \
  --save_dir output/${prefix} \
  --use_tfb

python -u ./tools/test_net.py \
  --dataset pascal_voc_2007 \
  --imdb voc_2007_test \
  --net vgg16 \
  --checksession 1 \
  --checkepoch 18 \
  --checkpoint 5010 \
  --cuda \
  --load_dir output/${prefix}


#  --r \
#  --checksession 1 \
#  --checkepoch 9 \
#  --checkpoint 5010 \
