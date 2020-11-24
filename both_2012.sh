#!/bin/bash

prefix=$1
export CUDA_VISIBLE_DEVICES=0
python -u ./tools/trainval_net.py \
  --dataset pascal_voc_2012 \
  --net vgg16 \
  --bs 1 \
  --lr 0.0005 \
  --lr_decay_step 10 \
  --epochs 18 \
  --cuda \
  --o sgd \
  --itersize 2 \
  --save_dir output/${prefix}

python -u ./tools/test_net.py \
  --dataset pascal_voc_2012 \
  --imdb voc_2012_test \
  --net vgg16 \
  --checksession 1 \
  --checkepoch 18 \
  --checkpoint 11539 \
  --cuda \
  --load_dir output/${prefix}

  # --r \
  # --checksession 1 \
  # --checkepoch 7 \
  # --checkpoint 5010 \
