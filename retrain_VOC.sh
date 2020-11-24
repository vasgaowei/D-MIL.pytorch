#!/bin/bash

prefix=$1
GPU_ID=$2
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -u ./tools/retrain_VOC.py \
  --dataset pascal_voc_2007 \
  --imdb voc_2007_trainval \
  --net vgg16 \
  --checksession 1 \
  --checkepoch 18 \
  --checkpoint 5010 \
  --cuda \
  --load_dir output/${prefix}

python -u ./tools/retrain_VOC.py \
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