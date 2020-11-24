#!/bin/bash
if [ ! -d "log" ]; then
  mkdir log
fi

prefix=$1
log=log/${prefix}_2012_trainval.log
export CUDA_VISIBLE_DEVICES=0
nohup python -u ./tools/test_net.py \
  --dataset pascal_voc_2012 \
  --imdb voc_2012_trainval \
  --net vgg16 \
  --checksession 1 \
  --checkepoch 15 \
  --checkpoint 11539 \
  --cuda \
  --load_dir output/${prefix} \
  > ${log} 2>&1 &
