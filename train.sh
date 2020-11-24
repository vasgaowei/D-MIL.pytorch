#!/bin/bash

prefix=$1
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u ./tools/trainval_net.py \
  --dataset coco \
  --net vgg16 \
  --bs 4 \
  --lr 0.0001 \
  --lr_decay_step 3 \
  --epochs 5 \
  --cuda \
  --o sgd \
  --save_dir output/${prefix} \
  --nw 4 \
  --itersize 1 \
  | tee log/${prefix}_train.log


#!/bin/bash
if [ ! -d "log" ]; then
  mkdir log
fi

prefix=$1
for i in {0..3}
do
log=log/${prefix}_2007_test_${i}.log
export CUDA_VISIBLE_DEVICES=${i}
nohup python -u ./tools/test_net.py \
  --dataset coco \
  --imdb coco_2014_val \
  --net vgg16 \
  --checksession 1 \
  --checkepoch 5 \
  --checkpoint 41039 \
  --cuda \
  --load_dir output/${prefix} \
  --num_worker 4 \
  --worker_id ${i} \
  > ${log} 2>&1 &
done

#  --r \
#  --checksession 1 \
#  --checkepoch 9 \
#  --checkpoint 5010 \
