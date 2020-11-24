#!/bin/bash

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


#prefix=$1
#export CUDA_VISIBLE_DEVICES=7
#python -u ./tools/test_net.py \
#  --dataset pascal_voc_2007 \
#  --imdb voc_2007_test \
#  --net vgg16 \
#  --checksession 1 \
#  --checkepoch 15 \
#  --checkpoint 5010 \
#  --cuda \
#  --load_dir output/${prefix} \
#  | tee log/${prefix}_test.log
