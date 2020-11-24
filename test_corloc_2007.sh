
prefix=$1
GPU_ID=$2
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python -u ./tools/test_net_corloc.py \
  --dataset pascal_voc_2007 \
  --imdb voc_2007_trainval \
  --net vgg16 \
  --checksession 1 \
  --checkepoch 18 \
  --checkpoint 5010 \
  --cuda \
  --load_dir output/${prefix}