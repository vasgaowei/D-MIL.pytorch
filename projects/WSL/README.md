# D-MIL


This project is based on [Detectron2](https://github.com/facebookresearch/detectron2) and [DRN](https://github.com/shenyunhang/DRN-WSOD-pytorch/tree/DRN-WSOD/projects/WSL)

## License

DRN-WSOD is released under the [Apache 2.0 license](LICENSE).
# Performances
  1). On PASCAL VOC 2007 dataset 
  model    | #GPUs | batch size | lr        | lr_decay | max_iterations     |  mAP 
---------|--------|-----|--------|-----|--------|-----
ResNet50+PCL     | 4 | 4 | 1e-2 | 35000   | 60000   | 50.8
ResNet50+DMIL     | 4 | 4 | 1e-2 | 35000   | 60000   | 55.1 

## Installation

Install our forked Detectron2:
First download the codes of detectron_wsod branch, then 
```
cd detectron_wsod
python3 -m pip install -e .
```
If you have problem of installing Detectron2, please checking [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Install DRN-WSOD project:
```
cd projects/WSL
pip3 install -r requirements.txt
git submodule update --init --recursive
python3 -m pip install -e .
cd ../../
```

## Dataset Preparation
Please follow [this](https://github.com/shenyunhang/DRN-WSOD-pytorch/blob/DRN-WSOD/datasets/README.md#expected-dataset-structure-for-pascal-voc) to creating symlinks for PASCAL VOC.

Download MCG proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:

```
cd datasets/proposals
tar xvzf MCG-Pascal-Main_trainvaltest_2007-boxes.tgz
cd ../../
python3 projects/WSL/tools/proposal_convert.py voc_2007_train datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_train_d2.pkl
python3 projects/WSL/tools/proposal_convert.py voc_2007_val datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_val_d2.pkl
python3 projects/WSL/tools/proposal_convert.py voc_2007_test datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/mcg_voc_2007_test_d2.pkl
```


## Model Preparation

Download models from this [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2):
```
mv models $DRN-WSOD
```

Then we have the following directory structure:
```
DRN-WSOD
|_ models
|  |_ DRN-WSOD
|     |_ resnet18_ws_model_120.pkl
|     |_ resnet150_ws_model_120.pkl
|     |_ resnet101_ws_model_120.pkl
|_ ...
```


## Quick Start: Using DRN-WSOD

### WSDDN

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_WSR_18_DC5_1x.yaml OUTPUT_DIR output/wsddn_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_WSR_50_DC5_1x.yaml OUTPUT_DIR output/wsddn_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_WSR_101_DC5_1x.yaml OUTPUT_DIR output/wsddn_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/wsddn_V_16_DC5_1x.yaml OUTPUT_DIR output/wsddn_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### OICR

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_WSR_18_DC5_1x.yaml OUTPUT_DIR output/oicr_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_WSR_50_DC5_1x.yaml OUTPUT_DIR output/oicr_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_WSR_101_DC5_1x.yaml OUTPUT_DIR output/oicr_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/oicr_V_16_DC5_1x.yaml OUTPUT_DIR output/oicr_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### OICR + Box Regression

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_WSR_18_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_WSR_50_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_WSR_101_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/reg/oicr_V_16_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### PCL

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_WSR_18_DC5_1x.yaml OUTPUT_DIR output/pcl_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_WSR_50_DC5_1x.yaml OUTPUT_DIR output/pcl_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_WSR_101_DC5_1x.yaml OUTPUT_DIR output/pcl_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/pcl_V_16_DC5_1x.yaml OUTPUT_DIR output/pcl_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### DMIL

#### ResNet18-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/dmil_WSR_18_DC5_1x.yaml OUTPUT_DIR output/dmil_WSR_18_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/dmil_WSR_50_DC5_1x.yaml OUTPUT_DIR output/dmil_WSR_50_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/dmil_WSR_101_DC5_1x.yaml OUTPUT_DIR output/dmil_WSR_101_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### VGG16
```
python3 projects/WSL/tools/train_net.py --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-Detection/dmil_V_16_DC5_1x.yaml OUTPUT_DIR output/dmil_V_16_DC5_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```


