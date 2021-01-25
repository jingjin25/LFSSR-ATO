# LFSSR-ATO
PyTorch implementation of **CVPR 2020** paper: "**Light Field Spatial Super-resolution via Deep Combinatorial Geometry Embedding and Structural Consistency Regularization**". 

[[arXiv]](https://arxiv.org/pdf/2004.02215.pdf)

## Requirements
- Python 3.6
- PyTorch 1.1
- Matlab (for training/test data generation)

## Dataset
We provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in `LFData`.

## Demo
To reproduce final SR reconstruction results in the paper, run:

```
python demo_LFSSR.py --model_dir pretrained_models --save_dir results --scale 2 --test_dataset Kalantari --angular_num 7 --save_img 1 --crop 1 --feature_num 64 --layer_num 5 2 2 3 --layer_num_refine 3
```

To reproduce intermediate all-to-one model results in the paper, run:
```
python demo_ATO.py --model_dir pretrained_models --save_dir results --scale 2 --test_dataset Kalantari --angular_num 7 --save_img 1 --crop 0 --feature_num 64 --layer_num 5 2 2 3
```

## Training
To train the all-to-one model, run:
```
python train_ATO.py --dataset all --scale 2  --angular_num 7 --feature_num 64 --layer_num 5 2 2 3 --lr 1e-4
python train_ATO.py --dataset all --scale 4  --angular_num 7 --feature_num 64 --layer_num 5 2 2 3 --lr 1e-5
```

To train the final SR model, run:
```
python train_LFSSR.py --dataset all --ATO_path pretrained_models/ATONet_2x.pth  --scale 2  --angular_num 7 --feature_num 64 --layer_num 5 2 2 3 --layer_num_refine 3 --weight_epi 0.1  --lr 1e-4
```
