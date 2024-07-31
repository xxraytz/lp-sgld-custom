#!/bin/bash

WL=8
python -u train.py --dataset IMAGENET1K --data_path "/home/dataset_caches/imagenet-1k" \
       --model ResNet18LP \
       --epochs=100 \
       --lr_init=0.5 \
       --wd=5e-4 \
       --wl-weight ${WL} \
       --wl-grad ${WL} \
       --fl-weight ${WL} \
       --fl-grad ${WL} \
       --seed 1 \
       --batch_size 128  \
       --weight-rounding stochastic \
       --grad-rounding stochastic \
       --weight-type fixed \
       --grad-type fixed\
       --noise 1 \
       --quant_acc -2 \
       --set_default_optimizer True \
       ;
