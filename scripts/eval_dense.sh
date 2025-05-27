#!/bin/bash

cd dense/
python3 -u eval_dense.py \
        --batch_size=4 \
        --nworkers=4 \
        --checkpoint_dir='../checkpoints/dense/FL1_4000_6x6_5e-05s_07-03_09:42/checkpoint_epoch_4000.pth' \
        --dataset_dir='../Dur360BEV_dataset/data/Dur360BEV_Dataset' \
        --log_freq=1 \
        --img_freq=1