#!/bin/bash

cd coarse_fine/
python -u src/eval_coarse_fine.py \
        --batch_size=4 \
        --nworkers=4 \
        --checkpoint_dir='../checkpoints/coarse_fine/FL2_4000_6x4_5e-05s_09-11_14:29/checkpoint_epoch_4000.pth' \
        --dataset_dir='../Dur360BEV_dataset/data/Dur360BEV_Dataset' \
        --log_freq=1 \
        --img_freq=10 \
