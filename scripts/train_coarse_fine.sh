#!/bin/bash

cd coarse_fine/
python -u src/train_coarse_fine.py \
        --max_iters=100 \
        --batch_size=1 \
        --nworkers=1 \
        --lr=5e-5 \
        --do_val=True \
        --use_scheduler=True \
        --dataset_dir='../Dur360BEV_dataset/data/Dur360BEV_Dataset' \
        --log_freq=1 \
        --img_freq=1 \