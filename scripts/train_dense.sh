#!/bin/bash

cd dense/
python3 -u train_dense.py \
        --max_iters=100 \
        --batch_size=1 \
        --nworkers=1 \
        --lr=5e-5 \
        --gamma=1 \
        --weight_decay=1e-7 \
        --load_ckpt_dir=None \
        --use_scheduler=True \
        --dataset_dir='../Dur360BEV_dataset/data/Dur360BEV_Dataset' \
        --log_freq=10 \
        --img_freq=10 \
        --do_val=True