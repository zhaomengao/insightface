#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u ./recognition/train_parall.py \
    --dataset retina \
    --network r100 \
    --loss arcface \
    --per-batch-size 64 \
    --models-root "hdfs://hobot-bigdata-aliyun/user/mengao.zhao/insightface"

