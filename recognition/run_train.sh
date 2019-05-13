#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py \
    --dataset retina \
    --network r100 \
    --loss arcface \
    --per-batch-size 8

