#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u gen_image_feature.py \
    --batch_size 32 \
    --image_size '3,112,112' \
    --input '/home/users/mengao.zhao/mengao.zhao/competition/LWFR/test_data/iccv19-challenge-data'
    --output ''
    --model "hdfs://hobot-bigdata-aliyun/user/mengao.zhao/mvppt/hobotface/mvppt_hobotface_115_ms1m_Normal/model,150"
