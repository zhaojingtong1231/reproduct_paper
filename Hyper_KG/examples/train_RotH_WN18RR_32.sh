#!/bin/bash
cd .. 
source set_env.sh
python run.py \
            --dataset PharmKG \
            --model AttH \
            --rank 256 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adagrad \
            --max_epochs 200 \
            --patience 20 \
            --valid 5 \
            --batch_size 256 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.01 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --double_neg \
            --multi_c 
cd examples/
