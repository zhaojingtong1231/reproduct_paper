#!/bin/bash
cd .. 
source set_env.sh
python run.py \
            --dataset PharmKG \
            --model RotatE \
            --rank 256 \
            --regularizer N3 \
            --reg 0.1 \
            --optimizer Adagrad \
            --max_epochs 200 \
            --patience 15 \
            --valid 1 \
            --batch_size 256 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none \
            --dtype single 
cd examples/
