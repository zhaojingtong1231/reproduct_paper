#!/usr/bin/env sh


echo "=====Cora====="
python train_revision.py  --lr=0.001 --weight-decay=1e-4 --zhongzi=0 --epochs=1000 --num-heads=4 --num-layers=1 --num-hidden=32 --tau=1 --seed=1 --in-drop=0.6 --attn-drop=0.5 --negative-slope=0.2 --gpu=0


nohup python train_RGCN2_lr_de.py  --lr=0.001 --weight-decay=1e-4 --zhongzi=0 --epochs=1000 --num-heads=8 --num-layers=1 --num-hidden=128 --loss-rate-mlp=0.8 --loss-rate-multi=0.5 --
