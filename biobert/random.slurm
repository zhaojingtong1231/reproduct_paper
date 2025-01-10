#!/bin/bash
SBATCH --job-name='random_1_2_3_4_5'
SBATCH --chdir=/datapool/home/info_wang/zhaojingtong/biobert
SBATCH --partition=info-wang
SBATCH --nodes=1
SBATCH --ntasks-per-node=8
SBATCH --time=1-00:00

SBATCH --mem=20000M
SBATCH --gres=gpu:1
echo job start time is `date`
echo `hostname`
PATH=/datapool/home/info_wang/cuda/cuda11.8/bin/:$PATH
LD_LIBRARY_PATH=/datapool/home/info_wang/cuda/cuda11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
conda activate zjt_pytorch
python biobert.py --seed=12 --epochs=20 --split random
python biobert.py --seed=22 --epochs=20 --split random
python biobert.py --seed=32 --epochs=20 --split random
python biobert.py --seed=42 --epochs=20 --split random
python biobert.py --seed=52 --epochs=20 --split random
echo job end time is `date`
