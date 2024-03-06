#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire

EXP_NAME="lsun_adm_release"
CKPT="/home/lizg/lizg/DIRE-main/checkpoints/lsun_adm.pth"
DATASETS_TEST="/home/lizg/lizg/DIRE-main/data/test/lsun_adm/"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST