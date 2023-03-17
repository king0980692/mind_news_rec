#!/bin/bash
set -xe


model=$1
data=$2
GPU=$3
tr_data=exp/$data.train
te_data=exp/$data.test
emb=exp/$model.$data.emb

DIM=64
UPDATE=300
WORKER=32
NEG=5
BATCH=2048

# Train Model
#python3 -c 'from pysmore.train import entry_points;entry_points()' \
#export CUDA_VISIBLE_DEVICES=1
pysmore_train \
    --dataset ui \
    --dim $DIM \
    --fetch_worker $WORKER \
    --batch_size $BATCH \
    --data_dir $tr_data \
    --saved_emb $emb \
    --gpu $GPU \
    --update_times $UPDATE 

./tools/rec.sh $model $data $DIM

./tools/eval.sh $model $data

