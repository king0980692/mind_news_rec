#!/bin/bash
set -ex
model=$1
data=$2
emb_dim=$3
tr_data=exp/$data.train
te_data=exp/$data.test
emb=exp/$model.$data.emb

python3 tools/emb_rec.py \
    --train $tr_data \
    --test  $te_data\
    --embed $emb \
    --emb_dim $emb_dim \
    --worker 8
