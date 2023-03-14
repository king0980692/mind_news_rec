#!/bin/bash
#set -xe
#if [[ ! -n $CONDA_PREFIX ]]; then
    #eval "$(conda shell.bash hook)"
    #conda activate recommender
    ##python3 -c 'from sentence_transformers import SentenceTransformer'
#fi
bash ./leon_code/tools/gen_libfm.sh

###          |####
# ClS        | REG
#  -s 0   LM  |  -s 3  LM
#  -s 1   FM  |  -s 4  FM
#  -s 2  FFM  |  -s 5  FFM

xlearn_train exp/train.fm -s 1 -e 40 -k 8 -m exp/fm.model --disk

xlearn_predict ./exp/test.fm ./exp/fm.model -o exp/fm.pred  --disk

python3 tools/fm_pred2.py --test_beh data/dev/behaviors.tsv --pred ./exp/fm.pred --out ./exp/prediction.txt

python3 tools/evaluate.py --pred exp/prediction.txt --truth ./truth.txt

# gen test fm format
#bash leon_code/tools/gen_test_libfm.sh

#xlearn_predict ./exp/test2.fm ./exp/fm.model -o exp/fm2.pred  --disk

#python3 tools/fm_pred2.py --test_beh data/test/behaviors.tsv --pred ./exp/fm2.pred --out ./exp/prediction2.txt
