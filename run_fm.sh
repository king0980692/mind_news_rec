set -xe

train_beh=data/MIND-200k/train/behaviors.tsv
test_beh=data/MIND-200k/dev/behaviors.tsv

#test_beh=raw_data/test/behaviors.tsv

#python3 tools/to_sparse.py --train $train_beh
#python3 tools/gen_tfidf.py --dataset 200k

#python3 tools/to_fm.py --train $train_beh --test $test_beh --dataset 200k

#python3 tools/to_fm.py --train raw_data/train/behaviors.tsv  --test raw_data/test/behaviors.tsv 

#python3 tools/split_fm.py

#xlearn_train exp/split_train.fm -v exp/split_valid.fm -s 0 -x f1 -m exp/ui.model -k 64 -e 300 -sw 5 

xlearn_train exp/train.fm -s 1 -e 20 -m exp/ui.model 

xlearn_predict ./exp/dev.fm ./exp/ui.model -o exp/ui.pred 

python3 tools/fm_pred2.py --test_beh $test_beh --pred ./exp/ui.pred --out ./exp/prediction.txt


python3 tools/evaluate.py --pred exp/prediction.txt --truth ./truth.txt

