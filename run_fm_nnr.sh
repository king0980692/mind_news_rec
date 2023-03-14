set -xe

train_beh=raw_data/train/behaviors.tsv

# using valid data as test file
test_beh=raw_data/valid/behaviors.tsv

# using test data as test file
#test_beh=raw_data/test/behaviors.tsv

#python3 tools/generate_tf_idf_feature_file.py --dataset large

# creat train,valid,test-200k.libfm
python3 tools/generate_libfm_data.py  --dataset large

exit
xlearn_train exp/train-200k.libfm  -s 0 -x f1 -m exp/ui.model -k 64 -e 300 -sw 5 

###xlearn_train exp/train.fm -s 4 -e 200 -m exp/ui.model 

xlearn_predict ./exp/test.fm ./exp/ui.model -o exp/ui.pred 

python3 tools/fm_pred2.py --test_beh $test_beh --pred ./exp/ui.pred --out ./exp/prediction.txt

python3 tools/evaluate.py --pred exp/prediction.txt --truth ./200k_truth.txt

#python3 tools/evaluate.py --pred exp/prediction.txt --truth ./truth.txt
