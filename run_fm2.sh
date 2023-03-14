set -xe

xlearn_train ./200k_libfm/train-200k.libfm -v ./200k_libfm/dev-200k.libfm -s 1 -m ./exp/200k_fm.model -x f1 -e 300

xlearn_predict ./200k_libfm/test-200k.libfm ./exp/200k_fm.model -o exp/200k.pred

python3 tools/fm_pred2.py --test_beh ../MIND-200k/test/behaviors.tsv --pred exp/200k.pred --out exp/prediction.txt


python3 tools/evaluate.py --pred exp/prediction.txt --truth ./200k_truth.txt

