build target:
    @echo 'Building {{target}}…'
    just {{target}}

asd:
    python3  leon_code/tools/gen_tfidf.py --data data --out exp 
  
tfm:
    python3 leon_code/fm_torch/train.py --dataset_path ./data/train/behaviors.tsv 
    python3 tools/fm_pred2.py --test_beh data/dev/behaviors.tsv --pred ./exp/tfm.pred --out ./exp/prediction.txt
    python3 tools/evaluate.py --pred exp/prediction.txt --truth ./truth.txt

split:
    bash ./leon_code/utils/split_data.sh

gen_fm:
    bash ./leon_code/tools/gen_libfm.sh

merge:
    bash ./leon_code/tools/merge.sh

pre: split gen_fm merge
    echo "All Training Data are well prepared !!"

train:
    xlearn_train exp/train.fm -s 1 -e 40 -k 8 -m exp/fm.model  -block 20480 --disk

    xlearn_predict ./exp/test.fm ./exp/fm.model -o exp/fm.pred -block 20480 --disk

    python3 tools/fm_pred2.py --test_beh data/dev/behaviors.tsv --pred ./exp/fm.pred --out ./exp/prediction.txt
    python3 tools/evaluate.py --pred exp/prediction.txt --truth ./truth.txt

eval:


all: pre train

