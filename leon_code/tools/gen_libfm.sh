set -xe

#python3 ./leon_code/tools/gen_ui_fm.py --train data/train/behaviors.tsv --out exp

python3 ./leon_code/tools/merge_fm.py \
    --beh_file data/train/behaviors.tsv \
    --out  exp/train.fm \
    --negative_num 20 \
    --pkl_list exp/user_encode.pkl \
               exp/user_decode.pkl \
               exp/news_encode.pkl \
               exp/news_decode.pkl &

python3 ./leon_code/tools/merge_fm.py \
    --beh_file data/dev/behaviors.tsv \
    --out  exp/test.fm \
    --pkl_list exp/user_encode.pkl \
               exp/user_decode.pkl \
               exp/news_encode.pkl \
               exp/news_decode.pkl &

wait


