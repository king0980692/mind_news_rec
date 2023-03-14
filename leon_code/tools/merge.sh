set -x
python3 ./leon_code/tools/merge_fm.py \
    --beh_file exp/split_data/train/behaviors.tsv \
    --news_map exp/news_map.pkl       \
    --out  exp/train.fm \
    --negative_num 30 \
    --pkl_list exp/user_encode.pkl    \
               exp/news_encode.pkl    \
               exp/cat_encode.pkl     \
               exp/sub_cat_encode.pkl \
               exp/user_tfidf.pkl     \
               exp/news_tfidf.pkl     &

python3 ./leon_code/tools/merge_fm.py \
    --beh_file data/dev/behaviors.tsv \
    --news_map exp/news_map.pkl       \
    --out  exp/test.fm                \
    --pkl_list exp/user_encode.pkl    \
               exp/news_encode.pkl    \
               exp/cat_encode.pkl     \
               exp/sub_cat_encode.pkl \
               exp/user_tfidf.pkl     \
               exp/news_tfidf.pkl     &

wait
