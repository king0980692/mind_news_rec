set -xe

# Feat-1. basic UI interaction
 #python3 ./leon_code/tools/gen_ui_fm.py --data data --out exp

# Feat-2. news_meta data (cat, subcat, news_map)
#python3  leon_code/tools/gen_i2c.py --data data --out exp

python3  leon_code/tools/gen_sbert.py --data data --out exp
exit

# Feat-3. news_meta data
python3  leon_code/tools/gen_tfidf.py --data data --out exp
exit

python3 ./leon_code/tools/merge_fm.py \
    --beh_file data/dev/behaviors.tsv \
    --news_map exp/news_map.pkl       \
    --out  exp/test.fm \
    --pkl_list exp/user_encode.pkl    \
               exp/user_decode.pkl    \
               exp/news_encode.pkl    \
               exp/cat_encode.pkl     \
               exp/sub_cat_encode.pkl \
               exp/news_tfidf.pkl     \
               exp/sub_cat_encode.pkl \
               exp/news_decode.pkl   
exit

python3 ./leon_code/tools/merge_fm.py \
    --beh_file data/train/behaviors.tsv \
    --news_map exp/news_map.pkl       \
    --out  exp/train.fm \
    --negative_num 30 \
    --pkl_list exp/user_encode.pkl    \
               exp/user_decode.pkl    \
               exp/news_encode.pkl    \
               exp/cat_encode.pkl     \
               exp/sub_cat_encode.pkl \
               exp/news_tfidf.pkl     \
               exp/news_decode.pkl    &

wait


