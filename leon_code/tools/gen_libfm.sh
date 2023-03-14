set -xe

## Feat-1. basic UI interaction
python3 ./leon_code/tools/gen_ui_fm.py --data data --out exp &

## Feat-2. news_meta data (cat, subcat, news_map)
python3  leon_code/tools/gen_i2c.py --data data --out exp &


## Feat-3. news_meta data
python3  leon_code/tools/gen_tfidf.py --data data --out exp &

#python3  leon_code/tools/gen_sbert.py --data data --out exp
wait

