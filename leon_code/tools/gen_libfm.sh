set -xe

mkdir -p exp/
mkdir -p exp/split_data
mkdir -p exp/split_data/train
mkdir -p exp/split_data/test
mkdir -p exp/split_data/valid

MERGE_BEH_FILE=exp/split_data/all_behaviors.tsv
if [ ! -f "$MERGE_BEH_FILE" ]; then
    cat data/train/behaviors.tsv data/dev/behaviors.tsv > exp/split_data/all_behaviors.tsv

fi

#python3 leon_code/tools/split_data.py --file exp/split_data/all_behaviors.tsv -d '\t' -f 2 --out exp/split_data/

## Feat-1. basic UI interaction
 #python3 ./leon_code/tools/gen_ui_fm.py --data data --out exp

## Feat-2. news_meta data (cat, subcat, news_map)
#python3  leon_code/tools/gen_i2c.py --data data --out exp


## Feat-3. news_meta data
##python3  leon_code/tools/gen_tfidf.py --data data --out exp
#python3  leon_code/tools/gen_sbert.py --data data --out exp

#total=$(wc -l < exp/split_data/train/behaviors.tsv)
#echo $total
#split_to=16
#part=$(($total/$split_to+1))
#echo $part
#c=$(($part*$split_to))
#echo $c

#split -a 1 -l $part  --verbose exp/split_data/train/behaviors.tsv exp/split_data/train/beh_

#mkdir -p exp/tmp/
#i=0 # var. for counting iterations
#for x in {a..z};
#do
    #python3 ./leon_code/tools/merge_fm.py \
        #--beh_file exp/split_data/train/beh_$x \
        #--news_map exp/news_map.pkl       \
        #--out  exp/split_data/out_$x.fm \
        #--negative_num 20 \
        #--pkl_list exp/user_encode.pkl    \
                   #exp/news_encode.pkl    \
                   #exp/cat_encode.pkl     \
                   #exp/sub_cat_encode.pkl \
                   #exp/news_tfidf.pkl     &
      ##(( ++i == 2 )) && break 
#done
#wait
#array=({a..z})
#echo "${array[@]}"|xargs -n4                               

cat exp/split_data/out_*.fm > exp/split_data/train.fm
#rm exp/split_data/out_*.fm

python3 ./leon_code/tools/merge_fm.py \
    --beh_file data/dev/behaviors.tsv \
    --news_map exp/news_map.pkl       \
    --out  exp/test.fm \
    --pkl_list exp/user_encode.pkl    \
               exp/news_encode.pkl    \
               exp/cat_encode.pkl     \
               exp/sub_cat_encode.pkl \
               exp/news_tfidf.pkl     &


python3 ./leon_code/tools/merge_fm.py \
    --beh_file exp/split_data/train/behaviors.tsv \
    --news_map exp/news_map.pkl       \
    --out  exp/train.fm \
    --negative_num 30 \
    --pkl_list exp/user_encode.pkl    \
               exp/news_encode.pkl    \
               exp/cat_encode.pkl     \
               exp/sub_cat_encode.pkl \
               exp/news_tfidf.pkl     &
wait
