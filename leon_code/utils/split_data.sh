
mkdir -p exp/
mkdir -p exp/split_data
mkdir -p exp/split_data/train
mkdir -p exp/split_data/test
mkdir -p exp/split_data/valid

MERGE_BEH_FILE=exp/split_data/all_behaviors.tsv
if [ ! -f "$MERGE_BEH_FILE" ]; 
then
    echo "Merge train&dev behaviors.tsv into all_behaviors.tsv"
    cat data/train/behaviors.tsv data/dev/behaviors.tsv > exp/split_data/all_behaviors.tsv
else
    echo "split_data/all_behaviors.tsv has existed yey !"
fi


python3 leon_code/tools/split_data.py --file exp/split_data/all_behaviors.tsv -d '\t' -f 2 --out exp/split_data/
