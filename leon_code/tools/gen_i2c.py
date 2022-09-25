import os
from tqdm import tqdm
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--out')
args = parser.parse_args()

# news_map = defaultdict(lambda:defaultdict(str))
news_map = defaultdict(dict)


cat_encode = {}
sub_cat_encode = {}
cat_decode = {}
sub_cat_decode = {}

for data_type in Path(args.data_path).iterdir():
    news_file = data_type / "news.tsv"
    print("News meta - Encode: ", news_file)
    with open(news_file) as f:
        for line in tqdm(f.readlines()):
            n_id, cat, sub_cat, title, abstr, *_ = line.split('\t')

            if cat not in cat_encode:
                cat_encode[cat] = len(cat_encode)
                cat_decode[len(cat_encode)-1] = cat

            if sub_cat not in sub_cat_encode:
                sub_cat_encode[cat] = len(sub_cat_encode)
                sub_cat_decode[len(sub_cat_encode)-1] = sub_cat

            news_map[n_id]['cat'] = cat 
            news_map[n_id]['sub_cat'] = sub_cat 
            news_map[n_id]['title'] = title 
            news_map[n_id]['abstr'] = abstr 

news_map['cold_news']['cat'] = "cold_cat" 
news_map['cold_news']['sub_cat'] = "cold_sub_cat"  
news_map['cold_news']['title'] = "cold_title" 
news_map['cold_news']['abstr'] = "cold_abstr"  

cat_encode['cold_cat'] = len(cat_encode) 
cat_decode[len(cat_encode)-1] = 'cold_cat'

sub_cat_encode['cold_sub_cat'] = len(sub_cat_encode) 
sub_cat_decode[len(cat_encode)-1] = 'cold_sub_cat'


with open(os.path.join(args.out, "cat_encode.pkl"), 'wb') as p:
    pickle.dump(cat_encode, p)

with open(os.path.join(args.out, "sub_cat_encode.pkl"),'wb') as p:
    pickle.dump(sub_cat_encode, p)

with open(os.path.join(args.out, "news_map.pkl"),'wb') as p:
    pickle.dump(news_map, p)
    
    
