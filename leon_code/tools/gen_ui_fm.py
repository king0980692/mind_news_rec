import os
import pickle
from tqdm import tqdm
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--out')
args = parser.parse_args()

user_encode = {}
news_encode = {}
user_decode = {}
news_decode = {}

for data_type in Path(args.data).iterdir():
    beh_file = data_type / "behaviors.tsv"
    print("UI - Encode: ", beh_file)
    with open(beh_file) as f:
        for line in tqdm(f.readlines()):
            imp_id, u_id, imp_time, history, imprs = line.split("\t")

            if u_id not in user_encode:
                user_encode[u_id] = len(user_encode)
                user_decode[len(user_encode)-1] = u_id

            for n_id in history.split():
                if n_id not in news_encode:
                    news_encode[n_id] = len(news_encode)
                    news_decode[len(news_encode)-1] = n_id

            for impr in imprs.split():
                n_id = impr[:-2]
                # print(nid)
                if n_id not in news_encode:
                    news_encode[n_id] = len(news_encode)
                    news_decode[len(news_encode)-1] = n_id


user_encode['cold_user'] = len(user_encode) 
user_decode[len(user_encode)-1] = 'cold_user'

news_encode['cold_news'] = len(news_encode) 
news_decode[len(news_encode)-1] = 'cold_news'


with open(os.path.join(args.out, "user_encode.pkl"), 'wb') as p:
    pickle.dump(user_encode, p)

with open(os.path.join(args.out, "user_decode.pkl"), 'wb') as p:
    pickle.dump(user_decode, p)

with open(os.path.join(args.out, "news_encode.pkl"), 'wb') as p:
    pickle.dump(news_encode, p)

with open(os.path.join(args.out, "news_decode.pkl"), 'wb') as p:
    pickle.dump(news_decode, p)

with open(os.path.join(args.out, "user_encode.json"), 'w') as p:
    p.write(str(len(user_encode))+"\n")
    p.write(json.dumps(user_encode))

with open(os.path.join(args.out, "news_encode.json"), 'w') as p:
    p.write(str(len(news_encode))+"\n")
    p.write(json.dumps(news_encode))
