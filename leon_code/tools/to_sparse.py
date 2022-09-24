import pickle
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train')
args = parser.parse_args()

user_encode = {}
news_encode = {}
user_decode = {}
news_decode = {}

with open(args.train) as f:
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
            n_id = impr.split('-')[0]
            # print(nid)
            if n_id not in news_encode:
                news_encode[n_id] = len(news_encode)
                news_decode[len(news_encode)-1] = n_id


user_encode['cold_user'] = len(user_encode) + len(news_encode)
user_decode[len(user_encode)+len(news_encode)-1] = 'cold_user'

news_encode['cold_news'] = len(user_encode) + len(news_encode)-2
news_decode[len(user_encode)+len(news_encode)-3] = 'cold_news'

# print(user_decode)
# print(news_decode)
# print(user_encode)
# print(news_encode)

# exit()

with open("exp/user_encode.pkl", 'wb') as p:
    pickle.dump(user_encode, p)
with open("exp/user_decode.pkl", 'wb') as p:
    pickle.dump(user_decode, p)
with open("exp/news_encode.pkl", 'wb') as p:
    pickle.dump(news_encode, p)
with open("exp/news_decode.pkl", 'wb') as p:
    pickle.dump(news_decode, p)

