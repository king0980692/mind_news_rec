from operator import neg
import os
import pickle
from nltk.classify.maxent import tempfile
from tqdm import tqdm
import argparse
import random
from itertools import accumulate
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_list', nargs='+')
parser.add_argument('--news_map', type=str)
parser.add_argument('--beh_file', type=str)
parser.add_argument('--negative_num', type=int, default=4)
parser.add_argument('--out')
args = parser.parse_args()


news_map = pickle.load(open(args.news_map,'rb'))
cat_encode = {}
sub_cat_encode = {}
user_encode = {}
news_encode = {}
news_tfidf = {}
user_tfidf = {}

# Load all pkl 
for pkl in args.pkl_list:
    var_name = os.path.split(pkl)[-1].split(".")[0]
    globals()[var_name] = pickle.load(open(pkl,'rb'))

offset_list = list(accumulate([
    news_tfidf['total_dim']*2, len(user_encode), len(news_encode), 
    len(cat_encode), len(sub_cat_encode)
    ]))

if "train" in args.beh_file:
    with open(args.beh_file) as f:

        out_f = open(args.out, 'w', encoding='utf-8')
        lines = f.readlines()
        sampled_lines = random.sample(lines, int(len(lines)*0.45))
        for line in tqdm(sampled_lines):
            imp_id, u_id, imp_time, history, imprs = line.split("\t")

            positive_samples = []
            negative_samples = []

            for impr in imprs.split():
                if impr[-1] == '1':
                    positive_samples.append(impr[:-2])
                else:
                    negative_samples.append(impr[:-2])


            positive_samples_num = len(positive_samples)
            negative_samples_num = len(negative_samples)
            '''
                keep the raito between pos:neg = 1:args.negative_num
                ,negative_samples_num < args.negative_num*positive_samples_num
            '''
            if positive_samples_num * args.negative_num > negative_samples_num:
                k = 0
                for i in range(positive_samples_num):
                    n_id = positive_samples[i]
                    cat = news_map[n_id]['cat']
                    sub_cat = news_map[n_id]['sub_cat']
                     
                    _tfidf = news_tfidf[n_id] if n_id in news_tfidf else ""

                    hist_tfidf = user_tfidf[u_id] if u_id in user_tfidf else ""


                    fea_list = [
                                idx+offset_list[i]
                                # idx + idx_list
                                for i, idx in enumerate(
                                [
                                    user_encode[u_id]
                                        if u_id in user_encode
                                        else user_encode['cold_user'],
                                    news_encode[n_id]
                                        if n_id in news_encode
                                        else news_encode['cold_news'],
                                    cat_encode[cat] 
                                        if cat in cat_encode
                                        else cat_encode['cold_cat'],
                                    sub_cat_encode[sub_cat] 
                                        if sub_cat in sub_cat_encode
                                        else sub_cat_encode['cold_sub_cat']
                                ])
                            ]
                    fea =  " ".join(["{}:1"]*len(fea_list)).format(*fea_list)
                    pos_out = " ".join(["1", _tfidf, hist_tfidf,fea])+"\n"

                    out_f.write(pos_out)
                    for j in range(args.negative_num):
                        n_k = negative_samples[k % negative_samples_num]

                        cat = news_map[n_k]['cat']
                        sub_cat = news_map[n_k]['sub_cat']
                        _tfidf = news_tfidf[n_k] if n_k in news_tfidf else ""

                        fea_list = [
                            idx+offset_list[i]
                            # idx + idx_list
                            for i, idx in enumerate(
                            [
                                user_encode[u_id]
                                    if u_id in user_encode
                                    else user_encode['cold_user'],
                                news_encode[n_id]
                                    if n_id in news_encode
                                    else news_encode['cold_news'],
                                cat_encode[cat] 
                                    if cat in cat_encode
                                    else cat_encode['cold_cat'],
                                sub_cat_encode[sub_cat] 
                                    if sub_cat in sub_cat_encode
                                    else sub_cat_encode['cold_sub_cat']
                            ])
                        ]
                        fea =  " ".join(["{}:1"]*len(fea_list)).format(*fea_list)
                        neg_out = " ".join(['0', _tfidf, hist_tfidf, fea])+"\n"

                        out_f.write(neg_out)
                        k += 1
            else:
                sample_index = random.sample(
                        range(negative_samples_num), 
                        positive_samples_num * args.negative_num
                    )

                k = 0
                for i in range(positive_samples_num):
                    n_id = positive_samples[i]
                    cat = news_map[n_id]['cat']
                    sub_cat = news_map[n_id]['sub_cat']

                    _tfidf = news_tfidf[n_id] if n_id in news_tfidf else ""

                    hist_tfidf = user_tfidf[u_id] if u_id in user_tfidf else ""

                    fea_list = [
                        idx+offset_list[i]
                        # idx + idx_list
                        for i, idx in enumerate(
                        [
                            user_encode[u_id]
                                if u_id in user_encode
                                else user_encode['cold_user'],
                            news_encode[n_id]
                                if n_id in news_encode
                                else news_encode['cold_news'],
                            cat_encode[cat] 
                                if cat in cat_encode
                                else cat_encode['cold_cat'],
                            sub_cat_encode[sub_cat] 
                                if sub_cat in sub_cat_encode
                                else sub_cat_encode['cold_sub_cat']
                        ])
                    ]
                    fea =  " ".join(["{}:1"]*len(fea_list)).format(*fea_list)
                    pos_out = " ".join(["1", _tfidf, hist_tfidf, fea])+"\n"

                    out_f.write(pos_out)
                    for j in range(args.negative_num):
                        n_k = negative_samples[sample_index[k]]
                        cat = news_map[n_k]['cat']
                        sub_cat = news_map[n_k]['sub_cat']

                        _tfidf = news_tfidf[n_id] if n_id in news_tfidf else ""

                        hist_tfidf = user_tfidf[u_id] if u_id in user_tfidf else ""

                        fea_list = [
                            idx+offset_list[i]
                            # idx + idx_list
                            for i, idx in enumerate(
                            [
                                user_encode[u_id]
                                    if u_id in user_encode
                                    else user_encode['cold_user'],
                                news_encode[n_id]
                                    if n_id in news_encode
                                    else news_encode['cold_news'],
                                cat_encode[cat] 
                                    if cat in cat_encode
                                    else cat_encode['cold_cat'],
                                sub_cat_encode[sub_cat] 
                                    if sub_cat in sub_cat_encode
                                    else sub_cat_encode['cold_sub_cat']
                            ])
                        ]
                        fea =  " ".join(["{}:1"]*len(fea_list)).format(*fea_list)
                        neg_out = " ".join(['0', _tfidf, hist_tfidf, fea])+"\n"
                        out_f.write(neg_out)
                        k += 1
        out_f.close()

else: # valid, test set use the default data
    with open(args.beh_file) as f:
        out_f = open(args.out, 'w', encoding='utf-8')
        for line in tqdm(f.readlines()):
            imp_id, u_id, imp_time, history, imprs = line.split("\t")
            
            for impr in imprs.strip().split(' '):
                label = impr[-1]

                n_id = impr[:-2] \
                    if impr[:-2] in news_map \
                    else 'cold_news'

                cat = news_map[n_id]['cat'] \
                    if len(news_map[n_id]['cat']) > 0 \
                    else "cold_cat"
                
                sub_cat = news_map[n_id]['sub_cat'] \
                    if len(news_map[n_id]['sub_cat']) > 0  \
                    else "cold_cat"
                    
                _tfidf = news_tfidf[n_id] if n_id in news_tfidf else ""

                hist_tfidf = user_tfidf[u_id] if u_id in user_tfidf else ""

                fea_list = [
                    idx+offset_list[i]
                    # idx + idx_list
                    for i, idx in enumerate(
                    [
                        user_encode[u_id]
                            if u_id in user_encode
                            else user_encode['cold_user'],
                        news_encode[n_id]
                            if n_id in news_encode
                            else news_encode['cold_news'],
                        cat_encode[cat] 
                            if cat in cat_encode
                            else cat_encode['cold_cat'],
                        sub_cat_encode[sub_cat] 
                            if sub_cat in sub_cat_encode
                            else sub_cat_encode['cold_sub_cat']
                    ])
                ]
                fea =  " ".join(["{}:1"]*len(fea_list)).format(*fea_list)

                pos_out = " ".join([label, _tfidf, hist_tfidf, fea])+"\n"
             
                out_f.write(pos_out)
