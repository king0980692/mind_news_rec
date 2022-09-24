import os
import pickle
from tqdm import tqdm
import argparse
import random
from itertools import accumulate



parser = argparse.ArgumentParser()
parser.add_argument('--pkl_list', nargs='+')
parser.add_argument('--beh_file', type=str)
parser.add_argument('--negative_num', type=int, default=4)
parser.add_argument('--out')
args = parser.parse_args()


user_encode = {}
news_encode = {}
# Load all pkl 
for pkl in args.pkl_list:
    var_name = os.path.split(pkl)[-1].split(".")[0]
    globals()[var_name] = pickle.load(open(pkl,'rb'))

offset_list = list(accumulate([0, len(user_encode), len(news_encode)]))



if "train" in args.beh_file:
    with open(args.beh_file) as f:

        out_f = open(args.out, 'w', encoding='utf-8')
        for line in tqdm(f.readlines()):
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
                    # pos_out = '1 %d:1 %d:1\n' % (news_encode[positive_samples[i]], user_encode[u_id]) 
                    pos_out = \
                        "1 {}:1 {}:1\n".format(
                            *[idx+offset_list[i]
                            for i, idx in enumerate(
                            [
                                user_encode[u_id] 
                                    if u_id in user_encode
                                    else user_encode['cold_user'],
                                news_encode[positive_samples[i]] 
                                    if positive_samples[i] in news_encode
                                    else news_encode['cold_news']
                            ])
                        ])
                    
                    out_f.write(pos_out)
                    for j in range(args.negative_num):
                        n_k = negative_samples[k % negative_samples_num]
                        neg_out = \
                            "0 {}:1 {}:1\n".format(
                                *[idx+offset_list[i]
                                for i, idx in enumerate(
                                [
                                    user_encode[u_id]
                                        if u_id in user_encode
                                        else user_encode['cold_user'],
                                    news_encode[n_k]
                                        if n_k in news_encode
                                        else news_encode['cold_news']
                                ])
                            ])

                        out_f.write(neg_out)
                        k += 1
            else:
                sample_index = random.sample(
                        range(negative_samples_num), 
                        positive_samples_num * args.negative_num
                    )

                k = 0
                for i in range(positive_samples_num):
                    pos_out = \
                            "1 {}:1 {}:1\n".format(
                                *[idx+offset_list[i]
                                for i, idx in enumerate(
                                [
                                    user_encode[u_id]
                                        if u_id in user_encode
                                        else user_encode['cold_user'],
                                    news_encode[positive_samples[i]]
                                        if positive_samples[i] in news_encode
                                        else news_encode['cold_news']
                                ])
                            ])

                    out_f.write(pos_out)
                    for j in range(args.negative_num):
                        n_k = negative_samples[sample_index[k]]
                        neg_out = \
                            "0 {}:1 {}:1\n".format(
                                *[idx+offset_list[i]
                                for i, idx in enumerate(
                                [
                                    user_encode[u_id]
                                        if u_id in user_encode
                                        else user_encode['cold_user'],
                                    news_encode[n_k]
                                        if n_k in news_encode
                                        else news_encode['cold_news']
                                ])
                            ])

                        out_f.write(neg_out)
                        k += 1
        out_f.close()

else:
    with open(args.beh_file) as f:
        out_f = open(args.out, 'w', encoding='utf-8')
        for line in tqdm(f):
            imp_id, u_id, imp_time, history, imprs = line.split("\t")

            for impr in imprs.strip().split(' '):
                pos_out = \
                        "1 {}:1 {}:1\n".format(
                            *[idx+offset_list[i]
                            for i, idx in enumerate(
                            [
                                user_encode[u_id]
                                    if u_id in user_encode
                                    else user_encode['cold_user'],
                                news_encode[impr]
                                    if impr in news_encode
                                    else news_encode['cold_news']
                            ])
                        ])

                out_f.write(pos_out)
