import sys
import gzip, json
import argparse
import random
import math
import concurrent.futures
from collections import defaultdict
from math import log, sqrt
from tqdm import tqdm
import numpy as np
import json

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--train', help='data.ui.train')
parser.add_argument('--test', help='data.ui.test')
parser.add_argument('--embed', help='embeddding file')
parser.add_argument('--emb_dim', type=int, default=64, help='emedding demensions')
parser.add_argument('--cold_user', type=int, default=0, help='to test cold user')
parser.add_argument('--cold_item', type=int, default=0, help='to test cold item')
parser.add_argument('--num_test', type=int, default=sys.maxsize, help='# of sampled tests')
parser.add_argument('--worker', type=int, default=1, help='# of threads')
parser.add_argument('--sim', choices=['dot', 'cosine'], default='dot', help='sim metric')

args = parser.parse_args()
seed = 2021

def eu_distance(v, v2):
    try:
        return -(sum( (a-b)**2 for a,b in zip(v, v2) )) # omit sqrt
    except:
        return -100.

def cosine_distance(v, v2):
    try:
        score = sum( (a*b) for a,b in zip(v, v2) ) / ( sqrt(sum(a*a for a in v)*sum(b*b for b in v2)) )
        if math.isnan(score):
            return 0
        return score
    except:
        return 0

def dot_sim(v, v2):
    try:
        score = sum( (a*b) for a,b in zip(v, v2) )
        if math.isnan(score):
            return 0
        return score
    except:
        return 0

sim = {'dot': dot_sim, 'cosine': cosine_distance}[args.sim]


def process_ui_query(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    item_pool = list(iids.keys()) + list(cold_iids.keys())

    ui_results = [uid, str(len(test_ui[uid]))]

    # scoring
    ui_scores = defaultdict(lambda: 0.)
    for rid in item_pool:
        if rid in train_ui[uid]:
            continue
        if uid in embed and rid in embed:
            ui_scores[rid] += sim(embed[uid], embed[rid]) #dot product
        else:
            ui_scores[rid] += 0.

    # ranking
    for rid in sorted(ui_scores, key=ui_scores.get, reverse=True)[:len(test_ui[uid])]:
        if rid in test_ui[uid]:
            ui_results.append('1') #hit or not
        else:
            ui_results.append('0')

    return ' '.join(ui_results)

def process_ui_query2(sid):

    global sim
    global iids, cold_iids
    # global train_ui, test_ui
    global embed
    global candidate_dict, id_user_dict
    global hist_dict

    uid = id_user_dict[sid]
    item_pool = candidate_dict[uid][sid]

    # item_pool = list(iids.keys()) + list(cold_iids.keys()) 

    observed_items = list(train_ui[uid].keys())

    # ui_results = [uid, str(len(test_ui[uid]))] # uid, total_len
    # scoring_matrix
    # q_vec = np.array(embed[uid]) if uid in embed else np.zeros((args.emb_dim,))
    q_vec = np.sum(
            [ embed[uid] if uid in embed else np.zeros((args.emb_dim,)) for uid in hist_dict[uid] ] ,
            axis=0)

    q_vec += embed[uid] if uid in embed else np.zeros((args.emb_dim))

    pool_vec = np.array([np.array(embed[it]) if it in embed and it not in observed_items else np.zeros((args.emb_dim,)) for it in item_pool])

    try:
        scores = pool_vec @ q_vec
    except:
        import IPython;IPython.embed(colors='linux');exit(1) 


    # [np.argsort(scores)[::-1]]
    # top_k_list = np.array(item_pool)[np.argsort(scores)[::-1]].tolist()[:len(test_ui[uid])]
    top_k_list = np.array(item_pool)[np.argsort(scores)[::-1]].tolist()

    out_list = json.dumps([item_pool.index(it)+1 for it in top_k_list],separators=(',', ':')) # +1, because eval is 1-indexing


    result = f'{sid} {out_list}\n' 

    # Evaluation
    # for rid in top_k_list:
    # # for rid in sorted(scores, key=ui_scores.get, reverse=True):
        # if rid in test_ui[uid]:
            # ui_results.append('1') #hit or not
        # else:
            # ui_results.append('0')

    return result

def process_ii_query2(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    item_pool = list(iids.keys()) + list(cold_iids.keys())
    observed_items = list(train_ui[uid].keys())

    ui_results = [uid, str(len(test_ui[uid]))]

    q_vec = np.sum([embed[i] for i in train_ui[uid]], axis=0)
    
    pool_vec = np.array([np.array(embed[it]) if it in embed and it not in observed_items else np.zeros((args.emb_dim,)) for it in item_pool])

    # scoring
    scores = pool_vec @ q_vec

    top_k_list = np.array(item_pool)[np.argsort(scores)[::-1]].tolist()[:len(test_ui[uid])]

    # ranking
    for rid in top_k_list:
        if rid in test_ui[uid]:
            ui_results.append('1')
        else:
            ui_results.append('0')

    return ' '.join(ui_results)
def process_ii_query(uid):

    global sim
    global iids, cold_iids
    global train_ui, test_ui
    global embed
    item_pool = list(iids.keys()) + list(cold_iids.keys())

    q_embed = np.sum([embed[i] for i in train_ui[uid]], axis=0)
    ui_results = [uid, str(len(test_ui[uid]))]

    # scoring
    ui_scores = defaultdict(lambda: 0.)
    for rid in item_pool:
        if rid in train_ui[uid]:
            continue
        if uid in embed and rid in embed:
            ui_scores[rid] += sim(q_embed, embed[rid])
        else:
            ui_scores[rid] += 0.

    # len(ui_score) = |item_pool| - ui_interactions

    # ranking
    for rid in sorted(ui_scores, key=ui_scores.get, reverse=True)[:len(test_ui[uid])]:
        if rid in test_ui[uid]:
            ui_results.append('1')
        else:
            ui_results.append('0')

    return ' '.join(ui_results)



print ('load train data from', args.train)
uids, iids = {}, {}
train_ui = defaultdict(dict)
train_counter = 0.
hist_dict = {}
with open(args.train) as f:
    for line in f:
        id, uid, time, history, imprs = line.rstrip().split('\t')
        train_counter += 1.
        uids[uid] = 1
        # iids[iid] = 1
        for iid in imprs.split():
            iid = iid[:-2]
            train_ui[uid][iid] = 1
        for iid in history.split():
            train_ui[uid][iid] = 1
        if uid not in hist_dict:
            hist_dict[uid] = history.split()

print ('load test data from', args.test)
test_ui = defaultdict(dict)
cold_uids, cold_iids = {}, {}
test_counter = 0.

s_id = []
id_user_dict = {}
candidate_dict = defaultdict(dict)
with open(args.test) as f:
    for line in f:
        id, uid, time, history, imprs = line.rstrip().split('\t')
        id = int(id)
        s_id.append(id)
        
        # if uid not in uids:
            # cold_uids[uid] = 1
        id_user_dict[id]=uid
        if uid not in hist_dict:
            hist_dict[uid] = history.split()
        # if uid not in candidate_dict:
            # candidate_dict[uid] = []
        if id not in candidate_dict[uid]:
            candidate_dict[uid][id] = []
        candidate_dict[uid][id].extend(list(map(lambda x:x[:-2], imprs.split())))


print ("load embeddings from", args.embed)
embed = {}
with open(args.embed, 'r') as f:
    lines = f.readlines()
    for line in lines[:]:
        line = line.rstrip().split('\t')
        ID = line[0]
        embed[ID] = list(map(float, line[1].split(' ')))


#  only using warm  user in test set
# warm_u_queries = [u for u in uids if u not in cold_uids] 
warm_u_queries = s_id

# print('warm user:', len(uids))
# print('cold user:', len(cold_uids))
# print('warm user query:', len(warm_u_queries))
# print('warm item:', len(iids))
# print('cold item:', len(cold_iids))
# print('avg. train item per user:', train_counter/len(train_ui))
# print('avg. test item per user:', test_counter/len(test_ui))

## U2I
random.shuffle(warm_u_queries)
warm_u_queries = warm_u_queries[:args.num_test]
# warm_u_queries = warm_u_queries[:]
rec_ui = []


for u in warm_u_queries:
    top_k = process_ui_query2(u)
exit()

print("recommendation ...")
with concurrent.futures.ProcessPoolExecutor(max_workers=args.worker) as executor:
    for res in tqdm(executor.map(process_ui_query2, warm_u_queries)):
        rec_ui.append(res)
print ('write the result to', args.embed+'.ui.rec')

rec_ui = sorted(rec_ui, key=lambda x:int(x.split()[0]))
with open(args.embed+'.ui.rec', 'w') as f:
    f.writelines(rec_ui)

exit()

## I2I
random.shuffle(warm_u_queries)
warm_u_queries = warm_u_queries[:args.num_test]
# warm_u_queries = warm_u_queries[:]
rec_ii = []
with concurrent.futures.ProcessPoolExecutor(max_workers=args.worker) as executor:
    for res in tqdm(executor.map(process_ii_query2, warm_u_queries)):
        rec_ii.append(res)
print ('write the result to', args.embed+'.ii.rec')
with open(args.embed+'.ii.rec', 'w') as f:
    f.write('%s\n' % ('\n'.join(rec_ii)))

