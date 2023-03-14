import sys
import os
import pickle
from pathlib import Path
from tqdm import tqdm
import random
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_beh')
parser.add_argument('--pred')
parser.add_argument('--out')

args = parser.parse_args()


# for pkl in Path("./utils/").glob("./*.pkl"):
    # var_name = os.path.split(pkl)[-1].split(".")[0]
    # globals()[var_name] = pickle.load(open(pkl,'rb'))

def write_result_file(probs, libfm_result_file):
    k = 0
    with open(args.test_beh, 'r', encoding='utf-8') as behaviors_f:
        with open(libfm_result_file, 'w', encoding='utf-8') as f:
            for i, line in enumerate(behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                num = len(impressions.strip().split(' '))
                scores = []
                for j in range(num):
                    scores.append([probs[k], j])
                    k += 1
                scores.sort(key=lambda x: x[0], reverse=True)
                result = [0 for _ in range(num)]
                for j in range(num):
                    result[scores[j][1]] = j + 1
                f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    assert len(probs) == k, str(len(probs)) + ' - ' + str(k)

# len(user_encode)
# len(user_decode)
# pred_pair_list = []
# cnt = 1
# with open("./exp/dev.fm") as f:
    # for line in tqdm(f.readlines()):

        # uid, iid = line.split()
        # uid = int(uid.split(":")[0])
        # # iid = int(iid.split(":")[0])-len(user_decode) if int(iid.split(":")[0]) != 812571 else ''
        # if iid.split(":")[0] != '812751':
            # iid = int(iid.split(":")[0])-len(user_decode) 
        # else:
            # iid = int(iid.split(":")[0])

        # ori_uid = user_decode[uid]
        # ori_iid = news_decode[iid]

        # pred_pair_list.append([ori_uid, ori_iid])
        # # cnt +=1 
        # # if cnt == 200:
            # # break
        
# group_partition_cnt = []
# with open("./raw_data/valid/behaviors.tsv") as f:
    # for line in f.readlines():
        # imp_id, u_id, imp_time, history, imprs = line.split("\t")
        
        # group_partition_cnt.append(len(imprs.split()))
        


# impr_cnt = 1
# out_lines = []
# score_list = []
# last_user = pred_pair_list[0][0]

# cnt = 0

# with open("./exp_libfm/dev-200k.libfm.out") as f:
    # for line, pred_pair in zip(f.readlines(), pred_pair_list):
        # score = float(line.rstrip())
        # user_id, news_id = pred_pair

        # score_list.append((news_id, score, len(score_list)+1))
        # cnt += 1
        
        # if cnt == group_partition_cnt[impr_cnt-1]:
            # # Sort the score
            # score_list.sort(key=lambda x:x[1], reverse=True)
            # ranking_list = json.dumps([id for *_, id in score_list],separators=(',', ': '))
            # out_line = " ".join([str(impr_cnt), ranking_list])
            # out_lines.append(out_line)

            # # if impr_cnt == 3:
                # # exit()

            # # Append score list
            # score_list = []
            # # last_user = user_id
            # # score_list.append((news_id, score, len(score_list)+1))
            # impr_cnt+=1
            # cnt = 0


# with open("./exp/prediction.txt", 'w') as o:
    # o.write("\n".join(out_lines))


probs = []
with open(args.pred, 'r') as f:
    for line in f:
        if len(line.strip()) > 0:
            probs.append(float(line.strip()))

write_result_file(probs, args.out)

# with open('test/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open(f'test/res/libfm/{run_index}/libfm.txt', 'r', encoding='utf-8') as res_f:

    # auc, mrr, ndcg, ndcg10 = scoring(truth_f, res_f)
    # print('AUC =', auc)
    # print('MRR =', mrr)
    # print('nDCG@5 =', ndcg)
    # print('nDCG@10 =', ndcg10)

