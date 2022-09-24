import os
import pickle
from pathlib import Path
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--train')
parser.add_argument('--valid')
parser.add_argument('--test')
args = parser.parse_args()

# from traitlets.config.loader import Config
# from IPython.terminal.embed import InteractiveShellEmbed
# cfg = Config()
# cfg.TerminalInteractiveShell.editing_mode = 'vi' 
# ipshell2 = InteractiveShellEmbed(config=cfg,banner1='asd')
# ipshell2()
# exit()
# -----

user_encode = {}
news_encode = {}
for pkl in Path("./utils/").glob("./*.pkl"):
    var_name = os.path.split(pkl)[-1].split(".")[0]
    globals()[var_name] = pickle.load(open(pkl,'rb'))


tfidf_encode =  pickle.load(open(f"./exp/news_tfidf-{args.dataset}.pkl", 'rb'))

# ---------

def get_hist_id(hist_id):
    offset = len(user_encode) + len(news_encode)
    return str(int(news_encode[hist_id]) + offset)

def get_news_id(news_id):
    offset = len(user_encode)
    if news_id not in news_encode:
        return news_encode['news_cold']
    else:
        return str(int(news_encode[news_id]) + offset)



out_lines = []
with open(args.train) as f:
    for line in tqdm(f.readlines()):
        imp_id, u_id, imp_time, history, imprs = line.split("\t")

        sparse_u_id = str(user_encode[u_id])

        # sparse_hist_ids = [get_hist_id(h) for h in history.split()]

        for impr in imprs.split():

            n_id, label = impr.split('-')
            # if label == '0':
                # continue

            sparse_n_id = get_news_id(n_id)

            # print(sparse_u_id, sparse_n_id, sparse_hist_ids)
            feature = " ".join("{}:1".format(fea) for fea in [sparse_u_id, sparse_n_id])
            if n_id in tfidf_encode:
                tfidf_feature = " ".join("{}:{}".format(id+len(user_encode)+len(news_encode),vec) for id,vec in tfidf_encode[n_id].items() )
            else:
                tfidf_feature = ""

            out = " ".join([f"{label}", feature, tfidf_feature])
            out_lines.append(out)

            # for _ in range(5):
                # neg_item = random.randint(len(user_encode), len(user_encode)+len(news_encode))

                # neg_sam =  " ".join("{}:1".format(fea) for fea in [sparse_u_id, neg_item])

                # out_lines.append("0 "+neg_sam)
                
with open("./exp/train.fm", 'w') as o:
    o.write("\n".join(out_lines))



out_lines = []
with open(args.test) as f:
    for line in tqdm(f.readlines()):
        imp_id, u_id, imp_time, history, imprs = line.split("\t")

        if u_id not in user_encode:
            sparse_u_id = user_encode['user_cold']
        else:
            sparse_u_id = str(user_encode[u_id])

        # sparse_hist_ids = [get_hist_id(h) for h in history.split()]

        for impr in imprs.split():
            
            n_id = impr.split('-')[0]
            sparse_n_id = get_news_id(n_id)

            # print(sparse_u_id, sparse_n_id, sparse_hist_ids)
            feature = " ".join("{}:1".format(fea) for fea in [sparse_u_id, sparse_n_id])


            out_lines.append(feature)
                    
with open("./exp/dev.fm", 'w') as o:
    o.write("\n".join(out_lines))

