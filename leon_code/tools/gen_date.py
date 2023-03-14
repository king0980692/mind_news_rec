import os
import pickle
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--out')
args = parser.parse_args()

ts_list = []

for data_type in Path(args.data).iterdir():
    if 'train' not in str(data_type):
        continue
    beh_file = data_type / "behaviors.tsv"
    print("TIME - Encode: ", beh_file)
    with open(beh_file) as f:
        for line in tqdm(f.readlines()):
            imp_id, u_id, imp_time, history, imprs = line.split("\t")
            t_stamp = datetime.strptime(imp_time,
                    '%m/%d/%Y %H:%M:%S %p')

            ts_list.append(t_stamp)

ts_list.sort()
import IPython;IPython.embed(colors="neutral");exit(1) 

