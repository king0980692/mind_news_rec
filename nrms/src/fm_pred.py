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



probs = []
with open(args.pred, 'r') as f:
    for line in tqdm(f):
        if len(line.strip()) > 0:
            probs.append(float(line.strip()))

write_result_file(probs, args.out)

