from tqdm import tqdm
from datetime import datetime
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--delimiter', '-d')
parser.add_argument('--target_field', '-f', type=int)
parser.add_argument('--out')
args = parser.parse_args()

t_list = []
with open (args.file) as f:
    print("Load file ...")
    lines = f.readlines()
    lines = [ line.rstrip()  for line in tqdm(lines) if datetime.strptime(line.split("\t")[2], '%m/%d/%Y %H:%M:%S %p') > datetime(2019, 11, 12, 1, 0, 1)]
    # for line in lines:
        # t = line.rstrip().split(args.delimiter)[args.target_field]
        # t = line.rstrip().split("\t")[args.target_field]
        # t_list.append(t)

    # t_list = np.array(t_list)
    lines.sort(key=lambda x: datetime.strptime(
        x.split("\t")[2] , '%m/%d/%Y %H:%M:%S %p'))

    train_path = os.path.join(args.out,"train", "behaviors.tsv")

    np.savetxt(train_path, lines, fmt='%s')

    # idx_list = np.argsort(t_list)
    # lines = np.array(lines,dtype=str)[idx_list]


    # split_ratio = 0.8
    # split_cut = int(split_ratio * len(lines))

    # # a = np.array_split(lines, [split_cut])
    # train_record, test_record = np.array_split(lines, [split_cut])
    # test_path = os.path.join(args.out,"test", "behaviors.tsv")
    # train_path = os.path.join(args.out,"train", "behaviors.tsv")

    # np.savetxt(test_path, test_record, fmt='%s')
    # np.savetxt(train_path, train_record, fmt='%s')

