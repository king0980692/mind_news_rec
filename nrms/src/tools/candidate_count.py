from tqdm import tqdm
import sys

cnt = 0
with open(sys.argv[1], 'r') as f:
    for line in tqdm(f):
        iid, uid, time, his, imprs =  line.rstrip().split('\t')
        cnt += len(imprs.split())

print(cnt)


