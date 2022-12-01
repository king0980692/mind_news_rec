import json
from tqdm import tqdm
import sys

cnt = 0
ans = []
with open('./data/MINDsmall_dev/behaviors.tsv', 'r') as f:
    for id, line in tqdm(enumerate(f,1)):
        iid, uid, time, his, imprs =  line.rstrip().split('\t')
        targets = [int(imp.split('-')[1]) for imp in imprs.split()]
        ans.append(str(id)+" "+json.dumps(targets,separators=(',', ':'))+"\n")


with open('./data/MINDsmall_dev/truth.txt', 'w') as f:
    f.writelines(ans)


