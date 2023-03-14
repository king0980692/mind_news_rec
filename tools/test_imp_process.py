import json
f = open("./data/dev/behaviors.tsv")
out_lines = []

for l_id, line in enumerate(f.readlines(), 1):
    *others, imps = line.split("\t")
    labels = [int(i[-1]) for i in imps.split()]
    new_line = " ".join([str(l_id), json.dumps(labels,separators=(',', ':'))])
    
    out_lines.append(new_line+"\n")
f.close()

open("./data/dev/truth.txt",'w').writelines(out_lines)

