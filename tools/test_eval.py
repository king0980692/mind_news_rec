import random
out_files = []
with open("./raw_data/valid/behaviors.tsv") as f:
    for line in f:
        id, uid, imp_time, history, impr = line.split("\t")
        impr = impr.split()
        # impr = [i.split('-')[0] for i in impr.split()]
        impr = list(map(str, range(1,len(impr)+1)))
        random.shuffle(impr)

        impr = ",".join(impr)
        fake_pred = "[" + impr  + "]"
        # print(fake_pred)
        out = id + " " + fake_pred + '\n'
        out_files.append(out)

with open("./fake_pred.txt", 'w') as o:
    o.writelines(out_files)


