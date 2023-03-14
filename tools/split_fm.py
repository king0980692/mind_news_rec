from tqdm import tqdm

train_list = []
test_list = []
with open("./exp/train.fm") as f:
    lines = f.readlines()
    total_len = len(lines)

    split_cut = int(total_len*0.8)+1

    train_list = lines[:split_cut]
    test_list = lines[-(total_len-split_cut):]

    # for id, line in tqdm(enumerate(lines), total=total_len):
        # if id < split_cut:
            # train_list.append(line)
        # else:
            # test_list.append(line)

with open("exp/split_train.fm",'w') as o:
    o.writelines(train_list)

with open("exp/split_valid.fm",'w') as o:
    o.writelines(test_list)
