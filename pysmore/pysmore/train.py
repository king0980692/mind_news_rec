import os
import argparse

import numpy as np
from tqdm import trange, tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pysmore.DataHandler import DataInput 
from pysmore.models.bpr import BPR
from pysmore.DataHandler.TripletLoader import TripletUniformPair, TripletWeightedPair, TripletWeightedPair2
from pysmore.utils.utils import *



def pre_process(args):

    if args.dataset == 'ui':
        df = DataInput.UI_data(args.data_dir).load()
    else:
        raise NotImplementedError

    df, user_mapping, rv_user_mapping = convert_unique_idx(df, 'user')
    df, item_mapping, rv_item_mapping = convert_unique_idx(df, 'item')
    print('Complete assigning unique index to user and item')

    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())

    train_user_list, test_user_list, train_rate_list = split_train_test(df,
                                                       user_size,
                                                       test_size=args.test_size,
                                                       time_order=args.time_order)
    print('Complete spliting items for training and testing')
    train_pair = create_pair(train_user_list)
    # train_pair = create_pair_weighted(train_user_list, train_rate_list)
    print('Complete creating pair')

    dataset = {'user_size': user_size, 'item_size': item_size, 
               'user_mapping': user_mapping, 'item_mapping': item_mapping,
               'rv_user_mapping': rv_user_mapping, 'rv_item_mapping': rv_item_mapping,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'train_pair': train_pair, 'train_rate_list':train_rate_list
               }

    return dataset

def collate_fn(batchs):

    u_batch = [b[0] for batch in batchs for b in batch]
    i_batch = [b[1] for batch in batchs for b in batch]
    j_batch = [b[2] for batch in batchs for b in batch]

    return u_batch, i_batch, j_batch



def main(args, dataset):
    # Initialize seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    user_size, item_size = dataset['user_size'], dataset['item_size']
    train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
    train_pair = dataset['train_pair']
    rv_user_mapping = dataset['rv_user_mapping']
    rv_item_mapping = dataset['rv_item_mapping']
    train_rate_list = dataset['train_rate_list']
    print('Load complete')

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(item_size, train_user_list, train_rate_list, train_pair, True, args.n_epochs)
    # dataset = TripletWeightedPair(item_size, train_user_list, train_rate_list, train_pair, True, args.n_epochs)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.fetch_worker, collate_fn=collate_fn)
    # loader = DataLoader(dataset, batch_size=2 , num_workers=args.fetch_worker, collate_fn=collate_fn)

    # x = iter(loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    print("\n===== Device : {} ======\n".format(device))

    model = BPR(user_size, item_size, args.dim, args.weight_decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # writer = SummaryWriter()

    # Training
    smooth_loss = 0

    # args.update_times = args.update_times * 100
    args.update_times = len(train_pair) * args.n_epochs
    pbar = trange(args.update_times, desc='Loss: ', leave=True, 
            # bar_format='{desc} {percentage:3.2f}%|{r_bar}')
            # bar_format="{desc} {lbar}|{n:.1f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
            bar_format="{desc} {percentage:3.2f}% ({elapsed}<{remaining}, {rate_fmt})")
            # bar_format='{desc}: {percentage:3.2f}%|{bar}{r_bar}')

    model = model.to(device)


    idx = 0

    for u, i, j in loader:

        u = torch.tensor(u).to(device)
        i = torch.tensor(i).to(device)
        j = torch.tensor(j).to(device)

        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()
        # writer.add_scalar('train/loss', loss, idx1
        smooth_loss = smooth_loss*0.99 + loss*0.01

        # if idx % args.print_every == (args.print_every - 1):
            # print('loss: %.4f %d' % (smooth_loss, idx))
        pbar.set_description('Loss %3.4f' % (smooth_loss))

        idx += u.shape[0] # accumalted update sampe times

        pbar.update(u.shape[0])
        pbar.refresh() # to show immediately the update

        # if idx % args.eval_every == (args.eval_every - 1):
            # plist, rlist = precision_and_recall_k(  model.W.detach(),
                                                    # model.H.detach(),
                                                    # train_user_list,
                                                    # test_user_list,
                                                    # klist=[1, 5, 10])

            # pbar.set_postfix_str('P@1: %.4f, P@5: %.4f P@10: %.4f, R@1: %.4f, R@5: %.4f, R@10: %.4f' % (plist[0], plist[1], plist[2], rlist[0], rlist[1], rlist[2]))
            # print('\nP@1: %.4f, P@5: %.4f P@10: %.4f, R@1: %.4f, R@5: %.4f, R@10: %.4f' % (plist[0], plist[1], plist[2], rlist[0], rlist[1], rlist[2]))

            # writer.add_scalars('eval', {'P@1': plist[0],
                                                    # 'P@5': plist[1],
                                                    # 'P@10': plist[2]}, idx)
            # writer.add_scalars('eval', {'R@1': rlist[0],
                                                # 'R@5': rlist[1],
                                                # 'R@10': rlist[2]}, idx)
                                                
        # if idx == args.update_times and False: 
            # pbar.close() 
            # # save_embedding(model, rv_user_mapping, rv_item_mapping, args.saved_emb)
            # model.save_embedding(rv_user_mapping, rv_item_mapping, args.saved_emb)
            # exit()

    print("\nOut of traingin loops")
    model.save_embedding(rv_user_mapping, rv_item_mapping, args.saved_emb)
    exit()

# if __name__ == '__main__':
def entry_points():
    args = proc_args()
    dataset = pre_process(args)
    main(args, dataset)
