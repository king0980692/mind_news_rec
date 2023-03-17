import os
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import math
import argparse

def convert_unique_idx(df, column_name):

    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    rv_column_dict = {i: x for i, x in enumerate(df[column_name].unique())}

    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')

    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict, rv_column_dict


def create_user_list(df, user_size):
    user_list = [list() for u in range(user_size)]

    for row in df.itertuples():
        user_list[row.user].append((row.time, row.item, row.rate))
    return user_list


def split_train_test(df, user_size, test_size=0.2, time_order=False):
    """Split a dataset into `train_user_list` and `test_user_list`.
    Because it needs `user_list` for splitting dataset as `time_order` is set,
    Returning `user_list` data structure will be a good choice."""
    # TODO: Handle duplicated items

    if not time_order:
        test_idx = np.random.choice(len(df), size=int(len(df)*test_size))
        train_idx = list(set(range(len(df))) - set(test_idx))
        test_df = df.loc[test_idx].reset_index(drop=True)
        train_df = df.loc[train_idx].reset_index(drop=True)

        test_user_list = create_user_list(test_df, user_size)
        train_user_list = create_user_list(train_df, user_size)
    else:
        total_user_list = create_user_list(df, user_size)
        train_user_list = [None] * len(total_user_list)
        test_user_list = [None] * len(total_user_list)
        for user, item_list in enumerate(total_user_list):
            # Choose latest item
            item_list = sorted(item_list, key=lambda x: x[0])
            # Split item
            test_item = item_list[math.ceil(len(item_list)*(1-test_size)):]
            train_item = item_list[:math.ceil(len(item_list)*(1-test_size))]
            # Register to each user list
            test_user_list[user] = test_item
            train_user_list[user] = train_item
            

    # Remove time
    train_rate_list = [list(map(lambda x: x[2], l)) for l in train_user_list]
    train_user_list = [list(map(lambda x: x[1], l)) for l in train_user_list]

    test_user_list = [list(map(lambda x: x[1], l)) for l in test_user_list]

    return train_user_list, test_user_list, train_rate_list


def create_pair(user_list):
    pair = []
    for user, item_list in enumerate(user_list):
        pair.extend([(user, item) for item in item_list])
    return pair

def create_pair_weighted(user_list, user_rate_list):
    pair = []
    for user, item_list in tqdm(enumerate(user_list)):
        rate_list = user_rate_list[user]

        a=[(user, item)  for rate,item in zip(item_list, rate_list) for _ in range(rate)]
        # b=[(user, item)  for rate,item in zip(item_list, rate_list) ]
        pair.extend(a)
        # pair.extend([(user, item) for item in item_list])
    return pair

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def proc_args():
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join('preprocessed', 'ml-1m.pickle'),
                        help="File path for data")
    # Output embedding
    parser.add_argument('--saved_emb',
                        type=str,
                        help="Embedding file output")
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=32,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.025,
                        help="Weight decay factor")
    # Training
    parser.add_argument('--gpu',
                        type=str2bool,
                        default=1,
                        help="use gpu or not")
    parser.add_argument('--n_epochs',
                        type=int,
                        default=100,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="Batch size in one iteration")
    parser.add_argument('--print_every',
                        type=int,
                        default=20,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=10000,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--update_times',
                        type=int,
                        default=200,
                        help="Period for saving model during training")
    parser.add_argument('--fetch_worker',
                        type=int,
                        default=8,
                        help="the core of dataloader use to fetch data.")
    ## for preprocess
    parser.add_argument('--dataset',
                        choices=['ui', 'list'])
    parser.add_argument('--data_dir',
                        type=str,
                        default=os.path.join('data', 'ml-1m'),
                        help="File path for raw data")
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help="Proportion for training and testing split")
    parser.add_argument('--time_order',
                        action='store_true',
                        help="Proportion for training and testing split")
    args = parser.parse_args()

    return args
