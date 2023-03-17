'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Input data path.')

    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei, Taobao}')

    parser.add_argument('--Ks', nargs='?', default='[10,50,100]',
                        help='Top k(s) recommend')

    parser.add_argument('--user_emb', type=str, help='user_embedding')
    
    parser.add_argument('--item_emb', type=str, help='item_embedding')

    parser.add_argument('--user_rel_flag', type=int, default=0,
                            help='0: No user relation embedding, 1: Have user embeddings')

    parser.add_argument('--user_rel_emb', type=str, help='user_relation_embedding')

    parser.add_argument('--item_rel_flag', type=int, default=0,
                            help='0: No item relation embedding, 1: Have item embeddings')

    parser.add_argument('--item_rel_emb', type=str, help='item_relation_embedding')


    return parser.parse_args()
