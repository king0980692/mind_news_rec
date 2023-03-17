'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run AIR.")
    
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')

    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Input data path.')

    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='Beibei',
                        help='Choose a dataset from {Beibei, Taobao}')

    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')

    parser.add_argument('--pretrain_path', nargs='?', default='',
                        help=' Get the results with stored models. ')    
    
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')

    parser.add_argument('--batch_size', type=int, default=264,
                        help='Batch size.')

    parser.add_argument('--neg_num', type=int, default=1,
                        help='Number of negative sample .')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='air',
                        help='Specify the name of model (air).')
 
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--Ks', nargs='?', default='[10,50,100]',
                        help='Top k(s) recommend')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    return parser.parse_args()
