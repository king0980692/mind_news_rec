import numpy as np
import pandas as pd
import pickle
from parser import parse_args

from load_data import *
from batch_test import *

user_emb = args.user_emb
item_emb = args.item_emb
user_rel_flag = args.user_rel_flag
item_rel_flag = args.item_rel_flag

# Note : 
# data_generator.test_set is a dict that store user and item in test set
# eg  For Beibei , {0: [7599],
#                   1: [7513], ....}
users_to_test = list(data_generator.test_set.keys()) 
items_to_test = list(data_generator.test_set.values())
items_to_test = [i[0] for i in items_to_test]


# read user and item embeddings
with open (user_emb, 'rb') as fp: # 'embedding/normal1/U_90000000'
    U = pickle.load(fp) # Beibei: (21716, dim), start with user_0, user_1, ...
with open (item_emb, 'rb') as fp: # 'embedding/normal1/I_90000000'
    I = pickle.load(fp) # Beibei: (7977, dim)


# Note: Each user has its own behaviors. Total user-relation matrix: ( 3 x num of user, dim)
# read relation embedding of user and item
if user_rel_flag == 1 or item_rel_flag == 1:
    with open (args.user_rel_emb, 'rb') as fp: # 'embedding/ui/UR_1000000'
        UR = pickle.load(fp) # Beibei: (21716x3, dim) = (65148, dim)

    # Note: Each user has its own behaviors. Total user-relation matrix: ( 3 x num of user, dim)
    purc_rel_to_test = [ 3*us + 0   for us in users_to_test] # 0: purchase, 1: pv, 2: cart
    # eg For Beibei,  user_0 :  index 0: purc , index 1: pv  , index 2: cart
    #                 user_1 :  index 3: purc , index 4: pv  , index 5: cart
    #                 user_k :  index 3*k+0: purc, index 3*k+1: pv, index 3*k+2: cart 

    if item_rel_flag == 1:
        with open (args.item_rel_emb, 'rb') as fp: # 'embedding/ui/IR_1000000'
            IR = pickle.load(fp) # Beibei: (7977x3, dim) = (23931, dim)

        purc_rel_i_to_test =[ 3*us + 0  for us in range(data_generator.n_items)] # similar as above


# u dot i 
if args.user_rel_flag == 0 and args.item_rel_flag==0:
    ret_ui = test_rel(  users_to_test, U, I )
    print('u dot i')
    print(ret_ui)

# (u+r) dot i 
if args.user_rel_flag == 1 and args.item_rel_flag==0:
    ret_ur = test_rel_ui( users_to_test, U, I, UR, purc_rel_to_test )
    print('(u+r) dot i')
    print(ret_ur)

# (u+r) dot (i+r) 
if args.user_rel_flag == 1 and args.item_rel_flag == 1 :
    ret_uir = test_rel_uir(  users_to_test, U, I, UR, IR, purc_rel_to_test , purc_rel_i_to_test )
    print('(u+r) dot (i+r)')
    print(ret_uir)

