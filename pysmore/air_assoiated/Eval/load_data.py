'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        cart_file = path + '/cart.txt'
        pv_file = path + '/pv.txt'
        

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0 # purchase
        self.n_train_pv, self.n_train_cart = 0, 0 
        self.neg_pools = {}

        self.exist_users = [] # purchase
        self.exist_users_pv = []
        self.exist_users_cart = []
        
        self.train_items, self.train_items_pv, self.train_items_cart = {}, {}, {}
        self.train_users, self.train_users_pv, self.train_users_cart = {}, {}, {}
        self.test_set = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_items[uid] = items
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    for iid in items:
                        if iid not in self.train_users:
                            self.train_users[iid] = []
                        self.train_users[iid] += [uid]
                    

        with open(cart_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_items_cart[uid] = items
                    self.exist_users_cart.append(uid)
                    #self.n_items = max(self.n_items, max(items))
                    #self.n_users = max(self.n_users, uid)
                    self.n_train_cart += len(items)
                    for iid in items:
                        if iid not in self.train_users_cart:
                            self.train_users_cart[iid] = []
                        self.train_users_cart[iid] += [uid]

        with open(pv_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_items_pv[uid] = items
                    self.exist_users_pv.append(uid)
                    #self.n_items = max(self.n_items, max(items))
                    #self.n_users = max(self.n_users, uid)
                    self.n_train_pv += len(items)
                    for iid in items:
                        if iid not in self.train_users_pv:
                            self.train_users_pv[iid] = []
                        self.train_users_pv[iid] += [uid]

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])
                    self.test_set[uid] = items
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        #self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        
        self.exist_users_all = [self.exist_users, self.exist_users_pv, self.exist_users_cart] # purchase, pv, cart
        self.train_items_all = [self.train_items, self.train_items_pv, self.train_items_cart] # purchase, pv, cart
        self.train_users_all = [self.train_users, self.train_users_pv, self.train_users_cart] # purchase, pv, cart
        
