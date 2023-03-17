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
    def __init__(self, path, batch_size, neg_num):
        self.path = path
        self.batch_size = batch_size
        self.neg_num = neg_num

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
        self.print_statistics()
        
        self.exist_users_all = [self.exist_users, self.exist_users_pv, self.exist_users_cart] # purchase, pv, cart
        self.train_items_all = [self.train_items, self.train_items_pv, self.train_items_cart] # purchase, pv, cart
        self.train_users_all = [self.train_users, self.train_users_pv, self.train_users_cart] # purchase, pv, cart


    def sample(self):
        self.relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        #self.relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r, r+
        users = [rd.choice(self.exist_users_all[i]) for i in self.relation_batch] # u

        self.neg_relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        #self.neg_relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r-
        neg_users = [rd.choice(self.exist_users_all[i]) for i in self.neg_relation_batch] # u-

        def sample_items_for_u(u, r, num): # i ,  i-
            pos_items = self.train_items_all[r][u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pos_items_for_u(u, r, i, num):
            pos_items = self.train_items_all[r][u]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                while n_pos_items != 1 and pos_i_id == i :
                    pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                    pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch


        def sample_pos_u(i, r, num):
            pos_users = self.train_users_all[r][i]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_u_id = pos_users[pos_id]
                if pos_u_id not in pos_batch:
                    pos_batch.append(pos_u_id)
            return pos_batch

        #def sample_neg_items_for_u_from_pools(u, num):
        #    neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
        #    return rd.sample(neg_items, num)

        nor_items, pos_items, neg_items = [], [], []
        pos_users = []
        for u in range(len(users)): 
            nor_items += sample_items_for_u(users[u], self.relation_batch[u], 1) # i
            neg_items += sample_items_for_u(neg_users[u], self.neg_relation_batch[u], 1) # i-
            pos_items += sample_pos_items_for_u(users[u], self.relation_batch[u], nor_items[-1], 1) # i+
            pos_users += sample_pos_u(pos_items[-1], self.relation_batch[u], 1) # u+

        return users, nor_items, pos_users, pos_items, neg_users, neg_items
    
#####################
    def sample_rel(self):

        #self.relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        self.relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r, r+
        users = [rd.choice(self.exist_users_all[i]) for i in self.relation_batch] # u
        u_rel_idx =  [ 3*users[i] + self.relation_batch[i] for i in range(self.batch_size) ]

        #self.neg_relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        self.neg_relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r-
        neg_users = [rd.choice(self.exist_users_all[i]) for i in self.neg_relation_batch] # u-
        neg_u_rel_idx =  [ 3*neg_users[i] + self.neg_relation_batch[i] for i in range(self.batch_size) ]

        def sample_items_for_u(u, r, num): # i ,  i-
            pos_items = self.train_items_all[r][u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pos_items_for_u(u, r, i, num):
            pos_items = self.train_items_all[r][u]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                while n_pos_items != 1 and pos_i_id == i :
                    pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                    pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch


        def sample_pos_u(i, r, num):
            pos_users = self.train_users_all[r][i]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_u_id = pos_users[pos_id]
                if pos_u_id not in pos_batch:
                    pos_batch.append(pos_u_id)
            return pos_batch

        #def sample_neg_items_for_u_from_pools(u, num):
        #    neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
        #    return rd.sample(neg_items, num)

        nor_items, pos_items, neg_items = [], [], []
        pos_users = []
        for u in range(len(users)): 
            nor_items += sample_items_for_u(users[u], self.relation_batch[u], 1) # i
            neg_items += sample_items_for_u(neg_users[u], self.neg_relation_batch[u], 1) # i-
            pos_items += sample_pos_items_for_u(users[u], self.relation_batch[u], nor_items[-1], 1) # i+
            pos_users += sample_pos_u(pos_items[-1], self.relation_batch[u], 1) # u+

        pos_u_rel_idx =  [ 3*pos_users[i] + self.relation_batch[i] for i in range(self.batch_size) ]

        return users, nor_items, pos_users, pos_items, neg_users, neg_items, u_rel_idx, pos_u_rel_idx, neg_u_rel_idx



    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_purchase_interactions=%d' % (self.n_train ))
        print('n_cart_interactions=%d' % (self.n_train_cart ))
        print('n_pv_interactions=%d' % (self.n_train_pv ))
        #print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))




    def sample_neg(self):
        #self.relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        self.relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r, r+
        users = [rd.choice(self.exist_users_all[i]) for i in self.relation_batch] # u

        #self.neg_relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        self.neg_relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size * self.neg_num)] # r-
        neg_users = [rd.choice(self.exist_users_all[i]) for i in self.neg_relation_batch] # u-

        def sample_items_for_u(u, r, num): # i ,  i-
            pos_items = self.train_items_all[r][u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pos_items_for_u(u, r, i, num):
            pos_items = self.train_items_all[r][u]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                while n_pos_items != 1 and pos_i_id == i :
                    pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                    pos_i_id = pos_items[pos_id]

                if n_pos_items > self.neg_num :
                    if pos_i_id not in pos_batch:
                        pos_batch.append(pos_i_id)
                else:
                    pos_batch.append(pos_i_id)

            return pos_batch


        def sample_pos_u(i, r, num):
            pos_users = self.train_users_all[r][i]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_u_id = pos_users[pos_id]
                if pos_u_id not in pos_batch:
                    pos_batch.append(pos_u_id)
            return pos_batch


        nor_items, pos_items, neg_items = [], [], []
        pos_users = []
        for u in range(len(users)): 
            nor_items += sample_items_for_u(users[u], self.relation_batch[u], 1) # i
            pos_items += sample_pos_items_for_u(users[u], self.relation_batch[u], nor_items[-1], self.neg_num) # i+
            pos_users += [sample_pos_u(pos_items[j + self.neg_num*u ], self.relation_batch[u], 1)[0] for j in range(self.neg_num) ] # u+

        multi_users = []
        multi_items = []
        multi_rel = []
        for u in range(len(users)):
            multi_users += [users[u]] * self.neg_num
            multi_items += [nor_items[u]] * self.neg_num
            multi_rel += [self.relation_batch[u]] * self.neg_num

        for u in range(len(neg_users)):
            neg_items += sample_items_for_u(neg_users[u], self.neg_relation_batch[u], 1) 

        return multi_users, multi_items, pos_users, pos_items, neg_users, neg_items


    def sample_negrel(self):

        #self.relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        self.relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r, r+
        users = [rd.choice(self.exist_users_all[i]) for i in self.relation_batch] # u
        u_rel_idx =  [ 3*users[i] + self.relation_batch[i] for i in range(self.batch_size) ]

        #self.neg_relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        self.neg_relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size * self.neg_num)] # r-
        neg_users = [rd.choice(self.exist_users_all[i]) for i in self.neg_relation_batch] # u-
        neg_u_rel_idx =  [ 3*neg_users[i] + self.neg_relation_batch[i] for i in range(self.batch_size * self.neg_num) ]

        def sample_items_for_u(u, r, num): # i ,  i-
            pos_items = self.train_items_all[r][u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pos_items_for_u(u, r, i, num):
            pos_items = self.train_items_all[r][u]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                while n_pos_items != 1 and pos_i_id == i :
                    pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                    pos_i_id = pos_items[pos_id]

                if n_pos_items >self.neg_num :
                    if pos_i_id not in pos_batch:
                        pos_batch.append(pos_i_id)
                else:
                    pos_batch.append(pos_i_id)

            return pos_batch

        def sample_pos_u(i, r, num):
            pos_users = self.train_users_all[r][i]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_u_id = pos_users[pos_id]
                if pos_u_id not in pos_batch:
                    pos_batch.append(pos_u_id)
            return pos_batch

        #def sample_neg_items_for_u_from_pools(u, num):
        #    neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
        #    return rd.sample(neg_items, num)

        nor_items, pos_items, neg_items = [], [], []
        pos_users = []
        pos_u_rel_idx = []
        for u in range(len(users)): 
            nor_items += sample_items_for_u(users[u], self.relation_batch[u], 1) # i
            pos_items += sample_pos_items_for_u(users[u], self.relation_batch[u], nor_items[-1], self.neg_num) # i+
            pos_users += [sample_pos_u(pos_items[j + self.neg_num*u ], self.relation_batch[u], 1)[0] for j in range(self.neg_num) ] # u+
            pos_u_rel_idx += [ 3*pos_users[j + self.neg_num*u] + self.relation_batch[u] for j in range(self.neg_num) ] 

        multi_users = []
        multi_items = []
        multi_u_rel_idx = []
        for u in range(len(users)):
            multi_users += [users[u]]*self.neg_num
            multi_items += [nor_items[u]]*self.neg_num
            multi_u_rel_idx += [u_rel_idx[u]]*self.neg_num

        for u in range(len(neg_users)):
            neg_items += sample_items_for_u(neg_users[u], self.neg_relation_batch[u], 1)

        return multi_users, multi_items, pos_users, pos_items, neg_users, neg_items, multi_u_rel_idx, pos_u_rel_idx, neg_u_rel_idx


    def sample_negrel_ui(self):

        def sample_items_for_u(u, r, num): # i ,  i-
            pos_items = self.train_items_all[r][u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_pos_items_for_u(u, r, i, num):
            pos_items = self.train_items_all[r][u]

            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                while n_pos_items != 1 and pos_i_id == i :
                    pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                    pos_i_id = pos_items[pos_id]

                if n_pos_items >self.neg_num :
                    if pos_i_id not in pos_batch:
                        pos_batch.append(pos_i_id)
                else:
                    pos_batch.append(pos_i_id)

            return pos_batch

        def sample_pos_u(i, r, num):
            pos_users = self.train_users_all[r][i]
            n_pos_users = len(pos_users)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
                pos_u_id = pos_users[pos_id]
                if pos_u_id not in pos_batch:
                    pos_batch.append(pos_u_id)
            return pos_batch

        #self.relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r, r+
        users = [rd.choice(self.exist_users_all[i]) for i in relation_batch] # u
        u_rel_idx =  [ 3*users[i] + relation_batch[i] for i in range(self.batch_size) ]

        #self.neg_relation_batch = [np.random.choice(np.arange(0, 3), p=[0.25,0.45,0.3])  for _ in range(self.batch_size)]
        neg_relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size * self.neg_num)] # r-
        neg_users = [rd.choice(self.exist_users_all[i]) for i in neg_relation_batch] # u-
        neg_items = [sample_items_for_u(neg_users[i], neg_relation_batch[i], 1)[0] for i in range(self.batch_size * self.neg_num) ]

        neg_u_rel_idx =  [ 3*neg_users[i] + neg_relation_batch[i] for i in range(self.batch_size * self.neg_num) ]
        neg_i_rel_idx =  [ 3*neg_items[i] + neg_relation_batch[i] for i in range(self.batch_size * self.neg_num) ]


        nor_items, pos_items = [], []
        pos_users = []
        pos_u_rel_idx, pos_i_rel_idx = [], []
        i_rel_idx = []
        for u in range(len(users)): 
            nor_items += sample_items_for_u(users[u], relation_batch[u], 1) # i
            i_rel_idx += [3*nor_items[u] + relation_batch[u]]

            pos_items += sample_pos_items_for_u(users[u], relation_batch[u], nor_items[-1], self.neg_num) # i+
            pos_users += [sample_pos_u(pos_items[j + self.neg_num*u ], relation_batch[u], 1)[0] for j in range(self.neg_num) ] # u+
            pos_u_rel_idx += [ 3*pos_users[j + self.neg_num*u] + relation_batch[u] for j in range(self.neg_num) ] 
            pos_i_rel_idx += [ 3*pos_items[j + self.neg_num*u] + relation_batch[u] for j in range(self.neg_num) ] 

        multi_users, multi_items = [], []
        multi_u_rel_idx, multi_i_rel_idx = [], []

        for u in range(len(users)):

            multi_users += [users[u]]*self.neg_num
            multi_items += [nor_items[u]]*self.neg_num
            multi_u_rel_idx += [u_rel_idx[u]]*self.neg_num
            multi_i_rel_idx += [i_rel_idx[u]]*self.neg_num

        return multi_users, multi_items, pos_users, pos_items, neg_users, neg_items, multi_u_rel_idx, pos_u_rel_idx, neg_u_rel_idx, multi_i_rel_idx, pos_i_rel_idx, neg_i_rel_idx
