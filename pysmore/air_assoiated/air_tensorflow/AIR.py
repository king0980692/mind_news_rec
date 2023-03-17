'''
Created on Oct 10, 2018
Tensorflow Implementation of the baseline of "Matrix Factorization with BPR Loss" in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
from utility.helper import *
import numpy as np
from utility.batch_test1 import *
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class BPRMF(object):
    def __init__(self, data_config, pretrain_data):
        self.model_type = args.model_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.pretrain_data = pretrain_data
        self.lr = args.lr
        # self.lr_decay = args.lr_decay

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_users = tf.placeholder(tf.int32, shape=(None,))
        self.neg_users = tf.placeholder(tf.int32, shape=(None,))

        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))


        self.weights = self._init_weights()

        # Original embedding.
        self.ua_embeddings, self.ia_embeddings = self.weights['user_embedding'], self.weights['item_embedding']

        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.pos_users)
        self.neg_u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.neg_users)

        self.i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.items)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        # All ratings for all users.
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                            self.pos_u_g_embeddings,
                                                            self.neg_u_g_embeddings,
                                                            self.i_g_embeddings,
                                                            self.pos_i_g_embeddings,
                                                            self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        #self.opt =tf.train.AdagradOptimizer(learning_rate=0.05, initial_accumulator_value=1e-8).minimize(self.loss)

        self._statistics_params()


    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using random initialization')#print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')
        
        return all_weights


    def create_bpr_loss(self, users, pos_users, neg_users, items, pos_items, neg_items):
        
        u_i_term = tf.add(users, items) # g
        u_i_pos_term = tf.add(pos_users, pos_items) # g+
        u_i_neg_term = tf.add(neg_users, neg_items) # g-
        
        pos_scores = tf.reduce_sum(tf.multiply(u_i_term, u_i_pos_term), axis=1) # g.g+
        neg_scores = tf.reduce_sum(tf.multiply(u_i_term, u_i_neg_term), axis=1) # g.g-
        
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_users) + tf.nn.l2_loss(neg_users) + \
                      tf.nn.l2_loss(items) +tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / self.batch_size

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
#         maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
#         mf_loss = tf.negative(tf.reduce_mean(maxi))
        
        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    f0 = time()
    
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    pretrain_data = None
    model = BPRMF(data_config=config, pretrain_data=pretrain_data)
    saver = tf.train.Saver()

    #Save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/l%s_r%s_b%s' % (args.weights_path, args.dataset, model.model_type,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), args.batch_size)                            
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    # printing informaiton save path
    save_path = '%soutput/%s/%s_%d.result' % (args.proj_path, args.dataset, model.model_type, args.batch_size)
    ensureDir(save_path)
    
    f = open(save_path, 'a')
    f.write(
        'embed_size=%d, lr=%.4f, regs=%s, batch size=%d, neg_num=%d \n'
        % (args.embed_size, args.lr, args.regs, args.batch_size, args.neg_num ))
    f.close()

    #*********************************************************

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.
    print('without pretraining.')

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    users_to_test = list(data_generator.test_set.keys())

    for epoch in range(1, args.epoch + 1):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        #n_batch = (data_generator.n_train + data_generator.n_train_pv + data_generator.n_train_cart ) // args.batch_size + 1

        for idx in range(n_batch):

            users, items, pos_users, pos_items, neg_users, neg_items = data_generator.sample_neg()

            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss],
                                                        feed_dict={model.users: users, model.pos_users: pos_users, model.neg_users: neg_users, 
                                                                    model.items: items,
                                                                    model.pos_items: pos_items,
                                                                    model.neg_items: neg_items})

            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            emb_loss += batch_emb_loss/n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch % args.verbose) != 0:

            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            print(perf_str)

            with open(save_path, 'a+') as f:
                f.write(f'{perf_str}\n')

            continue

        ret = test(sess, model, users_to_test ,drop_flag=False)

        t3 = time()
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f, %.5f], precision=[%.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f]' % \
                   (epoch, time() - t1, loss, mf_loss, emb_loss, 
                   ret['recall'][0], ret['recall'][1], ret['recall'][2],
                    ret['precision'][0], ret['precision'][1], ret['precision'][2], 
                    ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2])

        print(perf_str)

        with open(save_path, 'a+') as f:
            f.write(f'{perf_str}\n')

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=50)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%.5f, %.5f, %.5f], precision=[%.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f]" % \
                 (idx, time() - t0, recs[idx][0], recs[idx][1], recs[idx][2],
                  pres[idx][0], pres[idx][1], pres[idx][2],
                  ndcgs[idx][0], ndcgs[idx][1], ndcgs[idx][2])

    print(final_perf)

    with open(save_path, 'a+') as f:
        f.write(f'{final_perf}\n')
