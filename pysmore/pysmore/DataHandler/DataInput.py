import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import sys

from torch.utils.data import IterableDataset, Dataset
import random

class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

class UI_data(DatasetLoader):
    def __init__(self, data_path, sep='\t'):
        # self.fpath = os.path.join(data_dir, 'ratings.dat')
        self.fpath = data_path
        self.sep = sep

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep=self.sep,
                         engine='python',
                         # names=['user', 'item', 'rate'])
                         names=['user', 'item', 'rate', 'time'])
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return df

                


def load_test(test_file,neg_file,test_num=100):
    test_data = []
    with open(neg_file, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]

            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()

    return test_data

def load_train(train_file):
    import time
    # s_time = time.time()
    # train_data = pd.read_csv(
        # train_file, 
        # sep='\t', header=None, names=['user', 'item'], 
        # usecols=[0, 1], dtype={ 1: np.int32})
    # e_time = time.time()
    # print(e_time-s_time)

    def load_by_line(_f):
        user_dict = {}
        item_dict = {}
        max_user_id = -1
        max_item_id = -1
        data = []
        for line in open(_f):
            uid, iid, *_ = line.rstrip().split("\t")

            # if uid not in user_dict:
                # user_dict[uid] = len(user_dict)
            # if iid not in item_dict:
                # item_dict[iid] = len(item_dict)
            max_user_id = max(int(uid),max_user_id)
            max_item_id = max(int(iid),max_item_id)

            # data.append([user_dict[uid], item_dict[iid]])
            data.append([int(uid), int(iid)])

        # return user_dict, item_dict, data, len(user_dict)+1, len(item_dict)+1
        return user_dict, item_dict, data, max_user_id+1, max_item_id+1

    s_time = time.time()
    user_index, item_index, train_data, user_num, item_num = load_by_line(train_file)
        # print(prediction_i, prediction_i.shape)
    e_time = time.time()
    print('load_by_line: ',e_time-s_time)
    


    # user_num = train_data['user'].max() + 1
    # user_num = train_data['user'].apply(lambda x:int(x[1:])).max()+1
    # item_num = train_data['item'].max() + 1

    # train_data = train_data.values.tolist()

    # load ratings as a dok matrix

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float16)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    return train_data, train_mat, user_num, item_num

def load_all(train_file,test_file,neg_file,test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        train_file, 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float16)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(neg_file, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]

            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()

    return train_data, test_data, user_num, item_num, train_mat


'''
class UI_Data(IterableDataset):
        Description:
            Handle the UI interaction dataset .

        Example:
            the data format would like
            u1	i2	r1
            u2	i3	r4
            ...
            u10	i6	r10

    raise NotImplementedError

'''
class UI_List_Data(IterableDataset):
    '''
        Description:
            Handle the UI interaction dataset .

        Example:
            the data format would like
            u1	i2,i3,i4, ..., i_n

    '''
    def __init__(self, file_path, max_item_num, num_ng=5,  is_training=None):
        super(UI_List_Data, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        # self.features = features
        # self.num_item = num_item
        # self.train_mat = train_mat
        self.file_path = file_path
        self.num_ng = num_ng
        self.max_item_num = max_item_num
        self.is_training = is_training

        self.user_index, self.item_index = self.create_index(self.file_path)

    """
    @staticmethod
    def loadfile(filename):
        ''' load a file, return a generator. '''
        fp = open(filename, 'r')
        for i, line in enumerate(fp):
            yield line.strip('\r\n')
            if i % 100000 == 0:
                print ('loading %s(%s)' % (filename, i), file=sys.stderr)
        fp.close()
        print ('load %s succ' % filename, file=sys.stderr)


    def load_dataset(self, train_file, test_file):
        trainset_len = 0
        testset_len = 0

        for line in self.loadfile(train_file):
            user, movie, rating = line.split('\t')
            self.trainset.setdefault(user, {})
            self.trainset[user][movie] = int(rating)
            trainset_len += 1

        for line in self.loadfile(test_file):
            user, movie, rating = line.split('\t')
            self.testset.setdefault(user, {})
            self.testset[user][movie] = int(rating)
            testset_len += 1

        print ('split training set and test set succ', file=sys.stderr)
        print ('train set = %s' % trainset_len, file=sys.stderr)
        print ('test set = %s' % testset_len, file=sys.stderr)

    def line_proc(self, line):
        uid, iid_list = line.rstrip().split('\t')

        iid_list = eval(iid_list) # tuple

        for iid in iid_list:
            yield uid, str(iid)
    """

    def create_index(self, train_file, delimiter="\t"):
    # def get_list_index(file, delimiter="\t", target_field:List=[0]):

        ID_encoder = [dict(), dict()]
        for line in open(train_file,'r'):
            uid, iid_list = line.rstrip().split(delimiter)
            if uid not in ID_encoder[0]:
                ID_encoder[0][uid] = len(ID_encoder[0])
            
            for id in iid_list.split(','):
                if id not in ID_encoder[1]:
                   ID_encoder[1][id] = len(ID_encoder[1])
               
        return ID_encoder


    def __iter__(self):                        
       # file_iter = open(self.file_path)        
       # return map(self.line_proc, file_iter)
       neg_sample = random.sample(range(0, self.max_item_num), self.num_ng)

       for line in open(self.file_path, "r"):
            uid, iid_list = line.rstrip().split('\t')
            iid_list = eval(iid_list) # tuple


            for iid in iid_list:
                for n in range(self.num_ng):
                    # yield uid, iid, neg_sample[n]

                    yield int(uid), int(iid), int(neg_sample[n])
                    # iid = str(iid)
                    # n_iid = str(neg_sample[n])
                    # yield self.user_index[uid], self.item_index[iid], self.item_index[n_iid]

        


    # def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        # self.features_fill = []
        # for x in self.features:
            # u, i = x[0], x[1]
            # for t in range(self.num_ng):
                # j = np.random.randint(self.num_item)
                # while (u, j) in self.train_mat:
                    # j = np.random.randint(self.num_item)
                # self.features_fill.append([u, i, j])

   # def __len__(self):
        # return self.num_ng * len(self.features) if \
                # self.is_training else len(self.features)

    # def __getitem__(self, idx):
        # features = self.features_fill if \
                # self.is_training else self.features
        # user = features[idx][0]
        # item_i = features[idx][1]
        # item_j = features[idx][2] if \
                # self.is_training else features[idx][1]

        # return user, item_i, item_j 



class BPRData(Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
            
            train_data, item_num, train_mat, args.num_ng, True
        """
        self.features_fill = None
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u, i, j])

    def __len__(self):
        return self.num_ng * len(self.features) if \
                self.is_training else len(self.features)

    def __getitem__(self, idx):
        # assert self.features_fill is not None

        features = self.features_fill if \
                    self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]

        """
            if 
        """
        item_j = features[idx][2] if self.is_training \
                        else features[idx][1]
        return user, item_i, item_j 
		
