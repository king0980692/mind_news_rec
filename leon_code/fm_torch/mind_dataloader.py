import numpy as np
from sklearn.metrics import d2_tweedie_score
import torch.utils.data
from tqdm import tqdm
from sklearn import preprocessing
import pickle

class MIND_Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, news_map_path, u_enc, n_enc, cat_enc, scat_enc, is_train=True):
        # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header)

        self.slice_list = []
        self.news_map = pickle.load(open(news_map_path,'rb'))

        self.n_tfidf_map = pickle.load(open("./exp/news_tfidf.pkl", "rb"))
        self.u_tfidf_map = pickle.load(open("./exp/user_tfidf.pkl", "rb"))
        
        self.User_encoder = u_enc
        self.News_encoder = n_enc
        self.cat_encoder = cat_enc
        self.scat_encoder = scat_enc

        self.is_train = is_train
        if self.is_train:
            self.items, self.targets, self.field_dims = self.build_ID(dataset_path)
        else:
            self.items, self.targets = self.build_ID(dataset_path)
        
        # ID
        # self.items = data[:, 1].astype(np.int) - 1  # -1 because ID begins from 1

        # Score
        # self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)

        # Count every field's max id as its dimension
        # self.field_dims = np.max(self.items, axis=0) + 1

        

        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)


    def build_ID(self, dataset_path):

        u_ids = []
        n_ids = []
        cat_ids = []
        scat_ids = []
        data = []
        labels = []

        n_tfidfs = []
        u_tfidfs = []

        with open(dataset_path) as f:
            for line in tqdm(f.readlines()):
                imp_id, u_id, imp_time, history, imprs = line.split("\t")
                
                imprs = imprs.split()

                self.slice_list.append(len(imprs))

                for impr in imprs:

                    # his_ids.append(history.split())

                    # data.append([u_id,impr[:-2]])
                    n_id = impr[:-2]
                    u_ids.append(u_id)
                    n_ids.append(n_id) # n_id

                    cat_ids.append(self.news_map[n_id]['cat'])
                    scat_ids.append(self.news_map[n_id]['sub_cat'])
                    import IPython;IPython.embed(colors="linux");exit(1) 

                    n_tfidfs.append(self.n_tfidf_map[n_id])
                    n_tfidfs.append(self.u_tfidf_map[u_id])


                    if self.is_train:
                        labels.append(impr[-1])


        uniq_u_dim = len(set(u_ids))
        uniq_n_dim = len(set(n_ids))
        uniq_cat_dim = len(set(cat_ids))
        uniq_scat_dim = len(set(scat_ids))

        uniq_tfidf_dim = self.n_tfidf_map['total_dim']

        if self.is_train:
            u_ids = self.User_encoder.fit_transform(np.array(u_ids).reshape(-1,1))
            n_ids = self.News_encoder.fit_transform(np.array(n_ids).reshape(-1,1))

            # cat_ids = self.cat_encoder.fit_transform(np.array(cat_ids).reshape(-1,1))
            # scat_ids = self.scat_encoder.fit_transform(np.array(scat_ids).reshape(-1,1))

            # items = np.hstack([u_ids, n_ids, cat_ids, scat_ids]).astype(np.int_)
            items = np.hstack([u_ids, n_ids, n_tfidfs, u_tfidfs]).astype(np.int_)

            # uids = np.array(u_ids)
            # nids = np.array(n_ids)
            labels = np.array(labels, dtype=np.int8)

            # max_dim = np.array([uniq_u_dim, uniq_n_dim, uniq_cat_dim, uniq_scat_dim])
            max_dim = np.array([uniq_u_dim, uniq_n_dim, uniq_tfidf_dim, uniq_tfidf_dim])
                        
            return items, labels, max_dim

        else:
            # import bisect

            # le_classes = self.User_encoder.classes_.tolist()
            # bisect.insort_left(le_classes, '<unknown>')
            # self.User_encoder.classes_ = le_classes
            
            # ue_dict = dict(zip(self.User_encoder.classes_, self.User_encoder.transform(self.User_encoder.classes_)))

            # ne_dict = dict(zip(self.News_encoder.classes_, self.News_encoder.transform(self.News_encoder.classes_)))


                
            u_ids = self.User_encoder.transform(np.array(u_ids).reshape(-1,1))
            n_ids = self.News_encoder.transform(np.array(n_ids).reshape(-1,1))

            u_ids[u_ids<0] = 0
            n_ids[n_ids<0] = 0

            # cat_ids = self.cat_encoder.transform(np.array(cat_ids).reshape(-1,1))
            # scat_ids = self.scat_encoder.transform(np.array(scat_ids).reshape(-1,1))

            # cat_ids[cat_ids<0] = 0
            # scat_ids[scat_ids<0] = 0
            
            items = np.hstack([u_ids, n_ids, n_tfidfs, u_tfidfs]).astype(np.int_)
            # items = np.hstack([u_ids, n_ids, cat_ids, scat_ids]).astype(np.int_)

            # uids = np.array(u_ids)
            # nids = np.array(n_ids)
            labels = np.zeros((items.shape[0],1), dtype=np.int8)

            # max_dim = np.array([uniq_u_dim, uniq_n_dim])
            return items, labels
        


    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

