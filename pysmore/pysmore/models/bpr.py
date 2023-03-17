from tqdm.auto import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.kwargs = {'user_num': user_size, 'item_num': item_size, "emb_dim":dim}

        self.W = nn.Parameter(torch.empty(user_size, dim))
        self.H = nn.Parameter(torch.empty(item_size, dim))
        nn.init.xavier_normal_(self.W.data)
        nn.init.xavier_normal_(self.H.data)
        self.weight_decay = weight_decay

    def save_embedding(self, rv_user_mapping, rv_item_mapping, saved_path):

        print(f"\n\nSaving Embedding to {saved_path}")
        user_emb = self.W.detach()
        item_emb = self.H.detach()

        output = []
        # print(f"\tprocess user embedding")
        for _i in range(0, user_emb.shape[0]): 
            u_id = str(rv_user_mapping[_i])
            u_vec = user_emb[_i].tolist()
            vec_str = " ".join([ str(_v) for _v in u_vec ])

            output.append(u_id+"\t"+vec_str+"\n")

        # print(f"\tprocess item embedding")
        for _i in range(0, item_emb.shape[0]): 
            i_id = str(rv_item_mapping[_i])
            i_vec = item_emb[_i].tolist()
            vec_str = " ".join([ str(_v) for _v in i_vec ])

            output.append(i_id+"\t"+vec_str+"\n")

        with open(saved_path, 'w') as f:
            f.writelines(output)

    def forward(self, u, i, j):
        """Return loss value.
        
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        
        Returns:
            torch.FloatTensor
        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

class BPR2(nn.Module):
    def __init__(self, user_num, item_num, emb_dim=32):
        super(BPR, self).__init__()
        self.kwargs = {'user_num': user_num, 'item_num': item_num, "emb_dim":emb_dim}
        """
        user_num: number of users;
        item_num: number of items;
        emb_dim: number of latent factors.
        """		
        self.embed_user = nn.Embedding(user_num, emb_dim)
        self.embed_item = nn.Embedding(item_num, emb_dim)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j
