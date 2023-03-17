import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import random
from collections import deque

# from ..utils.alias_method import AliasTable
from ..utils.c_alias_method import AliasTable
from ..utils.alias_method import AliasTable

# Actually is a `multiple worker sampler` class TripletUniformPair(IterableDataset):
class TripletUniformPair(IterableDataset):
    def __init__(self, num_item, user_list, rate_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.rate_list = rate_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = get_worker_info()

        # Assume every epoch will update whole explict interaction
        self.sample_size = self.num_epochs * len(self.pair) # update every pair num_epochs times
        

        self.sample_index_queue = deque([])
        self.seed = 0
        if worker_info is not None: #  multiple worker
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:                       #  single worker
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.sample_size:
            raise StopIteration

        # """
        ## Maintain the `index queueing` of pair
        # If `sample_index_queue` is used up, replenish this list.
        while len(self.sample_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            # only multiple worker will enter this section
            if self.start_list_index is not None: 
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.sample_index_queue.extend(index_list)

        # """
        ## Sampling 

        self.index += self.num_workers
        return list(self._sample(self.sample_index_queue.popleft()))
        # return result

    def _sample(self, sidx):

        u = self.pair[sidx][0]
        i = self.pair[sidx][1]
        j = np.random.randint(self.num_item)

        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)

        yield u, i, j

class TripletWeightedPair(IterableDataset):
    def __init__(self, num_item, user_list, rate_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.rate_list = rate_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

        self.contexts = []
        self.vertex_sampler, self.vertex_uniform_sampler = AliasTable(), AliasTable()

        self.context_sampler, self.context_uniform_sampler = AliasTable(), AliasTable()
        self.negative_sampler = AliasTable()

        self.build_sampler()

        # self.triplet_len = len(self.pair)//(4*16)
        # self.triplet_len = len(self.pair)
        # print('triplet_len: ', self.triplet_len)
        # self.weighted_triplets= self.draw_triplet_list(self.triplet_len)

    def build_sampler(self):
        print('Build VC-Sampler')

        vertex_distribution          = [0.] * len(self.user_list)
        vertex_uniform_distribution  = [0.] * len(self.user_list)
        context_uniform_distribution = [0.] * self.num_item
        negative_distribution        = [0.] * self.num_item
        context_distribution         = []


        for u in trange(len(self.user_list)):
            # if len(self.user_list[u]) == 0 :
                # print(u," missing")
                # continue

            context_distribution.clear()
            for item, rate in zip(self.user_list[u], self.rate_list[u]):
                assert len(self.user_list[u]) == len(self.  rate_list[u]), 'item_list & rate_list not match'

                vertex_distribution[u] += rate
                negative_distribution[item] += rate
                context_distribution.append(rate)
                vertex_uniform_distribution[u] = 1.0
                context_uniform_distribution[item] = 1.0

                # accumlate the vertex be connected with
                self.contexts.append(item)

            # end-for with iterate rate & item
            self.context_sampler.append(context_distribution, 1.0)

        print('\tCreate vertex sampler')
        self.vertex_sampler.append(vertex_distribution, 1.0)
        print('\tCreate vertex sampler done')

        print('\tCreate vertex uniform sampler')
        self.vertex_uniform_sampler.append(vertex_uniform_distribution, 1.0)
        print('\tCreate vertex uniform sampler done')

        print('\tCreate context uniform sampler')
        self.context_uniform_sampler.append(context_uniform_distribution, 1.0)
        print('\tCreate context uniform sampler done')

        print('\tCreate negative sampler')
        self.negative_sampler.append(negative_distribution, 0.75)
        print('\tCreate negative sampler done') #end-for with iterate user
        print('Build VC-Sampler done')

    def draw_vertex(self):
        return self.vertex_sampler.draw()

    def draw_context(self, v_id):
        return self.contexts[self.context_sampler.draw_by_given(v_id)]

    def draw_context_uniform(self):
        return self.context_uniform_sampler.draw()

    def draw_triplet_list(self, times):
        print("Create triplet list ...")
        out = []

        for _ in trange(times):
            u = self.vertex_sampler.draw()
            i = self.draw_context(u)
            j = self.draw_context_uniform()
            out.append((u,i,j))

        print("Create triplet list done")
        return out

    def draw_neg_triplet_list(self, times, neg_times):

        out = []

        for _ in range(times):
            u = self.vertex_sampler.draw()
            for _n in range(neg_times):
                i = self.draw_context(u)
                j = self.draw_context_uniform()
                out.append((u,i,j))

        return out

    def __iter__(self):
        worker_info = get_worker_info()

        # Assume every epoch will update whole explict interaction
        self.sample_size = self.num_epochs * len(self.pair)

        self.sample_index_queue = deque([])
        self.seed = 0
        if worker_info is not None: #  multiple worker
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:                       #  single worker
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.sample_size:
            raise StopIteration

        """
        ## Maintain the `index queueing` of pair # If `sample_index_queue` is used up, replenish this list.
        while len(self.sample_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            # only multiple worker will enter this section
            if self.start_list_index is not None: 
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.sample_index_queue.extend(index_list)

        """
        ## Sampling 

        # self.index += self.num_workers
        # result = self._sample(self.sample_index_queue.popleft())
        return list(self._sample_generator())
        # return result



    def _sample_generator(self):

        u = self.draw_vertex()

        # u = self.weighted_triplets[sidx][0]
        # i = self.weighted_triplets[sidx][1]
        # j = self.weighted_triplets[sidx][2]

        for _n in range(5):
            i = self.draw_context(u)
            j = self.draw_context_uniform()
            yield u, i, j


class TripletWeightedPair2(IterableDataset):
    def __init__(self, num_item, user_list, rate_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.rate_list = rate_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs


        self.contexts = []
        self.vertex_sampler, self.vertex_uniform_sampler = AliasTable(), AliasTable()

        self.context_sampler, self.context_uniform_sampler = AliasTable(), AliasTable()
        self.negative_sampler = AliasTable()

        self.build_sampler()

        # self.triplet_len = len(self.pair)//(4*16)
        self.triplet_len = len(self.pair)
        print('triplet_len: ', self.triplet_len)
        # self.weighted_triplets= self.draw_triplet_list(self.triplet_len)
        self.weighted_triplets= self.draw_neg_triplet_list(self.triplet_len, 5)

    def build_sampler(self):
        print('Build VC-Sampler')

        vertex_distribution          = [0.] * len(self.user_list)
        vertex_uniform_distribution  = [0.] * len(self.user_list)
        context_uniform_distribution = [0.] * self.num_item
        negative_distribution        = [0.] * self.num_item
        context_distribution         = []


        for u in trange(len(self.user_list)):
            # if len(self.user_list[u]) == 0 :
                # print(u," missing")
                # continue

            context_distribution.clear()
            for item, rate in zip(self.user_list[u], self.rate_list[u]):
                assert len(self.user_list[u]) == len(self.  rate_list[u]), 'item_list & rate_list not match'

                vertex_distribution[u] += rate
                negative_distribution[item] += rate
                context_distribution.append(rate)
                vertex_uniform_distribution[u] = 1.0
                context_uniform_distribution[item] = 1.0

                # accumlate the vertex be connected with
                self.contexts.append(item)

            # end-for with iterate rate & item
            self.context_sampler.append(context_distribution, 1.0)

        print('\tCreate vertex sampler')
        self.vertex_sampler.append(vertex_distribution, 1.0)
        print('\tCreate vertex sampler done')

        print('\tCreate vertex uniform sampler')
        self.vertex_uniform_sampler.append(vertex_uniform_distribution, 1.0)
        print('\tCreate vertex uniform sampler done')

        print('\tCreate context uniform sampler')
        self.context_uniform_sampler.append(context_uniform_distribution, 1.0)
        print('\tCreate context uniform sampler done')

        print('\tCreate negative sampler')
        self.negative_sampler.append(negative_distribution, 0.75)
        print('\tCreate negative sampler done') #end-for with iterate user
        print('Build VC-Sampler done')



    def draw_vertex(self):
        return self.vertex_sampler.draw()

    def draw_context(self, v_id):
        return self.contexts[self.context_sampler.draw_by_given(v_id)]

    def draw_context_uniform(self):
        return self.context_uniform_sampler.draw()

    def draw_triplet_list(self, times):
        print("Create triplet list ...")
        out = []

        for _ in trange(times):
            u = self.vertex_sampler.draw()
            i = self.draw_context(u)
            j = self.draw_context_uniform()
            out.append((u,i,j))

        print("Create triplet list done")
        return out

    def draw_neg_triplet_list(self, times, neg_times):
        out = []

        print("Create triplet list ...")
        for _ in trange(times):
            u = self.vertex_sampler.draw()
            i = self.draw_context(u)
            for _n in range(neg_times):
                j = self.draw_context_uniform()
                out.append((u,i,j))
        print("Create triplet list done")

        return out
    def __iter__(self):
        worker_info = get_worker_info()

        # Assume every epoch will update whole explict interaction
        self.sample_size = self.num_epochs * len(self.pair)

        self.sample_index_queue = deque([])
        self.seed = 0
        if worker_info is not None: #  multiple worker
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:                       #  single worker
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.sample_size:
            raise StopIteration

        # """
        ## Maintain the `index queueing` of pair # If `sample_index_queue` is used up, replenish this list.
        while len(self.sample_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            # only multiple worker will enter this section
            if self.start_list_index is not None: 
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.sample_index_queue.extend(index_list)

        # """
        ## Sampling 

        self.index += self.num_workers
        result = self._sample(self.sample_index_queue.popleft())
        # result = self._sample(random.randint(0,len(self.pair)))
        return result

    def _sample(self, sidx):

        return self.weighted_triplets[sidx]
