from collections import Counter
from pygments import token
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize

from transformers import AutoTokenizer, AutoModel

def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


def read_news(news_path, args, mode='train'):
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}
    word_cnt = Counter()

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, url, _, _ = splited
            update_dict(news_index, doc_id)

            # title = title.lower()
            # title = word_tokenize(title)
            title = tokenizer.tokenize(title)
            # title = title
            update_dict(news, doc_id, [title, category, subcategory])
            if mode == 'train':
                if args.use_category:
                    update_dict(category_dict, category)
                if args.use_subcategory:
                    update_dict(subcategory_dict, subcategory)
                word_cnt.update(title)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > args.filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        return news, news_index, category_dict, subcategory_dict, word_dict
    elif mode == 'test' or mode == 'val':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, args):
    news_num = len(news) + 1
    news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
    news_subcategory = np.zeros((news_num, 1), dtype='int32') if args.use_subcategory else None

    plm = AutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    for key in tqdm(news):
        title, category, subcategory = news[key]
        doc_index = news_index[key]

        '''
            truncate news title with length 20
            and encode the news title
        '''

        # for word_id in range(min(args.num_words_title, len(title))):

        # in order to add [CLS] and [SEP] token, so preserve two position
        for word_id in range(min(args.num_words_title-2, len(title))):

            '''
            if title[word_id] in word_dict:
                news_title[doc_index, word_id] = word_dict[title[word_id]]
            '''

            news_title[doc_index, word_id+1] = tokenizer.convert_tokens_to_ids(title[word_id])

        news_title[doc_index, 0] = tokenizer.convert_tokens_to_ids('[CLS]')
        news_title[doc_index, args.num_words_title-1] = tokenizer.convert_tokens_to_ids('[SEP]')


        if args.use_category:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.use_subcategory:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return news_title, news_category, news_subcategory
