import os
from tqdm import tqdm
import argparse
import pickle
import re
from pathlib import Path
import numpy as np

import nltk
from nltk.tokenize import TweetTokenizer

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer


parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--out')
args = parser.parse_args()

tokenizer = TweetTokenizer()

pat = re.compile(r"[\w]+|[.,!?;|]")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False




def build_meta():
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words |= set(['.', ',', '\t', '\n', '\'', '\"', '?', '!', ';', ' ', '\n', '\t', '\r'])
    # with open('NLTK_stop_words', 'r', encoding='utf-8') as stop_words_f:
        # for line in stop_words_f:
            # if len(line.strip()) > 0:
                # stop_words.add(line.strip())

    tokenizer = TweetTokenizer()
    news_ID_set = set()
    word_cnt = Counter()

    for data_type in Path(args.data).iterdir():
        news_file = data_type / "news.tsv"
        print("tfidf - Encode: ", news_file)
        with open(news_file, 'r', encoding='utf-8') as news_f:
            for line in tqdm(news_f.readlines()):

                n_id, cat, sub_cat, title, abstr, *_ = line.split('\t')
                if n_id not in news_ID_set:
                    news_word_counter = Counter()

                    # tokenize by nltk
                    words = tokenizer.tokenize((title + ' ' + abstr).lower())

                    # tokenize by regex
                    # words = pat.findall((title + ' ' + abstract).lower()) if tokenizer == 'MIND' else word_tokenize((title + ' ' + abstract).lower())

                    # filter with stopwords
                    for id, word in enumerate(words):
                        if word not in stop_words:
                            if is_number(word):
                                word = 'NUMTOKEN'
                            news_word_counter[word] += 1

                    for word in news_word_counter:
                        word_cnt[word] += 1
                    news_ID_set.add(n_id)

    news_ID_dict = {}
    user_ID_dict = {}
    news_dict = {}
    sentence_corpus = []
    vectorizer = TfidfVectorizer()


    for data_type in Path(args.data).iterdir():
        news_file = data_type / "news.tsv"
        with open(news_file, 'r', encoding='utf-8') as news_f:
            for line in tqdm(news_f.readlines()):
                n_id, cat, sub_cat, title, abstr, *_ = line.split('\t')
                if n_id not in news_dict:
                    words = tokenizer.tokenize((title + ' ' + abstr).lower())
                    sentence = ''
                    for word in words:
                        if word not in stop_words and word_cnt[word] > 1:
                            if is_number(word):
                                word = 'NUMTOKEN'
                            sentence += word + ' '
                    sentence_corpus.append(sentence)
                    news_dict[n_id] = len(news_dict)
                if n_id not in news_ID_dict:
                    news_ID_dict[n_id] = len(news_ID_dict)
    tfidf_matrix = vectorizer.fit_transform(sentence_corpus)
    user_history_dict = {}
    for data_type in Path(args.data).iterdir():
        beh_file = data_type / "behaviors.tsv"
        print("tfidf - Encode: ", beh_file)
        with open(beh_file, 'r', encoding='utf-8') as behaviors_f:
            for line in tqdm(behaviors_f.readlines()):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if user_ID not in user_history_dict:
                    if len(history) > 0:
                        user_history_dict[user_ID] = history.split(' ')
                    else:
                        user_history_dict[user_ID] = {}

    # with open('news_ID-%s.pkl' % dataset, 'wb') as news_ID_f:
        # pickle.dump(news_ID_dict, news_ID_f)
    # with open('user_ID-%s.pkl' % dataset, 'wb') as user_ID_f:
        # pickle.dump(user_ID_dict, user_ID_f)
    # with open('offset-%s.txt' % dataset, 'w', encoding='utf-8') as f:
        # f.write(str(len(news_ID_dict)) + '\n')
        # f.write(str(len(user_ID_dict)) + '\n')
        # f.write(str(len(vectorizer.get_feature_names())) + '\n')

    return news_dict, tfidf_matrix, user_history_dict

def generate_user_tfidf(news_tfidf, user_history_dict):
    print("generate user tfidf...")
    user_tfidf = {}
    max_dim = news_tfidf['total_dim']
    for user_ID in tqdm(user_history_dict):
        # tfidf = {}
        tfidf = np.zeros((1, max_dim), dtype=np.float32)
        for news_ID in user_history_dict[user_ID]:
            _news_tfidf = news_tfidf[news_ID]

            for word in np.where( _news_tfidf > 0)[1]:
                if tfidf[0, word] == 0:
                    tfidf[0, word] = _news_tfidf[0, word]
                else:
                    tfidf[0,word] = max(tfidf[0, word], _news_tfidf[0, word])

        user_tfidf[user_ID] = tfidf
        # user_tfidf[user_ID] = " ".join("{}:{:.6f}".format(key+offset, item) for key,item in tfidf.items())

    # user_tfidf['total_dim'] = news_tfidf.get_shape()[1]
    user_tfidf['cold_hist'] = ""

    return user_tfidf

def generate_news_tfidf(news_dict, tfidf_matrix):
    print('trans every news token to tfidf encoding')

    news_tfidf = {}
    news_tfidf_str = {}

    news_tfidf['total_dim'] = tfidf_matrix.get_shape()[1]

    # iterate all news
    for news_ID in tqdm(news_dict):
        news_matrix = tfidf_matrix[news_dict[news_ID]]
        # tfidf = {}

        tfidf = np.zeros((1, news_tfidf['total_dim']))

        for word_index in news_matrix.indices:
            tfidf[0, word_index] = news_matrix[0, word_index]

        news_tfidf[news_ID] = tfidf


        # news_tfidf_str[news_ID] = " ".join("{}:{:.6f}".format(key,item) for key,item in tfidf.items())

    # news_tfidf_str['total_dim'] = tfidf_matrix.get_shape()[1]
    news_tfidf_str['cold_news'] = ""
    news_tfidf['cold_news'] = ""

    return news_tfidf, news_tfidf_str


if __name__ == '__main__':
    news_dict, tfidf_matrix, user_history_dict = build_meta()

    news_tfidf, news_tfidf_str = generate_news_tfidf(news_dict, tfidf_matrix)
    with open(os.path.join(args.out, 'news_tfidf.pkl'), 'wb') as p:
        pickle.dump(news_tfidf_str, p)

    user_tfidf = generate_user_tfidf(news_tfidf, user_history_dict)
    with open(os.path.join(args.out, 'user_tfidf.pkl'), 'wb') as p:
        pickle.dump(user_tfidf, p)
