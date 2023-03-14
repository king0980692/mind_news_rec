import os
from tqdm import tqdm
import argparse
import pickle
import nltk
import re
from pathlib import Path

from nltk.tokenize import TweetTokenizer

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')
# model = SentenceTransformer('all-roberta-large-v1')

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
        print("tokenize file: ", news_file)
        with open(news_file, 'r', encoding='utf-8') as news_f:
            for line in tqdm(news_f.readlines()):

                n_id, cat, sub_cat, title, abstr, *_ = line.split('\t')
                if n_id not in news_ID_set:
                    news_word_counter = Counter()

                    # tokenize by nltk
                    words = tokenizer.tokenize((title + ' ' + abstr).lower())

                    # tokenize by regex
                    # words = pat.findall((title + ' ' + abstract).lower()) if tokenizer == 'MIND' else word_tokenize((title + ' ' + abstract).lower())

                    # filter with stopwords & trans numba to NUMTOKEN
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

    print("SBERT encoding ...")

    # tfidf_matrix = vectorizer.fit_transform(sentence_corpus)
    sentence_embeddings = model.encode(sentence_corpus)

    user_history_dict = {}
    for data_type in Path(args.data).iterdir():
        beh_file = data_type / "behaviors.tsv"
        print("sbert - Encode: ", beh_file)
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

    return news_dict, sentence_embeddings, user_history_dict

def generate_user_tfidf(news_tfidf, user_history_dict):
    user_tfidf = {}
    for user_ID in user_history_dict:
        tfidf = {}
        for news_ID in user_history_dict[user_ID]:
            _news_tfidf = news_tfidf[news_ID]
            for word in _news_tfidf:
                if word not in tfidf:
                    tfidf[word] = _news_tfidf[word]
                else:
                    tfidf[word] = max(tfidf[word], _news_tfidf[word])

        user_tfidf[user_ID] = tfidf
        # user_tfidf[user_ID] = " ".join("{}:{}".format(key,item) for key,item in tfidf.items())


    user_tfidf['total_dim'] = news_tfidf.get_shape()[1]
    user_tfidf['cold_hist'] = {}

    return user_tfidf

def generate_news_embeddings(news_dict, sentence_embeddings):
    print('trans every news token to tfidf encoding')

    news_embed = {}
    news_embed_str = {}
    # iterate all news
    for news_ID in tqdm(news_dict):
        news_matrix = sentence_embeddings[news_dict[news_ID]]
        # _embed = {}

        # for word_index in news_matrix.indices:
            # _embed[word_index] = news_matrix[0, word_index]

        news_embed[news_ID] = news_matrix

        news_embed_str[news_ID] = " ".join("{}:{:8f}".format(id,emb) for id, emb in enumerate(news_matrix) )

    # news_tfidf['total_dim'] = tfidf_matrix.get_shape()[1]
    news_embed['cold_news'] = {}

    return news_embed, news_embed_str


if __name__ == '__main__':
    news_dict, sentence_embeddings, user_history_dict = build_meta()

    news_tfidf, news_tfidf_str = generate_news_embeddings(news_dict, sentence_embeddings)
    with open(os.path.join(args.out, 'news_tfidf.pkl'), 'wb') as p:
        pickle.dump(news_tfidf_str, p)

    # user_tfidf = generate_user_tfidf(news_tfidf, user_history_dict)
    # with open(os.path.join(args.out, 'user_tfidf.pkl'), 'wb') as p:
        # pickle.dump(user_tfidf, p)


