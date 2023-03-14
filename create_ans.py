from tqdm import tqdm
with open('../MIND-large/dev/behaviors.tsv', 'r', encoding='utf-8') as test_f:
    with open('./truth.txt', 'w', encoding='utf-8') as truth_f:
        for test_ID, line in tqdm(enumerate(test_f)):
            impression_ID, user_ID, time, history, impressions = line.split('\t')
            labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
            truth_f.write(('' if test_ID == 0 else '\n') + str(test_ID + 1) + ' ' + str(labels).replace(' ', ''))

