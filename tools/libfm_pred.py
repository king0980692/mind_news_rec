import sys

def write_result_file(probs, libfm_result_file):
    k = 0
    with open('../../MIND-%s/test/behaviors.tsv' % dataset, 'r', encoding='utf-8') as behaviors_f:
        with open(libfm_result_file, 'w', encoding='utf-8') as f:
            for i, line in enumerate(behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                num = len(impressions.strip().split(' '))
                scores = []
                for j in range(num):
                    scores.append([probs[k], j])
                    k += 1
                scores.sort(key=lambda x: x[0], reverse=True)
                result = [0 for _ in range(num)]
                for j in range(num):
                    result[scores[j][1]] = j + 1
                f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    assert len(probs) == k, str(len(probs)) + ' - ' + str(k)


probs = []
with open('exp/ui.pred', 'r', encoding='utf-8') as f:
    for line in f:
        if len(line.strip()) > 0:
            probs.append(float(line.strip()))

write_result_file(probs, 'test/res/libfm/%d/libfm.txt' % run_index)

with open('test/ref/truth.txt', 'r', encoding='utf-8') as truth_f, open('test/res/libfm/%d/libfm.txt' % run_index, 'r', encoding='utf-8') as res_f:
auc, mrr, ndcg, ndcg10 = scoring(truth_f, res_f)
print('AUC =', auc)
print('MRR =', mrr)
print('nDCG@5 =', ndcg)
print('nDCG@10 =', ndcg10)
with open('results/libfm/#%d-test' % run_index, 'w', encoding='utf-8') as f:
    f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg) + '\t' + str(ndcg10) + '\n')

