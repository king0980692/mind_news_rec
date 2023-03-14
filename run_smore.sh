set -xe
python3 mind_rec.py --train ../MIND-large/train/behaviors.tsv --test ../MIND-large/dev/behaviors.tsv --embed ./exp/bpr.mind.emb --emb_dim 128
python3 evaluate.py --pred ./exp/bpr.mind.emb.ui.rec --truth ./truth.txt


