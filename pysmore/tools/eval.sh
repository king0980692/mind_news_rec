#!/bin/bash
set -ex
smore=$1
DATA=$2
mkdir -p result
#python3 eval.py exp/$smore.emb.ui.rec all
python3 tools/eval.py ./exp/$smore.$DATA.emb.ui.rec  5  >  result/$DATA.$smore
python3 tools/eval.py ./exp/$smore.$DATA.emb.ii.rec  5  >> result/$DATA.$smore
python3 tools/eval.py ./exp/$smore.$DATA.emb.ui.rec 10  >> result/$DATA.$smore
python3 tools/eval.py ./exp/$smore.$DATA.emb.ii.rec 10  >> result/$DATA.$smore
python3 tools/eval.py ./exp/$smore.$DATA.emb.ui.rec all >> result/$DATA.$smore
python3 tools/eval.py ./exp/$smore.$DATA.emb.ii.rec all >> result/$DATA.$smore

cat result/$DATA.$smore
