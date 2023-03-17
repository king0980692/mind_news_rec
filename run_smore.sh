#!/bin/bash

display_help() {
   echo "Arguments are required."
   echo "Example usage:"
   echo "	bash run_smore.sh \\"
   echo "		 bpr \\"
   echo "		 ml-1m \\"
   exit 1
}
[ -z "$1" ] && display_help
[ -z "$2" ] && display_help

smore=$1
data=$2
tr_data=exp/$data.train
te_data=exp/$data.test
emb=exp/$smore.$data.emb

if ! command -v $smore &> /dev/null
then
    echo "smore binary not found."
	echo "please ensure you have installed smore and add it into your \$PATH"
    exit 1
fi

set -xe

pre_process(){

	rm -f $tr_data
	# Interaction
	cat ./data/MIND-200k/train/behaviors.tsv \
	| cut -f 2,5 \
	| awk -F'\t' '{OFS=FS;head=$1;split($2,a," ");for(i=1; i<=length(a); i++) {split(a[i],b,"-");if(b[2]==1)print head,b[1],b[2] }}' \
	> $tr_data

	# History
	cat ./data/MIND-200k/train/behaviors.tsv \
	| cut -f 2,4 \
	| awk -F'\t' '{OFS=FS;head=$1;split($2,a," ");for(i=1; i<=length(a); i++)print head,a[i],1 }' \
	>> $tr_data

	# News -> subcategory
	cat ./data/MIND-200k/train/news.tsv \
	| awk 'OFS="\t"{print $1,$3,"1"}' \
	>> $tr_data

	#cat ./data/MIND-200k/dev/behaviors.tsv | cut -f 2,5 | awk -F'\t' '{OFS=FS;head=$1;split($2,a," ");for(i=1; i<=length(a); i++)print head,a[i] }' | tr '-' '\t' | cut -f 1,2 > $te_data
}


DIM=128
UPDATE=1000
WORKER=16
NEG=4

train_smore()
{
	#Train Model
	$smore -train $tr_data \
		   -save $emb \
		   -dimension $DIM \
		   -update_times $UPDATE \
		   -num_negative $NEG \
		   -worker $WORKER
}

## ---------------


pre_process
train_smore

python3 mind_rec.py --train ./data/MIND-200k/train/behaviors.tsv --test ./data/MIND-200k/dev/behaviors.tsv --embed ./exp/$smore.$data.emb --emb_dim $DIM

python3 evaluate.py --pred exp/$smore.$data.emb.ui.rec  --truth ./truth.txt

