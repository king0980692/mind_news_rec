set -xe

raw_data=$1
total=$(wc -l < $raw_data)
time_cut=$(echo | awk -v x="$total" '{ print int(x*0.8); }')

name=exp/$2
sep=$3
header=$(($4+1))

mkdir -p exp
cat $raw_data | tail -n +"$header" | sed "s/$sep/\t/g" | sort -k 4 | awk -v x=$time_cut -v o="$name" -F "\t" 'OFS="\t"{if(NR<x) {print "u"$1,$2,"1" > o".train"} else {print "u"$1,$2,"1" > o".test"}}' 
