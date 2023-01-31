#!/bin/zsh
sequence_col="sequence"
# This variable can export more than 1 column separated by comma.
# export_cols="id,target"
export_cols="id"
data_path=$1
# Pending Download prottrans_xlnet_uniref100
declare -a models=("one_hot_encoding" "cpcprot" "fasttext" "glove" "plus_rnn" "prottrans_albert_bfd" "prottrans_bert_bfd" "seqvec" "word2vec" "bepler" "esm" "esm1b")
for data in "$data_path"/*.csv
do
    mkdir -p ${data::-4}
    echo "using data from $data"
    for model in ${models[@]};
    do
        python3 ./scripts/encoding_sequences.py $data ${data::-4}/ $sequence_col $export_cols $model
    done
done
