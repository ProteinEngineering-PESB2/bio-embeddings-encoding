#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Error: Missing arguments."
  echo "Usage: ./run_encoding.sh [DATA_PATH] [SEQUENCE_COLUMN]"
  exit 1
fi

data_path=$1
# Almost all the times is "sequence"
sequence_col=$2

# seqvec is a weird one. It creates a 3D embed
declare -a unreduceable_models=("cpcprot" "seqvec")
declare -a models=("bepler" "cpcprot" "esm" "esm1b" "fasttext" "glove" "onehot" "plusrnn" "prottrans_albert" "prottrans_bert" "prottrans_t5bfd" "prottrans_xlnet_uniref100" "prottrans_t5xlu50" "word2vec")
# TODO add support to .fasta files 
find $data_path -type f -name "*.csv" | while IFS= read -r filename; do
    mkdir -p ${filename%.*}
    echo "[BASH] Using data from $filename"
    
    for model in ${models[@]}; do
        echo "[BASH] Running 'python3 ./scripts/run_bio_embeddings.py -i $filename -o ${filename%.*}/ -s $sequence_col -e $model --reduce'"
        python3 ./scripts/run_bio_embeddings.py -i $filename -o ${filename%.*}/ -s $sequence_col -e $model --reduce
        if ! [[ " $unreduceable_models[@] " =~ $model ]]; then
            echo "[BASH] Running 'python3 ./scripts/run_bio_embeddings.py -i $filename -o ${filename%.*}/ -s $sequence_col -e $model'"
            python3 ./scripts/run_bio_embeddings.py -i $filename -o ${filename%.*}/ -s $sequence_col -e $model
        fi
    done

    for i in $(seq 0 7); do
        echo "[BASH] Running 'python scripts/run_physicochemical_encoder.py -i $filename -o ${filename%.*}/ -g Group_$i -e scripts/cluster_encoders.csv'"
        python scripts/run_physicochemical_encoder.py -i $filename -o ${filename%.*}/ -g Group_$i -e scripts/cluster_encoders.csv
    done

    echo "[BASH] Running 'python scripts/run_one_hot.py -i $filename -o ${filename%.*}/ -s $sequence_col'"
    python scripts/run_one_hot.py -i $filename -o ${filename%.*}/ -s $sequence_col
done

