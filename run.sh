#!/bin/zsh
# Almost all the times is "sequence"
sequence_col=$2
data_path=$1

# seqvec is a weird one. It creates a 3D 
declare -a unreduceable_models=("cpcprot" "seqvec")
#declare -a models=("bepler" "cpcprot" "esm" "esm1b" "fasttext" "glove" "onehot" "plusrnn" "prottrans_albert" "prottrans_bert" "prottrans_t5bfd" "prottrans_xlnet_uniref100" "prottrans_t5xlu50" "seqvec" "word2vec")
declare -a models=("bepler" "cpcprot")
for filename in "$data_path"/*.*; do
    mkdir -p ${filename%.*}
    echo "using data from $filename"
    
    for model in ${models[@]}; do
        python3 ./scripts/encoding_sequences.py -i $filename -o ${filename%.*}/ -s $sequence_col -e $model --reduce
        if ! [[ " $unreduceable_models[@] " =~ $model ]]; then
            python3 ./scripts/encoding_sequences.py -i $filename -o ${filename%.*}/ -s $sequence_col -e $model
        fi
    done
done

