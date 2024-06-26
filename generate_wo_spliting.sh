#!/bin/zsh
# Edit this line to include the exact .fasta filenames
declare -a datasets=("dataset")
encoded_path=$1
train_path="./output/train"
response_col="activity"

for ((i=1; i<=${#datasets[@]}; i++)); do
    # Add --balanced
    echo "[BASH] Running 'python3 scripts/generate_training.py -i $encoded_path/${datasets[i]}/-o $train_path/${datasets[i]}/ -t binary -r $response_col"
    python3 scripts/generate_training.py -i $encoded_path"/"${datasets[i]}"/" -o $train_path"/"${datasets[i]}"/" -t binary -r $response_col
done
