#!/bin/zsh
# Edit this line to include the exact .fasta filenames
declare -a datasets=("dataset")
encoded_path=$1
split_path="./output/split"
train_path="./output/train"
response_col="activity"

for ((i=1; i<=${#datasets[@]}; i++)); do
    # Add --balanced
    echo "[BASH] Running 'python3 scripts/split_data.py -i $encoded_path -f ${datasets[i]} -o $split_path -g 10 -t binary'"
    python3 scripts/split_data.py -i $encoded_path -f ${datasets[i]} -o $split_path -g 10 -t binary
    echo "[BASH] Running 'python3 scripts/generate_training.py -i $split_path/${datasets[i]}/-o $train_path/${datasets[i]}/ -d 80/20 -t binary -r $response_col"
    python3 scripts/generate_training.py -i $split_path"/"${datasets[i]}"/" -o $train_path"/"${datasets[i]}"/" -d 80/20 -t binary -r $response_col
done
