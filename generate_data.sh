#!/bin/zsh
declare -a datasets=("dna,non_nucleotidic" "rna,non_nucleotidic" "dna,rna,non_nucleotidic" "nucleotidic,non_nucleotidic" "single,double")
declare -a values=("0,1" "0,1" "0,1,2" "0,1" "0,1")
encoded_path="./data"
merged_path="./output/merged"
split_path="./output/split"
train_path="./output/train"

for ((i=0; i<${#datasets[@]}; i++)); do
    dataset_vs=${datasets[i]//,/vs}
    # Add --balanced
    #echo "[BASH] Running 'python3 scripts/generate_data.py -i $encoded_path -c ${datasets[i]} -o $merged_path -v ${values[i]}'"
    #python3 scripts/generate_data.py -i $encoded_path -c ${datasets[i]} -o $merged_path -v ${values[i]}
    #echo "[BASH] Running 'python3 scripts/split_data.py -i $merged_path -f $dataset_vs -o $split_path -g 10'"
    #python3 scripts/split_data.py -i $merged_path -f $dataset_vs -o $split_path -g 10
    echo "[BASH] Running 'python3 scripts/generate_training.py -i $split_path"/"$dataset_vs"/"-o $train_path"/"$dataset_vs"/" -d 70/20/10'"
    python3 scripts/generate_training.py -i $split_path"/"$dataset_vs"/" -o $train_path"/"$dataset_vs"/" -d 70/20/10
done




