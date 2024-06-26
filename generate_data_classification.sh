#!/bin/zsh
# Edit this line to include the exact .fasta filenames
declare -a datasets=("dna_binding_d90,rna_binding_d90" "rna_binding_d90,non_nucleotidic_data_d90" "dna_binding_d90,rna_binding_d90,non_nucleotidic_data_d90" "nucleotidic_data_d90,non_nucleotidic_data_d90" "single,double")
declare -a values=("0,1" "0,1" "0,1,2" "0,1" "0,1")
encoded_path=$1
merged_path="./output/merged"
split_path="./output/split"
train_path="./output/train"
sequence_col="sequence"

for ((i=1; i<=${#datasets[@]}; i++)); do
    dataset_vs=${datasets[i]//,/vs}
    # Add --balanced
    echo "[BASH] Running 'python3 scripts/generate_data.py -i $encoded_path -c ${datasets[i]} -o $merged_path -v ${values[i]} -s ${sequence_col}"
    python3 scripts/generate_data.py -i $encoded_path -c ${datasets[i]} -o $merged_path -v ${values[i]} -s ${sequence_col}
    echo "[BASH] Running 'python3 scripts/split_data.py -i $merged_path -f $dataset_vs -o $split_path -g 10'"
    python3 scripts/split_data.py -i $merged_path -f $dataset_vs -o $split_path -g 10
    echo "[BASH] Running 'python3 scripts/generate_training.py -i $split_path"/"$dataset_vs"/"-o $train_path"/"$dataset_vs"/" -d 80/20'"
    python3 scripts/generate_training.py -i $split_path"/"$dataset_vs"/" -o $train_path"/"$dataset_vs"/" -d 80/20
done
