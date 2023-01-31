#!/bin/zsh
# Changes pending DO NOT use

declare -a models=("fasttext" "plus_rnn" "prottrans_bert_bfd" "bepler" "esm1b")
for model in ${models[@]};
do
  ./venv/bin/python ./scripts/encoding_sequences_npz.py ../kegg-api-data-tools/data-david/sequences.csv ../kegg-api-data-tools/data-david/encoded/ sequence_aa sequence_id $model
done
