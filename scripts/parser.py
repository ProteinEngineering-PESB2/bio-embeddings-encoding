import pandas as pd
from Bio import SeqIO

def FastaParse(fasta_file: str, col_name: str):
    sequences = []
    ids = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))

    df = pd.DataFrame({col_name: sequences})
    return df

def FastaSave(sequences: list, filename: str):
    with open(filename, 'w') as file:
        for idx, sequence in enumerate(sequences):
            file.write(f">{idx}\n")
            file.write(sequence + '\n')


