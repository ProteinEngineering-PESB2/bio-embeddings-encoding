import pandas as pd
from Bio import SeqIO

def FastaParse(fasta_file: str, col_name: str):
    sequences = []
    ids = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))

    df = pd.DataFrame({col_name: sequences})
    return df
