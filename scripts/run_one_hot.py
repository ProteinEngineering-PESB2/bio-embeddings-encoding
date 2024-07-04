import pandas as pd
import numpy as np
from one_hot_encoding import OneHotEncoder
import sys, os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="Input file", required=True)
    parser.add_argument("-o","--output", help ="Output file path", required=True)
    parser.add_argument("-s","--seq_col", help="Name of the sequence column", required=True)
    args = parser.parse_args()

    print("Loading Data for One Hot Encoder")
    df = pd.read_csv(args.input)

    name_export = f"{args.output}/david_one_hot.npy"

    one_hot_instance = OneHotEncoder(dataset=df, column_sequence=args.seq_col, max_length=3000)
    output_data = one_hot_instance.run_process()
    output_data = output_data.to_numpy()
    
    print("Saving data")
    with open(name_export, 'wb') as f:
        np.save(f, np.array(output_data))
