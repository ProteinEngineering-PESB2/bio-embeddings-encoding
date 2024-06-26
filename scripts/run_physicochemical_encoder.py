import pandas as pd
import sys, os, argparse
import numpy as np

from physicochemical_properties import PhysicochemicalEncoder
from fft_encoder import FFTTransform

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", help="Input file", required=True)
parser.add_argument("-o","--output", help ="Output file path", required=True)
parser.add_argument("-g","--group", help="Name of the group", required=True)
parser.add_argument("-e","--encoder", help="Path to the encoder", required=True)
args = parser.parse_args()

print("Loading Data")
df = pd.read_csv(args.input)


name_export = f"{args.output}/{args.group}.npy"

physicochemical_encoder = PhysicochemicalEncoder(
        dataset=df,
        dataset_encoder=pd.read_csv(args.encoder),
        columns_to_ignore=["activity"],
        name_column_seq="sequence"
    )


physicochemical_encoder.run_process()

data = physicochemical_encoder.matrix

data_encoded = physicochemical_encoder.df_data_encoded

fft_transform = FFTTransform(
        dataset=data_encoded,
        size_data=len(data_encoded.columns)-1,
        columns_to_ignore=["activity"]
    )

data_fft = fft_transform.encoding_dataset()


with open("{}{}.npy".format(args.output, args.group), 'wb') as f:
        np.save(f, np.array(data))

with open("{}{}_fft.npy".format(args.output, args.group), 'wb') as f:
    np.save(f, np.array(data_fft))
