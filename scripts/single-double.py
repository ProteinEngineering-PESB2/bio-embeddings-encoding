import os, argparse
import pandas as pd
from parser import FastaSave

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", help="Input file", required=True)
parser.add_argument("-o","--output", help ="Output file path", required=True)
args = parser.parse_args()

dataframe = pd.read_csv(args.input)

weird = dataframe.loc[dataframe.is_single & dataframe.is_double]

single =dataframe.loc[dataframe.is_single & ~dataframe.is_double]

double =dataframe.loc[~dataframe.is_single & dataframe.is_double]

dataframe = dataframe[~dataframe.isin(weird)].dropna()

dataframe = dataframe[~dataframe.isin(single)].dropna()

dataframe = dataframe[~dataframe.isin(double)].dropna()

FastaSave(weird['sequence'].to_list(), "{}/weird.fasta".format(args.output))
FastaSave(single['sequence'].to_list(), "{}/single.fasta".format(args.output))
FastaSave(double['sequence'].to_list(), "{}/double.fasta".format(args.output))
FastaSave(dataframe['sequence'].to_list(), "{}/residues.fasta".format(args.output))

