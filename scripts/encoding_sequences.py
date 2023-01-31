from bio_embeddings.embed import *
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

data_path = sys.argv[1]
export_path = sys.argv[2]
sequence_col = sys.argv[3]
export_cols = sys.argv[4].split(",")
embedder_name = sys.argv[5]
device = 'cuda'

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def ProtTransT5XLU50(device: str):
    return ProtTransT5XLU50Embedder(half_precision_model=True, device=device)

def ESM1v(device: str):
    return ESM1vEmbedder(ensemble_id=5, device=device)

def default():
    print(f"Model not found.")
    exit(-1)

models = {
    'bepler': BeplerEmbedder,
    'cpcprot': CPCProtEmbedder,
    'esm': ESMEmbedder,
    'esm1b': ESM1bEmbedder,
    'esm1v': ESM1v,
    'fasttext': FastTextEmbedder,
    'glove': GloveEmbedder,
    'one_hot_encoding': OneHotEncodingEmbedder,
    'plus_rnn': PLUSRNNEmbedder,
    'prottrans_albert_bfd': ProtTransAlbertBFDEmbedder,
    'prottrans_bert_bfd': ProtTransBertBFDEmbedder,
    'prottrans_t5_xl_u50': ProtTransT5XLU50,
    'prottrans_xlnet_uniref100': ProtTransXLNetUniRef100Embedder,
    'seqvec': SeqVecEmbedder,
    'unirep': UniRepEmbedder,
    'word2vec': Word2VecEmbedder,
}

print("Loading Data")
sequences = pd.read_csv(data_path)

print(f"Loading {embedder_name} Model")
embedder = None
try:
    embedder = models[embedder_name](device=device)
except KeyError:
    default()

print("Encoding")
np_data = np.zeros(shape=(len(sequences),embedder.embedding_dimension))
embeddings = embedder.embed_many(sequences.loc[:,sequence_col].to_list())
for idx, embed in tqdm(enumerate(embeddings), desc="Reducing embeddings"):
    np_data[idx] = embedder.reduce_per_protein(embed)
np_data = trunc(np_data, decs=4)

print("Processing")
header = ['p_{}'.format(i) for i in range(embedder.embedding_dimension)]
df_data = pd.DataFrame(np_data, columns=header)
for idx, col in enumerate(export_cols):
    df_data.insert(loc=idx, column=col, value=sequences[col])

print("Saving data")
if not os.path.exists(export_path):
    os.mkdir(export_path)

df_data.to_csv(f"{export_path}{embedder_name}_encoding.csv",index=False)
