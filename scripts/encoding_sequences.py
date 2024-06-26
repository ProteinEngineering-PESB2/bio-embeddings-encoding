import sys, os, argparse
from bio_embeddings.embed import ( ESM1bEmbedder, BeplerEmbedder, ProtTransT5BFDEmbedder, CPCProtEmbedder, ESMEmbedder, FastTextEmbedder, GloveEmbedder, OneHotEncodingEmbedder, PLUSRNNEmbedder, ProtTransAlbertBFDEmbedder, ProtTransBertBFDEmbedder, ProtTransT5BFDEmbedder, ProtTransT5XLU50Embedder, ProtTransXLNetUniRef100Embedder, Word2VecEmbedder, SeqVecEmbedder)
import pandas as pd
import numpy as np
from tqdm import tqdm
from parser import FastaParse, FastaSave

class UsingBioembeddings:
    """"""
    def __init__( self, dataset=None, column_seq=None, is_reduced=True, device = None ):
        self.dataset = dataset
        self.column_seq = column_seq
        self.is_reduced=is_reduced
        self.device = device

        # to save the results
        self.embedder = None
        self.embeddings = None
        self.np_data = None

    def __reducing(self):
        self.np_data = np.zeros(shape=(len(self.dataset), self.embedder.embedding_dimension))
        for idx, embed in tqdm(enumerate(self.embeddings), desc="Reducing embeddings"):
            self.np_data[idx] = self.embedder.reduce_per_protein(embed)

    def __non_reducing(self):
        max_length = 150
        self.np_data = np.zeros(shape=(len(self.dataset), max_length, self.embedder.embedding_dimension))
        for idx, embed in tqdm(enumerate(self.embeddings), desc="Assigning embeddings"):
            if len(embed) >= max_length:
                embed = embed[:max_length]
            else:
                embed = np.pad(embed, ((0, max_length - len(embed)), (0, 0)), 'constant')
            self.np_data[idx] = embed
    

    def apply_model(self, model, embedding_dim):
        """"""
        if self.device is not None:
            self.embedder = model(device=self.device, embedding_dimension = embedding_dim)
        else:
            self.embedder = model( embedding_dimension = embedding_dim)
        self.embeddings = self.embedder.embed_many(
            self.dataset[self.column_seq].to_list())
        if self.is_reduced is True:
            self.__reducing()
        else:
            self.__non_reducing()
        return self.np_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="Input file", required=True)
    parser.add_argument("-o","--output", help ="Output file path", required=True)
    parser.add_argument("-s","--seq_col", help="Name of the sequence column", required=True)
    parser.add_argument("-e","--embedder", help="Name of the embedder", required=True)
    parser.add_argument("--reduced", action="store_true")
    args = parser.parse_args()
    max_rows = 50000

    if os.path.exists("{}{}{}.npy".format(args.output, args.embedder,"_reduced" if args.reduced else "")):
        print("[WARN] {} already exists! Skipping.".format(args.embedder))
        exit(0)

    
    print("Loading Data")
    
    if args.input.endswith('.fasta'):
        df = FastaParse(args.input, args.seq_col)
    else:
        df = pd.read_csv(args.input)

    if df.shape[0] > max_rows:
        print("[WARN] Max rows reached, reducing.")
        df = df.sample(max_rows, random_state=42).reset_index()
        os.rename(args.input, args.input+".backup")
        #FastaSave(df[args.seq_col], args.input)
        df.to_csv(args.input, index=False)
    
    bio_embeddings = UsingBioembeddings(df, args.seq_col, args.reduced, device='cuda')
    
    """
    ESM1v uses an ensemble of five models, called `esm1v_t33_650M_UR90S_[1-5]`. An instance of this class is one
    of the five, specified by `ensemble_id`.
    """
    dict_models = {
        "bepler" : (BeplerEmbedder, 121),
        "cpcprot" : (CPCProtEmbedder, 512),
        "esm" : (ESMEmbedder, 1280),
        "esm1b" : (ESM1bEmbedder, 1280),
        "fasttext" : (FastTextEmbedder, 512),
        "glove" : (GloveEmbedder, 512),
        "onehot" : (OneHotEncodingEmbedder, 21),
        "plusrnn" : (PLUSRNNEmbedder, 1024),
        "prottrans_albert" : (ProtTransAlbertBFDEmbedder, 4096),
        "prottrans_bert" : (ProtTransBertBFDEmbedder, 1024),
        "prottrans_t5bfd" : (ProtTransT5BFDEmbedder, 1024),
        "prottrans_xlnet_uniref100" : (ProtTransXLNetUniRef100Embedder, 1024),
        "prottrans_t5xlu50" : (ProtTransT5XLU50Embedder, 1024),
        "seqvec" : (SeqVecEmbedder, 1024),
        "word2vec" : (Word2VecEmbedder, 512),
    }
    
    print(f"Loading {args.embedder} Model")
    encoded_df = bio_embeddings.apply_model(dict_models[args.embedder][0], dict_models[args.embedder][1])

    print("Saving data")
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    with open("{}{}{}.npy".format(args.output, args.embedder,"_reduced" if args.reduced else ""), 'wb') as f:
        np.save(f, encoded_df)
    
