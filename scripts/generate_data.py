import sys, os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Bio import SeqIO

def parse_fasta(fasta_file: str, col_name: str, col_class: str):
    sequences = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))

    df = pd.DataFrame({col_name: sequences}).reset_index()
    df['class'] = col_class
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="Input data path", required=True)
    parser.add_argument("-c", "--classes", help="Classes that will be generated, separated by comma", required=True)
    parser.add_argument("-s","--seq_col", help="Name of the sequence column", required=True)
    parser.add_argument("-v", "--values", help="Values for the classes", required=True)
    parser.add_argument("-o","--output", help ="Output path", required=True)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()

    classes = args.classes.split(",")
    values = list(map(int, args.values.split(',')))

    models = [
            ("bepler", 121),
            ("cpcprot", 512),
            ("esm", 1280),
            ("esm1b", 1280),
            ("fasttext", 512),
            ("glove", 512),
            ("onehot", 21),
            ("plusrnn", 1024),
            ("prottrans_albert", 4096),
            ("prottrans_bert", 1024),
            ("prottrans_t5bfd", 1024),
            ("prottrans_xlnet_uniref100", 1024),
            ("prottrans_t5xlu50", 1024),
            ("seqvec", 1024),
            ("word2vec", 512)
        ]
    
    df_classes = []
    df_choices = []
    pathname = "vs".join(classes)
    for idx, cls in enumerate(classes):
        print("[CSV/FASTA] Loading {}/{}".format(args.input, cls))
        if os.path.exists("{}/{}.fasta".format(args.input, cls)):
            df_classes.append(parse_fasta("{}/{}.fasta".format(args.input, cls), args.seq_col, cls))
        elif os.path.exists("{}/{}.csv".format(args.input, cls)):
            tmp = pd.read_csv("{}/{}.csv".format(args.input, cls))
            tmp['class'] = cls
            df_classes.append(tmp)
        else:
            print("{}/{} not found. Exiting...".format(args.input, cls))
            exit(-1)
        
    if args.balanced:
        df_choices = pd.concat(df_classes, axis=0)
        g = df_choices.groupby("class")
        df_choices = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)
        df_choices = df_choices.sort_values(by=["class","index"]).reset_index(drop=True)

    if not os.path.exists("{}/{}".format(args.output, pathname)):
        os.mkdir("{}/{}".format(args.output, pathname))
    else:
        print("[ERROR] {}/{} already exists! Delete the folder to proceed".format(args.output, pathname))
        exit(0)

    for model in models:
        for reduced in ["", "_reduced"]:
            # Cargo el primero        
            if( not os.path.exists( "{}/{}/{}{}.npy".format(args.input, classes[0], model[0], reduced)) ):
                print("[WARN] {}/{}/{}{}.npy does not exists, skipping.".format(args.input, classes[0], model[0], reduced))
                continue

            print("[NPY] Loading {}/{}/{}{}.npy".format(args.input, classes[0], model[0], reduced))
            data = np.load("{}/{}/{}{}.npy".format(args.input, classes[0], model[0], reduced))

            # Reduzco
            #data = data[df_choices[0]['index'].to_numpy()]
            #    print(df_choices.loc[df_choices['class'] == classes[0]]['index'].to_numpy())
            if args.balanced:
                data = data[df_choices.loc[df_choices['class'] == classes[0]]['index'].to_numpy()]

            # Cargo el resto
            for idx,cls in enumerate(classes[1:]):
                print("[NPY] Loading {}/{}/{}{}.npy".format(args.input, cls, model[0], reduced))
                if( not os.path.exists( "{}/{}/{}{}.npy".format(args.input, cls, model[0], reduced) )):
                    #print("[WARN] {}/{}/{}.npy doesn't exist, skipping".format(args.input, cls, model[0]))
                    continue
                tmp = np.load("{}/{}/{}{}.npy".format(args.input, cls, model[0], reduced))
                # Reduzco
                if args.balanced:
                    tmp = tmp[df_choices.loc[df_choices['class'] == classes[idx+1]]['index'].to_numpy()]

                data = np.concatenate((data,tmp), axis=0)
            
            
            print("[NPY] Saving Into {}/{}/".format(args.output, pathname))
            np.save("{}/{}/{}{}.npy".format(args.output, pathname, model[0], reduced), data)
    

    print("[CSV] Saving into {}/{}.csv".format(args.output, pathname))
    classes_dict = dict(zip(classes, values))
    if args.balanced:
        df_choices['class'] = df_choices['class'].replace(classes_dict)
        df_choices.to_csv("{}/{}.csv".format(args.output, pathname), index=False)
    else:
        df_classes = pd.concat(df_classes, axis=0)
        df_classes['class'] = df_classes['class'].replace(classes_dict)
        df_classes.to_csv("{}/{}.csv".format(args.output, pathname), index=False)
    
