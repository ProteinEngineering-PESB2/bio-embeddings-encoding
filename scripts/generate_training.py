import sys, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="Path of the data", required=True)
    parser.add_argument("-o","--output", help ="Output path", required=True)
    parser.add_argument("-d","--distribution", help="Distribution of the output data, separated by /, example: 70/20/10", required=False, default='')
    parser.add_argument("--benchmark", help="Wether you want to use residues or benchmark", action="store_true")
    parser.add_argument("-t", "--task", help="Task, 'binary' or 'classification'", required=False, default='classification')
    parser.add_argument("-r","--response", help="Name of the response column", default="response")
    args = parser.parse_args()

    if len(args.distribution.split("/")) == 3:
        train_size, val_size, test_size = [int(num)/100 for num in args.distribution.split("/")]
    elif len(args.distribution.split("/")) == 2:
        train_size, val_size = [int(num)/100 for num in args.distribution.split("/")]
    elif args.distribution != '':
        print("Not valid distribution values given.")
        exit(-1)
    
    # The embedding dimension has no usage for the moment.
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
    models.extend([(f"Group_{i}", -1) for i in range(8)])
    models.extend([(f"Group_{i}_fft", -1) for i in range(8)])
    
    df = []
    
    if args.benchmark:
        filename = "benchmark"
    else:
        filename = "residue"

    print("[CSV/FASTA] Loading {}/{}".format(args.input, filename))
    df = pd.read_csv("{}/{}.csv".format(args.input, filename))
    

    if not os.path.exists(args.output):
        Path(args.output).mkdir(parents=True) 
    else:
        print("[WARN] {}/{} already exists! Adding new files...".format(args.output, filename))

    if args.task == 'classification':
        labels = pd.factorize(df['class'])[0]
    elif args.task == 'binary':
        labels = df[args.response]
    else:
        print("Not valid task provided. Exiting...")
        exit(-1)
    
    for model in models:
        for reduced in ["", "_reduced"]:
            # Cargo el primero        

            if( not os.path.exists( "{}/{}/{}{}.npy".format(args.input, filename, model[0], reduced)) ):
                continue

            print("[NPY] Loading {}/{}/{}{}.npy".format(args.input, filename, model[0], reduced))
            data = np.load("{}/{}/{}{}.npy".format(args.input, filename, model[0], reduced))
            
            if not args.benchmark:
                if len(args.distribution.split("/")) == 3:
                    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=test_size+val_size, random_state=42)
                    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(1-val_size), random_state=42)
                else:
                    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=val_size, random_state=42)
                
                
                print("[NPY] Saving X_Train Into {}".format(args.output))
                np.save("{}/X_train_{}{}.npy".format(args.output, model[0], reduced), X_train)
                print("[NPY] Saving Y_Train Into {}".format(args.output))
                np.save("{}/y_train_{}{}.npy".format(args.output, model[0], reduced), y_train)
                
                print("[NPY] Saving X_Val Into {}".format(args.output))
                np.save("{}/X_val_{}{}.npy".format(args.output, model[0], reduced), X_val)
                print("[NPY] Saving Y_Val Into {}".format(args.output))
                np.save("{}/y_val_{}{}.npy".format(args.output, model[0], reduced), y_val)
            
            if args.benchmark:
                X_test = data
                y_test = labels

            if len(args.distribution.split("/")) == 3 or args.benchmark:
                print("[NPY] Saving X_Test Into {}".format(args.output))
                np.save("{}/X_test_{}{}.npy".format(args.output, model[0], reduced), X_test)
                print("[NPY] Saving Y_Test Into {}".format(args.output))
                np.save("{}/y_test_{}{}.npy".format(args.output, model[0], reduced), y_test)
