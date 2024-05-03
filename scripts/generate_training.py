import sys, os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", help="Path of the data", required=True)
    parser.add_argument("-o","--output", help ="Output path", required=True)
    parser.add_argument("-d","--distribution", help="Distribution of the output data, separated by /, example: 70/20/10")
    parser.add_argument("--benchmark", help="Wether you want to use residues or benchmark", action="store_true")
    args = parser.parse_args()

    train_size, val_size, test_size = [int(num)/100 for num in args.distribution.split("/")]
    
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
    
    df = []
    
    if args.benchmark:
        filename = "benchmark"
    else:
        filename = "residue"

    print("[CSV/FASTA] Loading {}/{}".format(args.input, filename))
    df = pd.read_csv("{}/{}.csv".format(args.input, filename))
    

    if not os.path.exists("{}".format(args.output)):
        os.mkdir("{}".format(args.output)) 
    else:
        print("[ERROR] {}/{} already exists! Delete the folder to proceed".format(args.output, filename))
        exit(0)


    labels = pd.factorize(df['class'])[0]
    
    for model in models:
        for reduced in ["", "_reduced"]:
            # Cargo el primero        

            if( not os.path.exists( "{}/{}/{}{}.npy".format(args.input, filename, model[0], reduced)) ):
                continue

            print("[NPY] Loading {}/{}/{}{}.npy".format(args.input, filename, model[0], reduced))
            data = np.load("{}/{}/{}{}.npy".format(args.input, filename, model[0], reduced))

            
            X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=test_size+val_size, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(1-val_size), random_state=42)
            
            
            print("[NPY] Saving X_Train Into {}".format(args.output))
            np.save("{}/X_train_{}{}.npy".format(args.output, model[0], reduced), X_train)
            print("[NPY] Saving Y_Train Into {}".format(args.output))
            np.save("{}/y_train_{}{}.npy".format(args.output, model[0], reduced), y_train)
            
            print("[NPY] Saving X_Val Into {}".format(args.output))
            np.save("{}/X_val_{}{}.npy".format(args.output, model[0], reduced), X_val)
            print("[NPY] Saving Y_Val Into {}".format(args.output))
            np.save("{}/y_val_{}{}.npy".format(args.output, model[0], reduced), y_val)
            
            print("[NPY] Saving X_Test Into {}".format(args.output))
            np.save("{}/X_test_{}{}.npy".format(args.output, model[0], reduced), X_test)
            print("[NPY] Saving Y_Test Into {}".format(args.output))
            np.save("{}/y_test_{}{}.npy".format(args.output, model[0], reduced), y_test)
