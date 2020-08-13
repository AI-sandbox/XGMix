import argparse
import gzip
import logging
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from time import time
import xgboost as xgb

from Admixture.Admixture import split_sample_map, main_admixture
from Admixture.utils import read_vcf, join_paths, run_shell_cmd
from preprocess import load_np_data, data_process
from postprocess import vcf_to_npy, get_msp_data, write_msp_tsv
from visualization import plot_cm

from config import *

CLAIMER = 'When using this software, please cite: \n' + \
            'Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A. \n' + \
            '"XGMix: Local-Ancestry Inference With Stacked XGBoost" \n' + \
            'International Conference on Learning Representations Workshops \n' + \
            'ICLR, 2020, Workshop AI4AH \n' + \
            'https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1'

class XGMIX():

    def __init__(self,chmlen,win,sws,num_anc,snp_pos=None,snp_ref=None,population_order=None, save=None,
                base_params=[20,4],smooth_params=[100,4],cores=4,lr=0.1,reg_lambda=1,reg_alpha=0,model="xgb"):

        self.chmlen = chmlen
        self.win = win
        self.save = save
        self.sws = sws if sws %2 else sws-1
        self.num_anc = num_anc
        self.snp_pos = snp_pos
        self.snp_ref = snp_ref
        self.population_order = population_order
        self.trees, self.max_depth = base_params
        self.missing = 2
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.model = model

        self.s_trees, self.s_max_depth = smooth_params
        self.cores = cores

        self.num_windows = self.chmlen//self.win
        self.pad_size = (1+self.sws)//2

        self.base = {}
        self.smooth = None

        # model stats
        self.training_time = None
        self.base_acc_train = None
        self.base_acc_val = None
        self.smooth_acc_train = None
        self.smooth_acc_val = None

    def _train_base(self,train,train_lab,val,val_lab,evaluate=True):

        for idx in range(self.num_windows):

            # a particular window across all examples
            tt = train[:,idx*self.win:(idx+1)*self.win]
            vt = val[:,idx*self.win:(idx+1)*self.win]
            ll_t = train_lab[:,idx]
            ll_v = val_lab[:,idx]

            if idx == self.num_windows-1:
                tt = train[:,idx*self.win:]
                vt = val[:,idx*self.win:]

            # fit model
            if self.model == "xgb":
                if idx == 0:
                    print("using xgb")
                model = xgb.XGBClassifier(n_estimators=self.trees,max_depth=self.max_depth,
                        learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                        nthread=self.cores, missing=self.missing)
            if self.model == "rf":
                from sklearn import ensemble
                if idx == 0:
                    print("using rf")
                model = ensemble.RandomForestClassifier(n_estimators=self.trees,
                        max_depth=self.max_depth,n_jobs=self.cores) 
            elif self.model == "lgb":
                import lightgbm as lgb
                if idx == 0:
                    print("using lgb")
                model = lgb.LGBMClassifier(n_estimators=self.trees, max_depth=self.max_depth,
                            learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                            nthread=self.cores) 
            elif self.model == "cb":
                import catboost as cb
                if idx == 0:
                    print("using cb")
                model = cb.CatBoostClassifier(n_estimators=self.trees, max_depth=self.max_depth,
                            learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, 
                            thread_count=self.cores, verbose=0)
            elif self.model == "svm":
                from sklearn import svm
                if idx == 0:
                    print("using svm")
                model = svm.SVC(C=100., gamma=0.001, probability=True)

            model.fit(tt,ll_t)
            self.base["model"+str(idx*self.win)] = model

            sys.stdout.write("\rWindows done: %i/%i" % (idx+1, self.num_windows))
        
        print("")

        if evaluate:

            train_accr, val_accr = [], []

            for idx in range(self.num_windows):

                model = self.base["model"+str(idx*self.win)]

                # a particular window across all examples
                tt = train[:,idx*self.win:(idx+1)*self.win]
                vt = val[:,idx*self.win:(idx+1)*self.win]
                ll_t = train_lab[:,idx]
                ll_v = val_lab[:,idx]

                if idx == self.num_windows-1:
                    tt = train[:,idx*self.win:]
                    vt = val[:,idx*self.win:]
                
                y_pred = model.predict(tt)
                train_metric = accuracy_score(y_pred,ll_t)
                train_accr.append(train_metric)

                y_pred = model.predict(vt)
                val_metric = accuracy_score(y_pred,ll_v)
                val_accr.append(val_metric)

            self.base_acc_train = round(np.mean(train_accr),4)*100
            self.base_acc_val = round(np.mean(val_accr),4)*100
            print("Base Training Accuracy:   {}%".format(self.base_acc_train))
            print("Base Validation Accuracy: {}%".format(self.base_acc_val))

    def _get_smooth_data(self,data,labels):

        # get base output
        base_out = np.zeros((data.shape[0],len(self.base),self.num_anc))
        for i in range(len(self.base)):

            inp = data[:,i*self.win:(i+1)*self.win]
            if i == len(self.base)-1:
                inp = data[:,i*self.win:]
            base_model = self.base["model"+str(i*self.win)]
            base_out[:,i,base_model.classes_] = base_model.predict_proba(inp)

        # pad it.
        pad_left = np.flip(base_out[:,0:self.pad_size,:],axis=1)
        pad_right = np.flip(base_out[:,-self.pad_size:,:],axis=1)

        base_out_padded = np.concatenate([pad_left,base_out,pad_right],axis=1)

        # window it.
        windowed_data = np.zeros((data.shape[0],len(self.base),self.num_anc*self.sws))
        for ppl,dat in enumerate(base_out_padded):
            for win in range(windowed_data.shape[1]):
                windowed_data[ppl,win,:] = dat[win:win+self.sws].ravel()

        # reshape
        return windowed_data.reshape(-1,windowed_data.shape[2]), labels.reshape(-1)


    def _train_smooth(self,train,train_lab,val,val_lab,smoothlite):

        print("Smoothing predictions and evaluating...")

        tt,ttl = self._get_smooth_data(train,train_lab)
        vv,vvl = self._get_smooth_data(val,val_lab)

        # # Smooth lite option - to use less data for the smoother - faster if the data is good
        smoothlite = smoothlite if smoothlite else len(tt)
        indices = np.random.choice(len(tt), min(smoothlite,len(tt)), replace=False)
        tt = tt[indices]
        ttl = ttl[indices]

        # Train model
        if self.model == "xgb":
            self.smooth = xgb.XGBClassifier(n_estimators=self.s_trees,max_depth=self.s_max_depth,
                learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, nthread=self.cores)
        elif self.model == "rf":
            self.smooth = ensemble.RandomForestClassifier(n_estimators=self.s_trees, 
                max_depth=self.s_max_depth, n_jobs=self.cores) 
        elif self.model == "lr":
            self.smooth = linear_model.LogisticRegression(n_jobs=self.cores)
        elif self.model == "lgb":
            self.smooth = lgb.LGBMClassifier(n_estimators=self.s_trees,max_depth=self.s_max_depth,
                learning_rate=self.lr,reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, nthread=self.cores)
        elif self.model == "cb":
            self.smooth = cb.CatBoostClassifier(n_estimators=self.s_trees, max_depth=self.s_max_depth,
                learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, thread_count=self.cores,
                verbose=0)
        elif self.model == "svm":
            self.smooth = svm.SVC(C=100., gamma=0.001, probability=True)
        
        self.smooth.fit(tt,ttl)

        # Evaluate model
        y_pred = self.smooth.predict(vv)
        t_pred = self.smooth.predict(tt)
        self.smooth_acc_train = round(accuracy_score(t_pred,ttl),4)*100
        self.smooth_acc_val   = round(accuracy_score(y_pred,vvl),4)*100
        print("Smooth Training Accuracy:   {}%".format(self.smooth_acc_train))
        print("Smooth Validation Accuracy: {}%".format(self.smooth_acc_val))

    def train(self,train,train_lab,val,val_lab,smoothlite=10000):

        # smoothlite: int or False. If False train smoother on all data, else train only on that number of contexts.

        train_time_begin = time()
        self._train_base(train,train_lab,val,val_lab)
        self._train_smooth(train,train_lab,val,val_lab,smoothlite=smoothlite)
        self.training_time = time() - train_time_begin

        # Save model
        if self.save is not None:
            pickle.dump(self, open(self.save+"model.pkl", "wb" ))

    def predict(self,tt):
        n,_ = tt.shape
        tt,_ = self._get_smooth_data(tt,np.zeros((2,2)))
        y_preds = self.smooth.predict(tt)

        return y_preds.reshape(n,len(self.base))

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_model(path_to_model, verbose=True):
    if verbose:
        print("Loading model...")
    if path_to_model[-3:]==".gz":
        with gzip.open(path_to_model, 'rb') as unzipped:
            model = pickle.load(unzipped)
    else:
        model = pickle.load(open(path_to_model,"rb"))
    return model

def train(chm, model_name, data_path, generations = [2,4,6], window_size = 5000,
          smooth_size = 75, missing = 0.0, n_cores = 16, smooth_lite = 10000, verbose=True):

    if verbose:
        print("Preprocessing data...")
    
    # ------------------ Config ------------------
    model_name += "_chm_" + chm
    model_repo = join_paths("./models", model_name, verb=False) 
    model_path = model_repo + "/" + model_name + ".pkl"

    train_paths = [data_path + "/chm" + chm + "/simulation_output/train/gen_" + str(gen) + "/" for gen in generations]
    val_paths   = [data_path + "/chm" + chm + "/simulation_output/val/gen_"   + str(gen) + "/" for gen in generations] # only validate on 4th gen

    position_map_file   = data_path + "/chm"+ chm + "/positions.txt"
    reference_map_file  = data_path + "/chm"+ chm + "/references.txt"
    population_map_file = data_path + "/populations.txt"
    
    # ------------------ Process data ------------------
    # gather feature data files (binary representation of variants)
    X_fname = "mat_vcf_2d.npy"
    X_train_files = [p + X_fname for p in train_paths]
    X_val_files   = [p + X_fname for p in val_paths]

    # gather label data files (population)
    labels_fname = "mat_map.npy"
    labels_train_files = [p + labels_fname for p in train_paths]
    labels_val_files   = [p + labels_fname for p in val_paths]

    # load the data
    train_val_files = [X_train_files, labels_train_files, X_val_files, labels_val_files]
    X_train_raw, labels_train_raw, X_val_raw, labels_val_raw = [load_np_data(f) for f in train_val_files]

    # reshape according to window size
    X_train, labels_window_train = data_process(X_train_raw, labels_train_raw, window_size, missing)
    X_val, labels_window_val     = data_process(X_val_raw, labels_val_raw, window_size, missing)

    # necessary arguments for model
    snp_pos = np.loadtxt(position_map_file,  delimiter='\n').astype("int")
    snp_ref = np.loadtxt(reference_map_file, delimiter='\n', dtype=str)
    pop_order = np.genfromtxt(population_map_file, dtype="str")
    chm_len = len(snp_pos)
    num_anc = len(pop_order)

    # ------------------ Train model ------------------    
    # init, train, evaluate and save model
    if verbose:
        print("Initializing XGMix model and training...")
    model = XGMIX(chm_len, window_size, smooth_size, num_anc, snp_pos, snp_ref, pop_order, cores=n_cores)
    model.train(X_train, labels_window_train, X_val, labels_window_val, smooth_lite)

    # evaluate model
    analysis_path = join_paths(model_repo, "analysis", verb=False)
    CM(labels_window_val.ravel(), model.predict(X_val).ravel(), pop_order, analysis_path, verbose)
    pickle.dump(model, open(model_path,"wb"))
    
    return model

def CM(y, y_pred, labels, save_path=None, verbose=True):
    cm = confusion_matrix(y, y_pred)
    if verbose:
        print("Confusion matrix for validation data:")
        print(cm)
    if save_path is not None:
        n_digits = int(np.ceil(np.log10(np.max(cm))))
        str_fmt = '%-'+str(n_digits)+'.0f'
        np.savetxt(save_path+"/confusion_matrix.txt", cm, fmt=str_fmt)
        cm_figure = plot_cm(cm, normalize=True, labels=labels)
        cm_figure.figure.savefig(save_path+"/confusion_matrix_normalized.png")
        if verbose:
            print("Confusion matrix saved in", save_path)

def main(args, verbose=True):

    # Either load pre-trained model or simulate data from reference file, init model and train it
    if mode == "pre-trained":
        model = load_model(args.path_to_model, verbose=verbose)
    elif args.mode == "train":

        # Set output path
        data_path = join_paths('./generated_data', instance_name, verb=False)

        # Running simulation. If data is already simulated, skipping can save a lot of time
        if run_simulation:
            # Splitting the reference sample in train/val/test
            sub_instance_names = ["train", "val", "test"]
            sample_map_files, sample_map_files_idxs = split_sample_map(data_path, args.sample_map_file)

            # Simulating data
            main_admixture(args.chm, data_path, sub_instance_names, sample_map_files, sample_map_files_idxs,
                        args.reference_file, args.genetic_map_file, num_outs, generations)

            if verbose:
                print("Simulation done.")
                print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

        # Processing data, init and training model
        model = train(args.chm, model_name, data_path, generations, window_size, smooth_size, missing, n_cores, smooth_lite)
        if verbose:
            print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

    if args.query_file is not None:
        # Load and process user query file
        if verbose:
            print("Loading and processing query file...")
        X_query, query_pos, model_idx, query_samples = vcf_to_npy(args.query_file, args.chm, model.snp_pos,
                                                                model.snp_ref, verbose=verbose)

        # predict and finding effective prediction for intersection of query SNPs and model SNPs positions
        if verbose:
            print("Analyzing...")
        label_pred_query_window = model.predict(X_query)

        # writing the result to disc
        if verbose:
            print("Writing analyzis to disc...")
        msp_data = get_msp_data(args.chm, label_pred_query_window, model.snp_pos, query_pos,
                                model.num_windows, model.win, args.genetic_map_file)
        write_msp_tsv(args.output_basename, msp_data, model.population_order, query_samples)

    if mode=="train" and rm_simulated_data:
        if verbose:
            print("Removing simulated data...")
        chm_path = join_paths(data_path, "chm" + args.chm, verb=False)
        remove_data_cmd = "rm -r " + chm_path
        run_shell_cmd(remove_data_cmd, verbose=False)

    if verbose:
        print("Finishing up...")

if __name__ == "__main__":

    # Citation
    print("-"*80+"\n"+"-"*35+"  XGMix  "+"-"*36 +"\n"+"-"*80)
    print(CLAIMER)
    print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

    # Infer mode from number of arguments
    mode = None
    if len(sys.argv) == 6:
        mode = "pre-trained" 
    if len(sys.argv) == 7:
        mode = "train"

    # Usage message
    if mode is None:
        if len(sys.argv) > 1:
            print("Error: Not correct number of arguments.")
        print("Usage when training a model from scratch:")
        print("   $ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <reference_file> <sample_map_file>")
        print("Usage when using a pre-trained model:")
        print("   $ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <path_to_model>")
        sys.exit(0)

    # Deconstruct CL arguments
    base_args = {
        'mode': mode,
        'query_file': sys.argv[1] if sys.argv[1].strip() != "None" else None,
        'genetic_map_file': sys.argv[2],
        'output_basename': sys.argv[3],
        'chm': sys.argv[4]
    }
    args = Struct(**base_args)
    if mode == "train":
        args.reference_file  = sys.argv[5]
        args.sample_map_file = sys.argv[6]
    elif mode == "pre-trained":
        args.path_to_model = sys.argv[5]

    # Run it
    if verbose:
        print("Launching XGMix in", mode, "mode...")
    main(args)
