import argparse
import gzip
import logging
import numpy as np
import os
import pandas as pd
import pickle
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
from time import time
import xgboost as xgb

from utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy
from preprocess import load_np_data, data_process, get_gen_0
from postprocess import get_msp_data, write_msp_tsv
from visualization import plot_cm
from Calibration import calibrator_module, normalize_prob

from Admixture.Admixture import read_sample_map, split_sample_map, main_admixture

from XGFix.XGFIX import XGFix

from config import verbose, instance_name, run_simulation, founders_ratios, num_outs, generations, rm_simulated_data
from config import model_name, window_size_cM, smooth_size, missing, retrain_base, calibrate, n_cores

# The simulation can't handle generation 0, add it separetly
gen_0 = 0 in generations
generations = list(filter(lambda x: x != 0, generations))

CLAIMER = 'When using this software, please cite: \n' + \
          'Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A. \n' + \
          '"XGMix: Local-Ancestry Inference With Stacked XGBoost" \n' + \
          'International Conference on Learning Representations Workshops \n' + \
          'ICLR, 2020, Workshop AI4AH \n' + \
          'https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1'

np.random.seed(94305)

class XGMIX():

    def __init__(self,chmlen,win,sws,num_anc,snp_pos=None,snp_ref=None,population_order=None, save=None,
                base_params=[20,4],smooth_params=[100,4],cores=16,lr=0.1,reg_lambda=1,reg_alpha=0,model="xgb",
                mode_filter_size=5, calibrate=True,context_ratio=1.0):

        self.chmlen = chmlen
        self.win = win
        self.save = save
        self.sws = sws if sws %2 else sws-1
        # TODO: make it a separate parameter in config.
        self.context = int(self.win*context_ratio) # left and right are separate contexts.
        
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
        self.mode_filter_size = mode_filter_size
        self.calibrate = calibrate

        self.s_trees, self.s_max_depth = smooth_params
        self.cores = cores

        self.num_windows = self.chmlen//self.win
        self.pad_size = (1+self.sws)//2

        self.base = {}
        self.smooth = None
        self.calibrator = None

        # model stats
        self.training_time = None
        self.base_acc_train = None
        self.base_acc_val = None
        self.smooth_acc_train = None
        self.smooth_acc_val = None

    def _train_base(self,train,train_lab,evaluate=True):

        self.base = {}
        
        
        # TODO 1: pad train with extra data.
        
        pad_left = np.flip(train[:,0:self.context],axis=1)
        pad_right = np.flip(train[:,-self.context:],axis=1)
        train = np.concatenate([pad_left,train,pad_right],axis=1)
        
        start = self.context

        for idx in range(self.num_windows):

            # a particular window across all examples
            
            tt = train[:,start-self.context:start+self.context+self.win]
            
            ll_t = train_lab[:,idx]

            if idx == self.num_windows-1:
                tt = train[:,start-self.context:]
                
            start += self.win
            # TODO 2: pad train with left and right data.
            # print(tt.shape)
            # fit model
            model = xgb.XGBClassifier(n_estimators=self.trees,max_depth=self.max_depth,
                    learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                    nthread=self.cores, missing=self.missing, random_state=1) 
            model.fit(tt,ll_t)
            self.base["model"+str(idx*self.win)] = model

            sys.stdout.write("\rWindows done: %i/%i" % (idx+1, self.num_windows))
        
        print("")

    def _get_smooth_data(self, data=None, labels=None, base_out = None, return_base_out=False):
        
        # TODO 3: Make corresponding TODO 1, TODO 2 changes in this function

        if base_out is None:
            n_ind = data.shape[0]

            # get base output
            base_out = np.zeros((data.shape[0],len(self.base),self.num_anc),dtype="float32")
            start = self.context
            
            pad_left = np.flip(data[:,0:self.context],axis=1)
            pad_right = np.flip(data[:,-self.context:],axis=1)
            data = np.concatenate([pad_left,data,pad_right],axis=1)
            
            for i in range(len(self.base)):
                inp = data[:,start-self.context:start+self.context+self.win]
                if i == len(self.base)-1:
                    inp = data[:,start-self.context:]
                start += self.win
                base_model = self.base["model"+str(i*self.win)]
                # print(inp.shape)
                base_out[:,i,base_model.classes_] = base_model.predict_proba(inp)
    
            if return_base_out:
                return base_out
                
            del data, inp, base_model
            
        else:
            n_ind = base_out.shape[0]

        # pad it.
        pad_left = np.flip(base_out[:,0:self.pad_size,:],axis=1)
        pad_right = np.flip(base_out[:,-self.pad_size:,:],axis=1)
        base_out_padded = np.concatenate([pad_left,base_out,pad_right],axis=1)
        del base_out

        # window it.
        windowed_data = np.zeros((n_ind,len(self.base),self.num_anc*self.sws),dtype="float32")
        for ppl,dat in enumerate(base_out_padded):
            for win in range(windowed_data.shape[1]):
                windowed_data[ppl,win,:] = dat[win:win+self.sws].ravel()

        # reshape
        windowed_data = windowed_data.reshape(-1,windowed_data.shape[2])
        windowed_labels = None if labels is None else labels.reshape(-1)
    
        return windowed_data, windowed_labels


    def _train_smooth(self,train,train_lab,verbose=True):

        tt,ttl = self._get_smooth_data(train,train_lab)
        self.smooth = xgb.XGBClassifier(n_estimators=self.s_trees,max_depth=self.s_max_depth,
            learning_rate=self.lr, reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha, nthread=self.cores, random_state=1)
        self.smooth.fit(tt,ttl)

    def _evaluate_base(self,train,train_lab,val,val_lab,verbose=True):
        
        # TODO 4: Make the changes corressponding to TODO 1, TODO 2 here as well.

        train_accr, val_accr = [], []
        
        start = self.context

        pad_left = np.flip(train[:,0:self.context],axis=1)
        pad_right = np.flip(train[:,-self.context:],axis=1)
        train = np.concatenate([pad_left,train,pad_right],axis=1)
        
        pad_left = np.flip(val[:,0:self.context],axis=1)
        pad_right = np.flip(val[:,-self.context:],axis=1)
        val = np.concatenate([pad_left,val,pad_right],axis=1)
        
        for idx in range(self.num_windows):

            model = self.base["model"+str(idx*self.win)]

            # a particular window across all examples
            tt = train[:,start-self.context:start+self.context+self.win]
            vt = val[:,start-self.context:start+self.context+self.win]
            ll_t = train_lab[:,idx]
            ll_v = val_lab[:,idx]

            if idx == self.num_windows-1:
                tt = train[:,start-self.context:]
                vt = val[:,start-self.context:]
                
            start += self.win
            
            y_pred = model.predict(tt)
            train_metric = accuracy_score(y_pred,ll_t)
            train_accr.append(train_metric)

            y_pred = model.predict(vt)
            val_metric = accuracy_score(y_pred,ll_v)
            val_accr.append(val_metric)

        self.base_acc_train = round(np.mean(train_accr),4)*100
        self.base_acc_val = round(np.mean(val_accr),4)*100
        if verbose:
            print("Base Training Accuracy:   {}%".format(self.base_acc_train))
            print("Base Validation Accuracy: {}%".format(self.base_acc_val))

    def _evaluate_smooth(self,train,train_lab,val,val_lab,verbose=verbose):

        t_pred = self.predict(train)
        v_pred = self.predict(val)
        self.smooth_acc_train = round(accuracy_score(t_pred.reshape(-1),train_lab.reshape(-1)),4)*100
        self.smooth_acc_val = round(accuracy_score(v_pred.reshape(-1),val_lab.reshape(-1)),4)*100
        if verbose:
            print("Smooth Training Accuracy: {}%".format(self.smooth_acc_train))
            print("Smooth Validation Accuracy: {}%".format(self.smooth_acc_val))

    def train(self,train1,train1_lab,train2,train2_lab,val,val_lab,
             retrain_base=True,evaluate=True,verbose=True):

        train_time_begin = time()

        train1, train2, val = [np.array(data).astype("int8") for data in [train1, train2, val]]
        train1_lab, train2_lab, val_lab = [np.array(data).astype("int16") for data in [train1_lab, train2_lab, val_lab]]

        # Store both training data in one np.array for memory efficency
        train_split_idx = int(len(train1))
        train, train_lab = np.concatenate([train1, train2]), np.concatenate([train1_lab, train2_lab])
        del train1, train2, train1_lab, train2_lab
        
        if verbose:
            print("Training base models...")
        self._train_base(train[:train_split_idx], train_lab[:train_split_idx])

        if verbose:
            print("Training smoother...")
        self._train_smooth(train[train_split_idx:], train_lab[train_split_idx:])

        if retrain_base:
            # Re-using the smoother-training-data to re-train the base models
            if verbose:
                print("Re-training base models...")
            self._train_base(train, train_lab)

        if self.calibrate:
            # calibrates the predictions to be balanced w.r.t. the train1 class distribution
            if verbose:
                print("Calibrating...")
            calibrate_light = int(0.05*train_split_idx)
            calibrate_idxs = np.random.choice(train_split_idx,calibrate_light,replace=False)
            zs = self.predict_proba(train[calibrate_idxs],rtn_calibrated=False).reshape(-1,self.num_anc)
            self.calibrator = calibrator_module(zs, train_lab[calibrate_idxs].reshape(-1), self.num_anc, method ='Isotonic')        
            del zs 

        # Evaluate model
        if evaluate:
            if verbose:
                print("Evaluating model...")
            self._evaluate_base(train[:train_split_idx], train_lab[:train_split_idx],   val,val_lab)
            self._evaluate_smooth(train[train_split_idx:], train_lab[train_split_idx:], val,val_lab)

        # Save model
        if self.save is not None:
            pickle.dump(self, open(self.save+"model.pkl", "wb" ))

        self.training_time = time() - train_time_begin


    def _mode(self, arr):
        mode = stats.mode(arr)[0][0]
        if mode == -stats.mode(-arr)[0][0]:
            return mode # if mode is unambiguous
        else:
            return arr[len(arr)//2] # else return the center (default value)

    def _mode_filter(self, pred, size):
        if not size:
            return pred # if size is 0 or None
        pred_out = np.copy(pred)
        ends = size//2
        for i in range(len(pred))[ends:-ends]:
            pred_out[i] = self._mode(pred[i-ends:i+ends+1])
        
        return pred_out

    def predict(self,tt,rtn_calibrated=None,phase=False):
        if phase:
            X_phased, y_phased = self.phase(tt, calibrate=rtn_calibrated)
            return y_phased
        if rtn_calibrated is None:
            rtn_calibrated = self.calibrate
        if rtn_calibrated:
            y_cal_probs = self.predict_proba(tt,rtn_calibrated=True)
            y_preds = np.argmax(y_cal_probs, axis = 2)
        else:    
            n,_ = tt.shape
            tt,_ = self._get_smooth_data(tt)
            y_preds = self.smooth.predict(tt).reshape(n,len(self.base))
        
        if self.mode_filter_size:
            y_preds = np.apply_along_axis(func1d=self._mode_filter, axis=1, arr=y_preds, size=self.mode_filter_size)

        return y_preds

    def predict_proba(self,tt,rtn_calibrated=None):

        if rtn_calibrated is None:
            rtn_calibrated = self.calibrate

        n,_ = tt.shape
        tt,_ = self._get_smooth_data(tt)
        proba = self.smooth.predict_proba(tt).reshape(n,-1,self.num_anc)

        if rtn_calibrated:
            if self.calibrator is not None:
                proba_flatten=proba.reshape(-1,self.num_anc)
                iso_prob=np.zeros((proba_flatten.shape[0],self.num_anc))
                for i in range(self.num_anc):    
                    iso_prob[:,i] = self.calibrator[i].transform(proba_flatten[:,i])
                proba = normalize_prob(iso_prob, self.num_anc).reshape(n,-1,self.num_anc)
            else:
                print("No calibrator found, returning uncalibrated probabilities")

        return proba

    def write_config(self,fname):
        with open(fname,"w") as f:
            for attr in dir(self):
                val = getattr(self,attr)
                if type(val) in [int,float,str] :
                    f.write("{}\t{}\n".format(attr,val))

    def phase(self,X,verbose=False):
        """
        Wrapper for XGFix
        """
        n_haplo, n_snp = X.shape
        n_ind = n_haplo//2
        X_phased = np.zeros((n_ind,2,n_snp))
        Y_phased = np.zeros((n_ind,2,self.num_windows))

        for i, X_i in enumerate(X.reshape(n_ind,2,n_snp)):
            sys.stdout.write("\rPhasing individual %i/%i" % (i+1, n_ind))
            X_m, X_p = X_i
            X_m, X_p, Y_m, Y_p, history, XGFix_tracker = XGFix(X_m, X_p, self, verbose=verbose)
            X_phased[i] = np.copy(np.array((X_m,X_p)))
            Y_phased[i] = np.copy(np.array((Y_m,Y_p)))

        print()
        
        return X_phased.reshape(n_haplo, n_snp), Y_phased.reshape(n_haplo, self.num_windows)


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

    # This is temorary while there are still pre-trained models with no calibrate members
    try:
        model.calibrate
    except AttributeError:
        model.calibrate = None

    # Same for mode filter
    try:
        model.mode_filter_size
    except AttributeError:
        model.mode_filter_size = 5

    return model

def cM2nsnp(cM, chm, chm_len_pos, genetic_map_file):
    
    gen_map_df = pd.read_csv(genetic_map_file, sep="\t", comment="#", header=None, dtype="str")
    gen_map_df.columns = ["chm", "pos", "pos_cm"]
    gen_map_df = gen_map_df.astype({'chm': str, 'pos': np.int64, 'pos_cm': np.float64})
    gen_map_df = gen_map_df[gen_map_df.chm == chm]

    chm_len_cM = np.array(gen_map_df["pos_cm"])[-1]
    snp_len = int(round(cM*(chm_len_pos/chm_len_cM)))

    return snp_len

def train(chm, model_name, genetic_map_file, data_path, generations, window_size_cM, smooth_size, missing, n_cores, verbose):

    if verbose:
        print("Preprocessing data...")
    
    # ------------------ Config ------------------
    model_name += "_chm_" + chm
    model_repo = join_paths("./"+instance_name, "models", verb=False)
    model_repo = join_paths(model_repo, model_name, verb=False)
    model_path = model_repo + "/" + model_name + ".pkl"

    train1_paths = [data_path + "/chm" + chm + "/simulation_output/train1/gen_" + str(gen) + "/" for gen in generations]
    train2_paths = [data_path + "/chm" + chm + "/simulation_output/train2/gen_" + str(gen) + "/" for gen in generations]
    val_paths    = [data_path + "/chm" + chm + "/simulation_output/val/gen_"    + str(gen) + "/" for gen in generations]

    position_map_file   = data_path + "/chm"+ chm + "/positions.txt"
    reference_map_file  = data_path + "/chm"+ chm + "/references.txt"
    population_map_file = data_path + "/populations.txt"

    snp_pos = np.loadtxt(position_map_file,  delimiter='\n').astype("int")
    snp_ref = np.loadtxt(reference_map_file, delimiter='\n', dtype=str)
    pop_order = np.genfromtxt(population_map_file, dtype="str")
    chm_len = len(snp_pos)
    num_anc = len(pop_order)

    window_size_pos = cM2nsnp(cM=window_size_cM, chm=chm, chm_len_pos=chm_len, genetic_map_file=genetic_map_file)
    
    # ------------------ Process data ------------------
    # gather feature data files (binary representation of variants)
    X_fname = "mat_vcf_2d.npy"
    X_train1_files = [p + X_fname for p in train1_paths]
    X_train2_files = [p + X_fname for p in train2_paths]
    X_val_files    = [p + X_fname for p in val_paths]

    # gather label data files (population)
    labels_fname = "mat_map.npy"
    labels_train1_files = [p + labels_fname for p in train1_paths]
    labels_train2_files = [p + labels_fname for p in train2_paths]
    labels_val_files    = [p + labels_fname for p in val_paths]

    # load the data
    train_val_files = [X_train1_files, labels_train1_files, X_train2_files, labels_train2_files, X_val_files, labels_val_files]
    X_train1_raw, labels_train1_raw, X_train2_raw, labels_train2_raw, X_val_raw, labels_val_raw = [load_np_data(f) for f in train_val_files]

    # adding generation 0
    if gen_0:
        if verbose:
            print("Including generation 0...")
        
        # get it
        gen_0_sets = ["train1", "train2"]
        X_train1_raw_gen_0, y_train1_raw_gen_0, X_train2_raw_gen_0, y_train2_raw_gen_0 = get_gen_0(data_path + "/chm" + chm, population_map_file, gen_0_sets)

        # add it
        X_train1_raw = np.concatenate([X_train1_raw, X_train1_raw_gen_0])
        labels_train1_raw = np.concatenate([labels_train1_raw, y_train1_raw_gen_0])
        X_train2_raw = np.concatenate([X_train2_raw, X_train2_raw_gen_0])
        labels_train2_raw = np.concatenate([labels_train2_raw, y_train2_raw_gen_0])

        # delete it
        del X_train1_raw_gen_0, y_train1_raw_gen_0, X_train2_raw_gen_0, y_train2_raw_gen_0 

    # reshape according to window size 
    X_train1, labels_window_train1 = data_process(X_train1_raw, labels_train1_raw, window_size_pos, missing)
    X_train2, labels_window_train2 = data_process(X_train2_raw, labels_train2_raw, window_size_pos, missing)
    X_val, labels_window_val       = data_process(X_val_raw, labels_val_raw, window_size_pos, missing)

    del X_train1_raw, X_train2_raw, X_val_raw, labels_train1_raw, labels_train2_raw, labels_val_raw

    # ------------------ Train model ------------------    
    # init, train, evaluate and save model
    if verbose:
        print("Initializing XGMix model and training...")
    model = XGMIX(chm_len, window_size_pos, smooth_size, num_anc, snp_pos, snp_ref, pop_order, calibrate=calibrate, cores=n_cores)
    model.train(X_train1, labels_window_train1, X_train2, labels_window_train2, X_val, labels_window_val, retrain_base=retrain_base, verbose=verbose)

    # evaluate model
    analysis_path = join_paths(model_repo, "analysis", verb=False)
    CM(labels_window_val.ravel(), model.predict(X_val).ravel(), pop_order, analysis_path, verbose)
    pickle.dump(model, open(model_path,"wb"))


    # write the model parameters of type int, float, str into a file config.
    # so there is more clarity on what the model parameters were.
    # NOTE: Not tested fully yet. # TODO
    model_config_path = os.path.join(model_repo,"config.txt")
    model.write_config(model_config_path)

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
    return cm

def main(args, verbose=True):

    # Either load pre-trained model or simulate data from reference file, init model and train it
    if mode == "pre-trained":
        model = load_model(args.path_to_model, verbose=verbose)
    elif args.mode == "train":

        # Set output path
        data_path = join_paths('./'+instance_name, 'generated_data', verb=False)

        # Running simulation. If data is already simulated, skipping can save a lot of time
        if run_simulation:

            # Splitting the data into train1 (base), train2 (smoother), val, test 
            if verbose:
                print("Reading sample maps and splitting in train/val...")
            samples, pop_ids = read_sample_map(args.sample_map_file, population_path = data_path)
            set_names = ["train1", "train2", "val"]
            sample_map_path = join_paths(data_path, "sample_maps", verb=verbose)
            sample_map_paths = [sample_map_path+"/"+s+".map" for s in set_names]
            sample_map_idxs = split_sample_map(sample_ids = np.array(samples["Sample"]),
                                                populations = np.array(samples["Population"]),
                                                ratios = founders_ratios,
                                                pop_ids = pop_ids,
                                                sample_map_paths=sample_map_paths)

            # Simulating data
            if verbose:
                print("Running simulation...")
            num_outs_per_gen = [n//len(generations) for n in num_outs]
            main_admixture(args.chm, data_path, set_names, sample_map_paths, sample_map_idxs,
                           args.reference_file, args.genetic_map_file, num_outs_per_gen, generations)

            if verbose:
                print("Simulation done.")
                print("-"*80+"\n"+"-"*80+"\n"+"-"*80)
        else:
            print("Using simulated data from " + data_path + " ...")

        # Processing data, init and training model
        model = train(args.chm, model_name, args.genetic_map_file, data_path, generations,
                        window_size_cM, smooth_size, missing, n_cores, verbose)
        if verbose:
            print("-"*80+"\n"+"-"*80+"\n"+"-"*80)

    # Predict the query data
    if args.query_file is not None:
        # Load and process user query vcf file
        if verbose:
            print("Loading and processing query file...")
        query_vcf_data = read_vcf(args.query_file, chm=args.chm)
        X_query = vcf_to_npy(query_vcf_data, model.snp_pos, model.snp_ref, verbose=verbose)

        # predict and finding effective prediction for intersection of query SNPs and model SNPs positions
        if verbose:
            print("Inferring ancestry on query data...")
        if args.phase:
            X_query_phased, label_pred_query_window = model.phase(X_query)
        else: 
            label_pred_query_window = model.predict(X_query)

        # writing the result to disc
        if verbose:
            print("Writing inference to disc...")
        msp_data = get_msp_data(args.chm, label_pred_query_window, model.snp_pos,
                                query_vcf_data['variants/POS'], model.num_windows,
                                model.win, args.genetic_map_file)
        write_msp_tsv(args.output_basename, msp_data, model.population_order, query_vcf_data['samples'])

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
    if len(sys.argv) == 7:
        mode = "pre-trained" 
    if len(sys.argv) == 8:
        mode = "train"

    # Usage message
    if mode is None:
        if len(sys.argv) > 1:
            print("Error: Incorrect number of arguments.")
        print("Usage when training a model from scratch:")
        print("   $ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <reference_file> <sample_map_file>")
        print("Usage when using a pre-trained model:")
        print("   $ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <phase> <path_to_model>")
        sys.exit(0)

    # Deconstruct CL arguments
    base_args = {
        'mode': mode,
        'query_file': sys.argv[1] if sys.argv[1].strip() != "None" else None,
        'genetic_map_file': sys.argv[2],
        'output_basename': sys.argv[3],
        'chm': sys.argv[4],
        'phase': True if sys.argv[5].lower() == "true" else False
    }
    args = Struct(**base_args)
    if mode == "train":
        args.reference_file  = sys.argv[6]
        args.sample_map_file = sys.argv[7]
    elif mode == "pre-trained":
        args.path_to_model = sys.argv[6]

    # Run it
    if verbose:
        print("Launching XGMix in", mode, "mode...")
    main(args)
