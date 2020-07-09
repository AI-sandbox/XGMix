import numpy as np
import os
import xgboost as xgb
import pickle
import argparse
import logging
import sklearn.metrics
import sys

from postprocess import read_vcf, snp_intersection, vcf_to_npy, get_effective_pred, write_fb

class XGMIX():

    def __init__(self,chmlen,win,sws,num_anc,snp_pos=None,population_order=None, save=None,
                base_params=[20,4],smooth_params=[100,4],cores=4,lr=0.1,reg_lambda=1):

        self.chmlen = chmlen
        self.win = win
        self.save = save
        self.sws = sws
        self.num_anc = num_anc
        self.snp_pos = snp_pos
        self.population_order = population_order
        self.trees, self.max_depth = base_params
        self.missing = 2
        self.lr = lr
        self.reg_lambda = reg_lambda

        self.s_trees,self.s_max_depth = smooth_params
        self.cores = cores

        self.num_windows = self.chmlen//self.win
        self.pad_size = (1+self.sws)//2

        self.base = {}
        self.smooth = None

    def _train_base(self,train,train_lab,val,val_lab):

        train_accr=[]
        val_accr=[]

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
            model = xgb.XGBClassifier(n_estimators=self.trees,max_depth=self.max_depth,
                learning_rate=self.lr ,reg_lambda=self.reg_lambda,missing=self.missing,nthread=self.cores)
            model.fit(tt,ll_t)

            y_pred = model.predict(tt)
            train_metric = sklearn.metrics.accuracy_score(y_pred,ll_t)
            train_accr.append(train_metric)

            y_pred = model.predict(vt)
            val_metric = sklearn.metrics.accuracy_score(y_pred,ll_v)
            val_accr.append(val_metric)

            self.base["model"+str(idx*self.win)] = model

            if idx%100 == 0:
                print("Windows done: {}, ".format(idx))
                print("Base Training Accuracy: {}, Base Validation Accuracy: {}".format(np.mean(train_accr), np.mean(val_accr)))


    def _get_smooth_data(self,data,labels):

        # get base output
        base_out = np.zeros((data.shape[0],len(self.base),self.num_anc))
        for i in range(len(self.base)):

            inp = data[:,i*self.win:(i+1)*self.win]
            if i == len(self.base)-1:
                inp = data[:,i*self.win:]
            base_out[:,i,:] = self.base["model"+str(i*self.win)].predict_proba(inp)

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

        tt,ttl = self._get_smooth_data(train,train_lab)
        vv,vvl = self._get_smooth_data(val,val_lab)

        # # Smooth lite option - to use less data for the smoother - faster if the data is good
        smoothlite = smoothlite if smoothlite else len(tt)
        indices = np.random.choice(len(tt), min(smoothlite,len(tt)), replace=False)
        tt = tt[indices]
        ttl = ttl[indices]

        # Train model
        self.smooth = xgb.XGBClassifier(n_estimators=self.s_trees,max_depth=self.s_max_depth,
            learning_rate=self.lr,reg_lambda=self.reg_lambda,nthread=self.cores)
        self.smooth.fit(tt,ttl)

        # Evaluate model
        y_pred = self.smooth.predict(vv)
        t_pred = self.smooth.predict(tt)

        print("Smooth Training Accuracy: {} ".format(sklearn.metrics.accuracy_score(t_pred,ttl)))
        print("Smooth Validation Accuracy: {} ".format(sklearn.metrics.accuracy_score(y_pred,vvl)))

    def train(self,train,train_lab,val,val_lab,smoothlite=10000):

        # smoothlite: int or False. If False train smoother on all data, else train only on that number of contexts.

        self._train_base(train,train_lab,val,val_lab)
        self._train_smooth(train,train_lab,val,val_lab,smoothlite=smoothlite)

        # Save model
        if self.save is not None:
            pickle.dump(self, open(self.save+"model.pkl", "wb" ))

    def predict(self,tt):
        n,_ = tt.shape
        tt,_ = self._get_smooth_data(tt,np.zeros((2,2)))
        y_preds = self.smooth.predict(tt)

        return y_preds.reshape(n,len(self.base))


def predict(tt,path):
    # data must match model's window size else error.
    n, chmlen = tt.shape
    xgm = pickle.load( open(path, "rb" ))
    models = xgm.base
    model = xgm.smooth

    tt,_ = xgm._get_smooth_data(tt,np.zeros((2,2)))
    y_preds = model.predict(tt)

    return y_preds.reshape(n,len(models))


def main(args, model_path, verbose=True):

    # Load pre-trained model
    if verbose:
        print("Loading pre-trained model...")
    model = pickle.load(open(model_path,"rb"))

    # Load and process user query file
    if verbose:
        print("Loading and processing query file...")
    X_query, _, query_pos_eff, model_idx, _, query_samples = vcf_to_npy(args.query_file, args.chm, model.snp_pos, verbose=True)

    # predict and finding effective prediction for intersection of query SNPs and model SNPs positions
    if verbose:
        print("Making predictions for query file...")
    label_pred_query_window = model.predict(X_query)
    pred_eff = get_effective_pred(label_pred_query_window, model.chmlen, model.win, model_idx) 

    # writing the result to disc
    if verbose:
        print("Writing predictions to disc...")
    write_fb(args.output_basename, pred_eff, query_pos_eff, model.population_order, args.chm, query_samples)
    
if __name__ == "__main__":

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args = {
        'query_file': sys.argv[1],
        'output_basename': sys.argv[2],
        'chm': sys.argv[3]
    }
    args = Struct(**args)

    model_path= "./trained_models/missing_0/chm_"+args.chm+".pkl"

    main(args, model_path)