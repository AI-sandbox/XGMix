import numpy as np
import os
import xgboost as xgb
import pickle
import argparse
import logging
import sklearn.metrics


class XGMIX():

    def __init__(self,win,sws,num_anc,save,base_params=[20,4],smooth_params=[100,4],missing_value=2,cores=4,lr=0.1,reg_lambda=1):

        self.win = win
        self.save = save
        self.sws = sws
        self.num_anc = num_anc
        self.trees,self.max_depth = base_params
        self.missing = missing_value
        self.lr = lr
        self.reg_lambda = reg_lambda

        self.s_trees,self.s_max_depth = smooth_params
        self.cores = cores

        pickle.dump(self, open(self.save+"/config.pkl", "wb" ))


    def _train_base(self,train,train_lab,val,val_lab):

        models = {}
        train_accr=[]
        val_accr=[]

        for idx in range(train.shape[1]):

            # a particular window across all examples
            tt = train[:,idx,:]
            vt = val[:,idx,:]
            ll_t = train_lab[:,idx]
            ll_v = val_lab[:,idx]

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

            models["model"+str(idx*self.win)] = model

            if idx%100 == 0:
                print("Train iteration: {}, ".format(idx))
                print("Training Accuracy: {}, Val Accuracy: {}".format(np.mean(train_accr), np.mean(val_accr)))

        return models,train_accr,val_accr

    def _gen_smooth_labels(self,ttl):
        new_ttl = np.zeros((ttl.shape[0],ttl.shape[1]-self.sws))

        sww = int(self.sws/2)
        for ppl,data in enumerate(ttl):
            new_ttl[ppl,:] = data[sww:-sww]
        new_ttl = new_ttl.reshape(new_ttl.shape[0] * new_ttl.shape[1])
        
        return new_ttl

    def _gen_smooth_data(self,tt,models):

        params_per_window = self.num_anc # num_anc
        tt_list = np.zeros((tt.shape[0],len(models),params_per_window))
        for i in range(len(models)):
            tt_list[:,i,:] = models["model"+str(i*self.win)].predict_proba(tt[:,i,:])

        new_tt = np.zeros((tt.shape[0],tt.shape[1]-self.sws,params_per_window*(self.sws+1)))

        for ppl,data in enumerate(tt_list):
            for win in range(new_tt.shape[1]):
                new_tt[ppl,win,:] = data[win:win+self.sws+1].ravel()


        d1,d2,d3 = new_tt.shape
        new_tt = new_tt.reshape(d1*d2,d3)

        return new_tt


    def train(self,train,train_lab,val,val_lab,smoothlite=10000):

        # smoothlite: int or False. If False train smoother on all data, else train only on that number of contexts.

        models, base_taccr, base_vaccr = self._train_base(train,train_lab,val,val_lab)
        pickle.dump(models, open(self.save+"/base.pkl", "wb" ))

        tt = self._gen_smooth_data(train,models)
        vv = self._gen_smooth_data(val,models)

        # Get the labels for the windowed data
        ttl = self._gen_smooth_labels(train_lab)
        vvl = self._gen_smooth_labels(val_lab)

        # # Smooth lite option - to use less data for the smoother - faster if the data is good
        smoothlite = smoothlite if smoothlite else len(tt)
        indices = np.random.choice(len(tt), min(smoothlite,len(tt)), replace=False)
        tt = tt[indices]
        ttl = ttl[indices]

        # Train model
        model = xgb.XGBClassifier(n_estimators=self.s_trees,max_depth=self.s_max_depth,
            learning_rate=self.lr,reg_lambda=self.reg_lambda,nthread=self.cores)
        model.fit(tt,ttl)

        # Evaluate model
        y_pred = model.predict(vv)

        print("Smooth Accuracy: {} ".format(sklearn.metrics.accuracy_score(y_pred,vvl)))
        # Save model
        pickle.dump(model, open(self.save+"/smooth.pkl", "wb" ))

def predict(tt,path):
    # data must match model's window size else error.
    models = pickle.load( open(path+"/base.pkl", "rb" ))
    model = pickle.load( open(path+"/smooth.pkl", "rb" ))
    xgm = pickle.load( open(path+"/config.pkl", "rb" ))

    params_per_window = xgm.num_anc # num_anc
    tt_list = np.zeros((tt.shape[0],len(models),params_per_window))
    for i in range(len(models)):
        tt_list[:,i,:] = models["model"+str(i*xgm.win)].predict_proba(tt[:,i,:])

    new_tt = np.zeros((tt.shape[0],tt.shape[1]-xgm.sws,params_per_window*(xgm.sws+1)))

    for ppl,data in enumerate(tt_list):
        for win in range(new_tt.shape[1]):
            new_tt[ppl,win,:] = data[win:win+xgm.sws+1].ravel()


    d1,d2,d3 = new_tt.shape
    new_tt = new_tt.reshape(d1*d2,d3)


    y_preds = model.predict(new_tt)

    return y_preds.reshape(d1,d2)



