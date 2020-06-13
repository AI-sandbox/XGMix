#2#

import numpy as np
import os
import xgboost as xgb
import scipy
import pickle
import argparse
import logging

from metrics import return_metric_function
from losses import return_loss_function
from preprocess import *

def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--data', dest='val_data',type=str, help='path to directory with validation npy files', required=True)
    parser.add_argument('--task', type=str, help='classification or regression', choices=['c','r'], required=True)
    parser.add_argument('--base', type=str, nargs='*', help='path to directories with base model', required=True)
    parser.add_argument('--smooth_size',dest='sws',type=int, help='smoother size',required=True)
    parser.add_argument('--window_size', type=int, help='window size', required=True) # need to assert this when used.

    parser.add_argument('--smooth', type=str, default=None, help='path to smooth model')
    parser.add_argument('--metric', type=str, default=None, help='prediction metric')

    parser.add_argument('--missing', default=0.0, type=float, help='missing snps percent')
    parser.add_argument('--coord_map', default = None, type=str, \
        help='mapping labels to coordinates. Format: label coordinate in each line')

    args = parser.parse_args()

    if args.sws != 0 and args.smooth == None:
        raise Exception("Smooth model not given and smoothing window is non-zero")

    args.val_files = []
    with open(args.val_data,"r") as f:
        for i in f.readlines():
            args.val_files.append(i.strip("\n"))


    if args.task == 'r' and args.metric == None:
        args.metric = 'mad'
    elif args.task == 'c' and args.metric == None:
        args.metric = 'auroc'

    return args


def gen_smooth_data(tt,base_models,window_size,sws):

    all_models = []

    for f in base_models:
        all_models.append(pickle.load(open(f+"/model.pkl", "rb")))

    total_params = 0
    tt_list = []

    for idx,models in enumerate(all_models):

        if hasattr(models["model0"],"predict_proba"):
            params_per_window = 3
            tt_temp = np.zeros((tt.shape[0],len(models),params_per_window))
            for i in range(len(models)):
                tt_temp[:,i,:] = models["model"+str(i*window_size)].predict_proba(tt[:,i,:])

        else:
            params_per_window = 1
            tt_temp = np.zeros((tt.shape[0],len(models),params_per_window))
            for i in range(len(models)):
                tt_temp[:,i,:] = models["model"+str(i*window_size)].predict(tt[:,i,:])[:,np.newaxis]

        tt_list.append(tt_temp)
        total_params += params_per_window

    tt_list = np.concatenate(tt_list,2)
    new_tt = np.zeros((tt.shape[0],tt.shape[1]-sws,total_params*(sws+1)))

    for ppl,data in enumerate(tt_list):
        for win in range(new_tt.shape[1]):
            new_tt[ppl,win,:] = data[win:win+sws+1].ravel()


    d1,d2,d3 = new_tt.shape
    new_tt = new_tt.reshape(d1*d2,d3)

    return new_tt

def gen_smooth_labels(ttl,sws):
    new_ttl = np.zeros((ttl.shape[0],ttl.shape[1]-sws))

    sww = int(sws/2)
    for ppl,data in enumerate(ttl):
        new_ttl[ppl,:] = data[sww:-sww]
    new_ttl = new_ttl.reshape(new_ttl.shape[0] * new_ttl.shape[1])
    
    return new_ttl

# the following could be done in parallel for sure.
def predict_multiple(val,val_lab,args):

    eval_metric_func = return_metric_function(args.metric)
    fit_model = None
    if args.task == 'r':
        fit_model = xgb.XGBRegressor
    elif args.task =='c':
        fit_model = xgb.XGBClassifier


    models = pickle.load(open(args.base[0]+"/model.pkl", "rb"))
    val_accr=[]

    for idx in range(val.shape[1]):

        # a particular window across all examples
        vt = val[:,idx,:]
        ll_v = val_lab[:,idx]

        model = models["model"+str(idx*args.window_size)]

        y_pred = model.predict(vt)
        val_metric = eval_metric_func(y_pred,ll_v)
        val_accr.append(val_metric)

    return val_accr

if __name__=="__main__":

    args = parse_args()
    
    if args.task == 'r':
        if args.coord_map == None:
            raise Exception("Co-ordinates not given...")
        coord_map = np.loadtxt(args.coord_map)
        coord_map = dict(zip(coord_map[:,0].astype(int),coord_map[:,1]))
        print("Co-ordinates map: {}".format(coord_map))

    # Load data
    validation = load_data(args.val_files, "mat_vcf_2d.npy")
    validation_labels = load_data(args.val_files, "mat_map.npy")
    # maybe a message showing size of training, validation arrays.

    # Reshape data
    print("Reshaping data...")
    validation = reshape_data(validation,args.window_size)
    validation_labels = reshape_data(validation_labels,args.window_size)


    # Process the labels - regression: apply transform and take mode, classification: take mode only
    print("Generating labels...")
    validation_labels = scipy.stats.mode(validation_labels,axis=2)[0].squeeze()


    if args.task == 'r':
        print("Mapping labels to coordinates...")
        validation_labels = np.vectorize(coord_map.get)(validation_labels)

    if args.missing != 0.0:
        print("Adding missing values...")
        validation = np.apply_along_axis(dropout_row,axis=1,arr=validation,missing_percent=args.missing)


    if args.sws != 0:
        print("Smoothing prediction...")
 
        # Get the windowed data, each window is transformed
        vv = gen_smooth_data(validation,args.base,args.window_size,args.sws)

        # Get the labels for the windowed data
        vvl = gen_smooth_labels(validation_labels,args.sws)

        print("Data shape: ", vv.shape)

        model = pickle.load(open(args.smooth, "rb"))

        # Evaluate model
        eval_metric_func = return_metric_function(args.metric)
        y_pred = model.predict(vv)

        print("Test {}: {} ".format(args.metric, eval_metric_func(y_pred,vvl)))

    elif args.sws == 0:
        print("No smoothing prediction...")
        val_ac = predict_multiple(validation, validation_labels, args)
        print("Test {}: {} ".format(args.metric, np.mean(val_ac)))

