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
from preprocess import post_parse, data_process

def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--train_data', type=str, help='path to directory with training npy files', required=True)
    parser.add_argument('--val_data', type=str, help='path to directory with validation npy files', required=True)
    parser.add_argument('--task', type=str, help='classification or regression', choices=['c','r'], required=True)
    parser.add_argument('--base', type=str, nargs='*', help='path to directories with base model', required=True)
    parser.add_argument('--save',type=str, help='where to store the model files', required=True)
    parser.add_argument('--coord_map', default = None, type=str, \
        help='mapping labels to coordinates. Format: label coordinate in each line')

    parser.add_argument('--loss',type=str, default=None, help='loss function')
    parser.add_argument('--metric', type=str, default=None, help='validation metric')


    parser.add_argument('--window_size', default=500, type=int, help='window size') # need to assert this when used.
    parser.add_argument('--smooth_size',default=20,dest='sws',type=int, help='smoother size')

    parser.add_argument('--trees', default=100, type=int, help='n_estimators argument in xgboost')
    parser.add_argument('--max_depth', default=4, type=int, help='max_depth argument in xgboost')
    parser.add_argument('--cores', default=4,type=int,help='number of cores to run on')
    parser.add_argument('--smoothlite',default=1000000000000000000,type=int,help='train smoother with less data')

    parser.add_argument('--overwrite', action='store_true', help='overwrite if directory already exists')
    parser.add_argument('--missing', default=0.0, type=float, help='missing snps percent')


    args = parser.parse_args()
    args = post_parse(args)

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

if __name__=="__main__":

    args = parse_args()
    training, training_labels, validation, validation_labels = data_process(args)

    print("Model will be saved at: {}".format(args.save))
 
    # Get the windowed data, each window is transformed
    tt = gen_smooth_data(training,args.base,args.window_size,args.sws)
    vv = gen_smooth_data(validation,args.base,args.window_size,args.sws)

    # Get the labels for the windowed data
    ttl = gen_smooth_labels(training_labels,args.sws)
    vvl = gen_smooth_labels(validation_labels,args.sws)

    print("Training data shape: ", tt.shape)

    # Smooth lite option - to use less data for the smoother - faster if the data is good
    indices = np.random.choice(len(tt), min(args.smoothlite,len(tt)), replace=False)
    tt = tt[indices]
    ttl = ttl[indices]

    loss_func = return_loss_function(args.loss) if args.loss is not None else None

    fit_model = None
    if args.task == 'r':
        fit_model = xgb.XGBRegressor
    elif args.task =='c':
        fit_model = xgb.XGBClassifier

    # Train model
    if loss_func == None:
        model = fit_model(n_estimators=args.trees,max_depth=args.max_depth,learning_rate=0.3,\
            verbosity=1,reg_lambda=1,nthread=args.cores)
        model.fit(tt,ttl)
    else:
        model = fit_model(n_estimators=args.trees,max_depth=args.max_depth,learning_rate=0.3,\
            verbosity=1,reg_lambda=1,nthread=args.cores, objective=loss_func)
        model.fit(tt,ttl)

    # Evaluate model
    eval_metric_func = return_metric_function(args.metric)
    y_pred = model.predict(vv)

    print("{}: {} ".format(args.metric, eval_metric_func(y_pred,vvl)))
    # Save model
    pickle.dump(model, open(args.save+"/model.pkl", "wb" ))
