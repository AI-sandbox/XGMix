#2#

import numpy as np
import os
import xgboost as xgb
import pickle
import argparse
import logging

from losses import return_loss_function
from metrics import return_metric_function
from preprocess import post_parse, data_process

def parse_args():
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--train_data', type=str, help='path to directory with training npy files', required=True)
    parser.add_argument('--val_data', type=str, help='path to directory with validation npy files', required=True)
    parser.add_argument('--task', type=str, help='classification or regression', choices=['c','r'], required=True)
    parser.add_argument('--save',type=str, help='where to store the model files', required=True)
    parser.add_argument('--coord_map', default = None, type=str, \
        help='mapping labels to coordinates. Format: label coordinate in each line')

    parser.add_argument('--loss',type=str, default=None, help='loss function')
    parser.add_argument('--metric', type=str, default=None, help='validation metric')

    parser.add_argument('--window_size', default=500, type=int, help='window size')
    parser.add_argument('--missing', default=0.0, type=float, help='missing snps percent')

    parser.add_argument('--trees', default=100, type=int, help='n_estimators argument in xgboost')
    parser.add_argument('--max_depth', default=4, type=int, help='max_depth argument in xgboost')
    parser.add_argument('--overwrite', action='store_true', help='overwrite if directory already exists')
    
    args = parser.parse_args()
    args = post_parse(args)
    return args


# the following could be done in parallel for sure.
def train_multiple(train,train_lab,val,val_lab,args):

    logging.basicConfig(level=logging.INFO, filename=args.save+'/base.log')
    logging.info("Begin training")
    logging.info("Number of windows: {}".format(train.shape[1]))

    eval_metric_func = return_metric_function(args.metric)
    loss_func = return_loss_function(args.loss) if args.loss is not None else None
    fit_model = None
    if args.task == 'r':
        fit_model = xgb.XGBRegressor
    elif args.task =='c':
        fit_model = xgb.XGBClassifier


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
        if loss_func == None:
            model = fit_model(n_estimators=args.trees,max_depth=args.max_depth,
                learning_rate=0.1,verbosity=1,reg_lambda=1,missing=2.0)
            model.fit(tt,ll_t)

        else:
            # TODO: ADD LOSS FUNCTION
            model = fit_model(n_estimators=args.trees,max_depth=args.max_depth,
                learning_rate=0.1,verbosity=1,reg_lambda=1,missing=2.0,objective=loss_func)
            model.fit(tt,ll_t)

        y_pred = model.predict(tt)
        train_metric = eval_metric_func(y_pred,ll_t)
        train_accr.append(train_metric)

        y_pred = model.predict(vt)
        val_metric = eval_metric_func(y_pred,ll_v)
        val_accr.append(val_metric)

        models["model"+str(idx*args.window_size)] = model

        if idx%10 == 0:
            logging.info("Train iteration: {}, ".format(idx))
            logging.info("Train {0}: {1}, Val {0}: {2}".format(args.metric, np.mean(train_accr), np.mean(val_accr)))

    return models,train_accr,val_accr

if __name__=="__main__":

    args = parse_args()
    training, training_labels, validation, validation_labels = data_process(args)

    print("Model will be saved at: {}".format(args.save))

    # Run training

    models,train_accr,val_accr = train_multiple(training,training_labels,validation,validation_labels,args)
    # Save
    pickle.dump(models, open(args.save+"/model.pkl", "wb" ))

