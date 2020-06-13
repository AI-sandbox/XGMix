#2#

import numpy as np
import os
import torch
#import scipy.stats

def load_data(files,file_ext):

    data = []
    for f in files:
        print("Reading {0}".format(f+'/'+file_ext))
        data.append(np.load(f+'/'+file_ext).astype(np.int8))

    data = np.concatenate(data,axis=0)
    return data

def reshape_data(data,win_size):
    drop_last = int(len(data[0])/win_size)*win_size
    data = data[:,0:drop_last]
    r,c=data.shape
    num_winds = int(c/win_size)
    return data.reshape(r,num_winds,win_size)

# this function is not used as of now, may be useful when things are parallelized.
def mode_labels(labels):
    ele,cnt=np.unique(labels,return_counts=True)
    mapper = dict(zip(ele,cnt))
    return max(mapper, key=mapper.get)


def dropout_row(data,missing_percent):
    num_drops = int(len(data)*missing_percent)
    drop_indices = np.random.choice(np.arange(len(data)),size=num_drops,replace=False)
    data[drop_indices] = 2
    return data


def post_parse(args):

    if not os.path.exists(args.save):
        os.mkdir(args.save)
    else:
        if args.overwrite == False:
            raise Exception("Directory already exists")
        else:
            os.system("rm -r {}".format(args.save))
            os.mkdir(args.save)

    args.train_files = []
    with open(args.train_data,"r") as f:
        for i in f.readlines():
            args.train_files.append(i.strip("\n"))

    args.val_files = []
    with open(args.val_data,"r") as f:
        for i in f.readlines():
            args.val_files.append(i.strip("\n"))


    if args.task == 'r' and args.metric == None:
        args.metric = 'mad'
    elif args.task == 'c' and args.metric == None:
        args.metric = 'auroc'

    return args

def data_process(args):
    if args.task == 'r':
        if args.coord_map == None:
            raise Exception("Co-ordinates not given...")
        coord_map = np.loadtxt(args.coord_map)
        coord_map = dict(zip(coord_map[:,0].astype(int),coord_map[:,1]))
        print("Co-ordinates map: {}".format(coord_map))

    # Load data
    print("Loading data...")
    training = load_data(args.train_files, "mat_vcf_2d.npy")
    training_labels = load_data(args.train_files, "mat_map.npy")
    validation = load_data(args.val_files, "mat_vcf_2d.npy")
    validation_labels = load_data(args.val_files, "mat_map.npy")
    # maybe a message showing size of training, validation arrays.

    # Reshape data
    print("Reshaping data...")
    training = reshape_data(training,args.window_size)
    training_labels = reshape_data(training_labels,args.window_size)
    validation = reshape_data(validation,args.window_size)
    validation_labels = reshape_data(validation_labels,args.window_size)


    # Process the labels - regression: apply transform and take mode, classification: take mode only
    print("Generating labels...")
    training_labels = torch.mode(torch.tensor(training_labels),dim=2)[0].numpy()
    validation_labels = torch.mode(torch.tensor(validation_labels),dim=2)[0].numpy()
    # training_labels = scipy.stats.mode(training_labels,axis=2)[0].squeeze()
    # validation_labels = scipy.stats.mode(validation_labels,axis=2)[0].squeeze()


    if args.task == 'r':
        print("Mapping labels to coordinates...")
        training_labels = np.vectorize(coord_map.get)(training_labels)
        validation_labels = np.vectorize(coord_map.get)(validation_labels)

    if args.missing != 0.0:
        print("Adding missing values...")
        training = np.apply_along_axis(dropout_row,axis=1,arr=training,missing_percent=args.missing)
        validation = np.apply_along_axis(dropout_row,axis=1,arr=validation,missing_percent=args.missing)


    return training, training_labels, validation, validation_labels

