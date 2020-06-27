import numpy as np
import torch


"""
Pre-processing pipeline.
Functions to load data, generate labels based on window size.

"""

def load_data(files,file_ext):

    data = []
    for f in files:
        print("Reading {0}".format(f+'/'+file_ext))
        data.append(np.load(f+'/'+file_ext).astype(np.int16))

    data = np.concatenate(data,axis=0)
    return data

def reshape_data(data,win_size):
    drop_last = int(len(data[0])/win_size)*win_size
    drop_last = drop_last - win_size
    cache = data[:,drop_last:]
    data = data[:,0:drop_last]
    r,c=data.shape
    num_winds = int(c/win_size)
    return data.reshape(r,num_winds,win_size), cache


def post_parse(args):

    args.train_files = []
    with open(args.train_data,"r") as f:
        for i in f.readlines():
            args.train_files.append(i.strip("\n"))

    args.val_files = []
    with open(args.val_data,"r") as f:
        for i in f.readlines():
            args.val_files.append(i.strip("\n"))

    return args

def data_process(args):
    
    args = post_parse(args)

    # Load data
    print("Loading data...")
    training = load_data(args.train_files, "mat_vcf_2d.npy")
    training_labels = load_data(args.train_files, "mat_map.npy")
    validation = load_data(args.val_files, "mat_vcf_2d.npy")
    validation_labels = load_data(args.val_files, "mat_map.npy")
    # maybe a message showing size of training, validation arrays.

    # Reshape data
    print("Reshaping data...")
    training_labels,train_rem = reshape_data(training_labels,args.window_size)
    validation_labels,val_rem = reshape_data(validation_labels,args.window_size)
    
    # Process the labels - regression: apply transform and take mode, classification: take mode only
    print("Generating labels...")
    training_labels = torch.mode(torch.tensor(training_labels),dim=2)[0].numpy()
    rem_label = torch.mode(torch.tensor(train_rem),dim=-1)[0].numpy()
    training_labels = np.concatenate((training_labels,rem_label[:,np.newaxis]),axis=1)
    validation_labels = torch.mode(torch.tensor(validation_labels),dim=2)[0].numpy()
    rem_label = torch.mode(torch.tensor(val_rem),dim=-1)[0].numpy()
    validation_labels = np.concatenate((validation_labels,rem_label[:,np.newaxis]),axis=1)

        
    # if using in xgmix, convert back to numpy.
    training = torch.tensor(training).float()
    training_labels = torch.tensor(training_labels).long()
    validation = torch.tensor(validation).float()
    validation_labels = torch.tensor(validation_labels).long()
        
    return training, training_labels, validation, validation_labels


def dropout_row(data,missing_percent):
    num_drops = int(len(data)*missing_percent)
    drop_indices = np.random.choice(np.arange(len(data)),size=num_drops,replace=False)
    data[drop_indices] = 2
    return data

def simulate_missing_values(train,missing_percent=0.0):
    return np.apply_along_axis(dropout_row,axis=1,arr=train,missing_percent=missing_percent)


