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

from Utils.utils import run_shell_cmd, join_paths, read_vcf, vcf_to_npy, cM2nsnp
from Utils.preprocess import load_np_data, data_process, get_gen_0
from Utils.postprocess import get_msp_data, write_msp_tsv
from Utils.visualization import plot_cm, CM
from Utils.Calibration import calibrator_module, normalize_prob
from Utils.XGMix import XGMIX
from Admixture.Admixture import read_sample_map, split_sample_map, main_admixture

from XGFix.XGFIX import XGFix

from config import verbose, run_simulation, founders_ratios, num_outs, generations, rm_simulated_data
from config import model_name, window_size_cM, smooth_size, missing, n_cores
from config import retrain_base, calibrate, context_ratio, instance_name, mode_filter_size, smooth_depth

CLAIMER = 'When using this software, please cite: \n' + \
          'Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A. \n' + \
          '"XGMix: Local-Ancestry Inference With Stacked XGBoost" \n' + \
          'International Conference on Learning Representations Workshops \n' + \
          'ICLR, 2020, Workshop AI4AH \n' + \
          'https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1'

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

    # Same for context_ratio
    try:
        model.context_ratio
    except AttributeError:
        model.context_ratio = 0.0

    return model

def train(chm, model_name, genetic_map_file, data_path, generations, window_size_cM, 
          smooth_size, missing, n_cores, verbose, instance_name, 
          retrain_base, calibrate, context_ratio, mode_filter_size, smooth_depth, gen_0,
          output_basename):

    if verbose:
        print("Preprocessing data...")
    
    # ------------------ Config ------------------
    model_name += "_chm_" + chm
    model_repo = join_paths(output_basename+"./"+instance_name, "models", verb=False)
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
    model = XGMIX(chm_len, window_size_pos, smooth_size, num_anc, 
                  snp_pos, snp_ref, pop_order, calibrate=calibrate, 
                  cores=n_cores, context_ratio=context_ratio,
                  mode_filter_size=mode_filter_size, 
                  base_params = [20,4], smooth_params=[100,smooth_depth])
    # other params: mode_filter_size
    model.train(X_train1, labels_window_train1, X_train2, labels_window_train2, X_val, labels_window_val, retrain_base=retrain_base, verbose=verbose)

    # evaluate model
    analysis_path = join_paths(model_repo, "analysis", verb=False)
    CM(labels_window_val.ravel(), model.predict(X_val).ravel(), pop_order, analysis_path, verbose)
    print("Saving model at {}".format(model_path))
    pickle.dump(model, open(model_path,"wb"))


    # write the model parameters of type int, float, str into a file config.
    # so there is more clarity on what the model parameters were.
    # NOTE: Not tested fully yet. # TODO
    model_config_path = os.path.join(model_repo,"config.txt")
    print("Saving model info at {}".format(model_config_path))
    model.write_config(model_config_path)

    return model

def main(args, verbose=True, **kwargs):

    run_simulation=kwargs["run_simulation"]
    founders_ratios=kwargs["founders_ratios"]
    num_outs=kwargs["num_outs"]
    generations=kwargs["generations"]
    rm_simulated_data=kwargs["rm_simulated_data"]
    model_name=kwargs["model_name"]
    window_size_cM=kwargs["window_size_cM"]
    smooth_size=kwargs["smooth_size"]
    missing=kwargs["missing"]
    n_cores=kwargs["n_cores"]
    retrain_base=kwargs["retrain_base"]
    calibrate=kwargs["calibrate"]
    context_ratio=kwargs["context_ratio"]
    instance_name=kwargs["instance_name"]
    mode_filter_size=kwargs["mode_filter_size"]
    smooth_depth=kwargs["smooth_depth"]


    mode = args.mode # this needs to be done. master change 1.
    # The simulation can't handle generation 0, add it separetly
    gen_0 = 0 in generations
    generations = list(filter(lambda x: x != 0, generations))

    np.random.seed(94305)

    # Either load pre-trained model or simulate data from reference file, init model and train it
    if mode == "pre-trained":
        model = load_model(args.path_to_model, verbose=verbose)
    elif args.mode == "train":

        # Set output path: master change 2
        data_path = join_paths(args.output_basename+instance_name, 'generated_data', verb=False)

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
                        window_size_cM, smooth_size, missing, n_cores, verbose,
                        instance_name, retrain_base, calibrate, context_ratio,
                        mode_filter_size, smooth_depth, gen_0, args.output_basename)
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
