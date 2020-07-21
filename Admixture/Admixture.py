import allel
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import random
import sys

from Admixture.utils import *
from Admixture.simulation import simulate

def split_sample_map(OUTPUT_PATH, sample_map_file, verbose=True):

    print("Reading sample maps and splitting in train/val/test...")
    # path for the maps
    sample_map_path = join_paths(OUTPUT_PATH, "sample_maps", verb=verbose)

    # read the data
    sample = pd.read_csv(sample_map_file, sep="\t")
    sample.columns = ['Sample', 'Population']

    # Register and writing population mapping to labels: [Population, Label]
    pops = [dat["Population"] for idx, dat in sample.iterrows()]
    pop_ids = dict(zip(sorted(set(pops)),range(len(pops))))
    sorted_pop = sorted(pop_ids.items(), key=itemgetter(1))
    pop_order = [p[0] for p in sorted_pop]
    with open(OUTPUT_PATH+"/populations.txt", 'w') as f:
        f.write(" ".join(pop_order))

    # split train, val, test
    sample_train, sample_val, sample_test = split_map(sample)

    # write sample_maps to disc
    sample_fnames = [sample_map_path+"/"+s+".map" for s in ["train","val","test"]]
    [write_sample_map([sample_train, sample_val, sample_test][i], sample_fnames[i]) for i in range(3)]
    
    # Get sample file indicies
    sample_map_file_idxs = [get_sample_map_file_idxs(f, pop_ids) for f in sample_fnames]

    return sample_fnames, sample_map_file_idxs

    
def main_admixture(chm, root, sub_instance_names, sample_map_files, sample_map_files_idxs, reference_file, genetic_map_file,
    num_outs, generations = [2,4,6], use_phase_shift = False, verbose=True):

    output_path = join_paths(root, 'chm{}'.format(chm), verb=verbose)
    
    # path for simulation output
    simulation_output_path = join_paths(output_path, 'simulation_output')

    # Register and writing SNP physical positions
    ref = read_vcf(reference_file)
    np.savetxt(output_path +  "/positions.txt",ref['variants/POS'], delimiter='\n')
    np.savetxt(output_path + "/references.txt",ref['variants/REF'], delimiter='\n', fmt="%s")

    # Convert to .bcf file format if not already there (format required by rfmix-simulate)
    reference_file_bcf = convert_to_bcf(reference_file, output_path=output_path)

    # simulate for each sub-instance
    for i, instance_name in enumerate(sub_instance_names):

        if num_outs[i] > 0:
            # paths for each set
            instance_path = join_paths(simulation_output_path, instance_name, verb=verbose)
            
            simulate(reference_file_bcf, sample_map_files[i], sample_map_files_idxs[i],
                    genetic_map_file, generations, num_outs[i], instance_path, chm,
                    use_phase_shift)

if __name__ == "__main__":

    # ARGS
    instance_name, sample_map_file, genetic_map_file, reference_file = sys.argv[1:5]

    # Set output path
    root = './generated_data'
    if instance_name is not None:
        root += '/' + instance_name

    # Splitting the sample in train/val/test
    sub_instance_names = ["train", "val", "test"]
    sample_map_files, sample_map_files_idxs = split_sample_map(root, sample_map_file)
    num_outs = [700, 200, 0] # how many individuals in each sub-instance

    # Simulate for all chromosomes
    chms = np.array(range(22))+1
    for chm in chms:
        print("-"*80+"\n"+"-"*80+"\n"+"-"*80+"\n")
        print("Simulating chromosome " + chm)
        main_admixture(chm, root, sub_instance_names, sample_map_files, sample_map_files_idxs, reference_file, genetic_map_file, num_outs)


