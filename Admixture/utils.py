import allel
from collections import Counter, OrderedDict
import gzip
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from operator import itemgetter
import os
import sys
import re

# -------------- utils --------------

def run_shell_cmd(cmd, verb=True):
    if verb:
        print("Running: %s\n" % cmd,"\n")
    rval = os.system(cmd)
    if rval != 0:
        signal = rval & 0xFF
        exit_code = rval >> 8
        if signal != 0:
            sys.stderr.write("\nCommand %s exits with signal %d\n\n" % (cmd, signal))
            sys.exit(signal)
        sys.stderr.write("\nCommand %s failed with return code %d\n\n" % (cmd, exit_code))
        sys.exit(exit_code)

def read_vcf(vcf_file, verb=True):
    """
    Reads vcf files into a dictionary
    fields="*" extracts more information, take out if ruled unecessary
    """
    if vcf_file[-3:]==".gz":
        with gzip.open(vcf_file, 'rb') as vcf:
            data = allel.read_vcf(vcf) #, fields="*")
    else: 
        data = allel.read_vcf(vcf_file) #, fields="*")

    if verb:    
        chmlen, n, _ = data["calldata/GT"].shape
        print("File read:", chmlen, "SNPs for", n, "individuals")

    return data

def join_paths(p1,p2="",verb=True):
    path = os.path.join(p1,p2)
    if not os.path.exists(path):
        os.makedirs(path)
        if verb:
            print("path created:", path)
    return path

def convert_to_bcf(vcf_file, output_path):
    bcf_path = re.sub('.vcf(|.gz|.bgz)$', '.bcf.gz', vcf_file)    
    
    if not os.path.exists(bcf_path):

        path = join_paths(output_path, "temp_files")
        bcf_path = path + "/reference.bcf.gz"
        
        print("Converting " + vcf_file + " to .bcf.gz format ...")
        convert_to_bcf = "bcftools view --output-type b --output-file %s --threads 12 %s" % (bcf_path, vcf_file)
        run_shell_cmd(convert_to_bcf)
        
        indexing_bcf = "bcftools index -f %s" % (bcf_path)
        run_shell_cmd(indexing_bcf)
    
    return bcf_path

# -------------- sample map/split utils --------------

def sample_map_to_matrix(map_path):

    ff = open(map_path, "r", encoding="latin1")
    matrix = []
    loc_func = lambda x: ord(x.rstrip("\n"))
    for i in ff.readlines()[1:]:
        row = i.split("\t")[2:]
        row = np.vectorize(loc_func)(row)
        matrix.append(row-49)
    matrix = np.asarray(matrix).T
    ff.close()

    return matrix

def _sort_pop(population):
    pop = Counter(population)
    sorted_pop = OrderedDict(sorted(pop.items(), key=itemgetter(1), reverse=True))
    return sorted_pop

def plot_pop_dist(population):
    sorted_pop = _sort_pop(population)
    figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(list(sorted_pop.keys()), list(sorted_pop.values()))
    plt.xticks(rotation=90, fontsize=8)
    plt.show()

def down_sample_pop_(sample, max_samples=60):
    pop = _sort_pop(sample["Population"])
    oversized = {k:v for (k,v) in pop.items() if v > max_samples}
    for o in oversized:
        temp = sample[sample["Population"] == o]
        for i, dat in temp.sample(n=len(temp) - max_samples).iterrows():
            sample.drop([i],inplace=True)

def split_map(sample_data, train_ratio=0.7, val_ratio=0.2):
    """
    Takes in vcf data in a pandas dataframe. Splits the data into train, val and test set. 
    Could be extended to take superpopulation as input and filter.
    
    Procedure:
    - If only one - straight up put in training set
    - If >5, split train_ratio-val_ratio-test_ratio.
    - Take care of specific cases
    - For populations with 2,3,4 individuals take care of them individually.
    """
    train_list, val_list, test_list = [], [], []
    
    cnt = 0
    for idx, dat in sample_data.iterrows():
        num = np.random.rand()
        sample1, pop1 = dat["Sample"], dat["Population"]
        if num >= 0 and num < train_ratio:
            train_list.append([sample1,pop1])
        elif num >= 0.7 and num < (train_ratio+val_ratio):
            val_list.append([sample1,pop1])
        else:
            test_list.append([sample1,pop1])

    train_list = sorted(train_list, key=lambda train_list: train_list[1])
    val_list = sorted(val_list, key=lambda val_list: val_list[1])
    test_list = sorted(test_list, key=lambda test_list: test_list[1])

    return train_list, val_list, test_list

def write_sample_map(sample_map, fname, verb=False):
    if verb:
        print("writing sample map:", fname)

    with open(fname,"w") as f:
        for i in sample_map:
            f.write("{}\t{}\n".format(i[0],i[1]))
    
    return fname

def get_sample_map_file_idxs(fname, pop_ids):
    
    with open(fname,"r") as f:
        seen = []
        for i in f.readlines():
            popname = i.split("\t")[1].rstrip("\n")
            popname = pop_ids[popname]
            if popname not in seen:
                seen.append(popname)
        idx_to_pop_map = dict(zip(range(len(seen)),seen))
    
    return idx_to_pop_map

def get_lat_lon(sample_data, pop_ids):
    lat_lon = {}
    for k in pop_ids.keys():
        for i,dat in sample_data.iterrows():
            if dat["Population"] == k:
                lat_lon[pop_ids[k]] = [dat["Latitude"], dat["Longitude"]]
                break
    return lat_lon


