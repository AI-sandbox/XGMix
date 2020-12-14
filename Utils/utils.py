import allel
from collections import Counter
import gzip
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
import sys

def get_num_outs(sample_map_paths, r_admixed=1.0):
    # r_admixed: generated r_admixed * num-founders for each set
    # TODO: cap train2 lengths to a pre-defined value.
    num_outs = []
    for path in sample_map_paths:
        with open(path,"r") as f:
            length = len(f.readlines()) # how many founders.
            num_outs.append(int(length *r_admixed))
    return num_outs

def run_shell_cmd(cmd, verb=True):
    if verb:
        print("Running:", cmd)
    rval = os.system(cmd)
    if rval != 0:
        signal = rval & 0xFF
        exit_code = rval >> 8
        if signal != 0:
            sys.stderr.write("\nCommand %s exits with signal %d\n\n" % (cmd, signal))
            sys.exit(signal)
        sys.stderr.write("\nCommand %s failed with return code %d\n\n" % (cmd, exit_code))
        sys.exit(exit_code)

def join_paths(p1,p2="",verb=True):
    path = os.path.join(p1,p2)
    if not os.path.exists(path):
        os.makedirs(path)
        if verb:
            print("path created:", path)
    return path

def read_vcf(vcf_file, chm=None, fields=None, verbose=False):
    """
    Reads vcf files into a dictionary
    fields="*" extracts more information, take out if ruled unecessary
    """
    if fields is None:
        fields = ['variants/CHROM', 'variants/POS', 'calldata/GT', 'variants/REF', 'samples']

    if vcf_file[-3:]==".gz":
        with gzip.open(vcf_file, 'rb') as vcf:
            data = allel.read_vcf(vcf,  region=chm, fields=fields)
    else: 
        data = allel.read_vcf(vcf_file, region=chm, fields=fields)

    if verbose:    
        chmlen, n, _ = data["calldata/GT"].shape
        print("File read:", chmlen, "SNPs for", n, "individuals")

    return data

def sample_map_to_matrix(map_path):
    """
    Handles the weird latin1 encoding of some sample maps
    """

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

def snp_intersection(pos1, pos2, verbose=False):
    """
    Finds interception of snps given two arrays of snp position 
    """

    if len(pos2) == 0:
        print("Error: No SNPs of specified chromosome found in query file.")
        print("Exiting...")
        sys.exit(0)
    
    # find indices of intersection
    idx1, idx2 = [], []
    for i2, p2 in enumerate(pos2):
        match = np.where(pos1==p2)[0]
        if len(match) == 1:
            idx1.append(int(match))
            idx2.append(i2)

    intersection = set(pos1) & set(pos2)
    if len(intersection) == 0:
        print("Error: No matching SNPs between model and query file.")
        print("Exiting...")
        sys.exit(0)

    if verbose:
        print("- Number of SNPs from model:", len(pos1))
        print("- Number of SNPs from file:",  len(pos2))
        print("- Number of intersecting SNPs:", len(intersection))
        intersect_percentage = round(len(intersection)/len(pos1),4)*100
        print("- Percentage of model SNPs covered by query file: ",
              intersect_percentage, "%", sep="")

    return idx1, idx2


def vcf_to_npy(vcf_data, snp_pos_fmt=None, snp_ref_fmt=None, miss_fill=2, verbose=True):
    """
    Converts vcf file to numpy matrix. 
    If SNP position format is specified, then comply with that format by filling in values 
    of missing positions and ignoring additional positions.
    If SNP reference variant format is specified, then comply with that format by swapping where 
    inconsistent reference variants.
    Inputs
        - vcf_data: already loaded data from a vcf file
        - snp_pos_fmt: desired SNP position format
        - snp_ref_fmt: desired reference variant format
        - miss_fill: value to fill in where there are missing snps
    Outputs
        - npy matrix on standard format
    """

    # reshape binary represntation into 2D np array 
    data = vcf_data["calldata/GT"]
    chm_len, n_ind, _ = data.shape
    data = data.reshape(chm_len,n_ind*2).T
    mat_vcf_2d = data
    vcf_idx, fmt_idx = np.arange(n_ind*2), np.arange(n_ind*2)

    if snp_pos_fmt is not None:
        # matching SNP positions with standard format (finding intersection)
        vcf_pos = vcf_data['variants/POS']
        fmt_idx, vcf_idx = snp_intersection(snp_pos_fmt, vcf_pos, verbose=verbose)
        # only use intersection of variants (fill in missing values)
        fill = np.full((n_ind*2, len(snp_pos_fmt)), miss_fill)
        fill[:,fmt_idx] = data[:,vcf_idx]
        mat_vcf_2d = fill

    if snp_ref_fmt is not None:
        # adjust binary matrix to match model format
        # - find inconsistent references
        vcf_ref = vcf_data['variants/REF']
        swap = vcf_ref[vcf_idx] != snp_ref_fmt[fmt_idx] # where to swap w.r.t. intersection
        if swap.any() and verbose:
            swap_n = sum(swap)
            swap_p = round(np.mean(swap)*100,4)
            print("- Found ", swap_n, " (", swap_p, "%) different reference variants. Adjusting...", sep="")
        # - swapping 0s and 1s where inconsistent
        fmt_swap_idx = np.array(fmt_idx)[swap]  # swap-index at model format
        mat_vcf_2d[:,fmt_swap_idx] = (mat_vcf_2d[:,fmt_swap_idx]-1)*(-1)

    # make sure all missing values are encoded as required
    missing_mask = np.logical_and(mat_vcf_2d != 0, mat_vcf_2d != 1)
    mat_vcf_2d[missing_mask] = miss_fill

    vcf_samples = vcf_data['samples']

    # return npy matrix
    return mat_vcf_2d

def cM2nsnp(cM, chm, chm_len_pos, genetic_map_file):
    
    gen_map_df = pd.read_csv(genetic_map_file, sep="\t", comment="#", header=None, dtype="str")
    gen_map_df.columns = ["chm", "pos", "pos_cm"]
    gen_map_df = gen_map_df.astype({'chm': str, 'pos': np.int64, 'pos_cm': np.float64})
    gen_map_df = gen_map_df[gen_map_df.chm == chm]

    chm_len_cM = np.array(gen_map_df["pos_cm"])[-1]
    snp_len = int(round(cM*(chm_len_pos/chm_len_cM)))

    return snp_len

def fb2proba(path_to_fb, n_wind=None):
    
    with open(path_to_fb) as f:
        header = f.readline().split("\n")[0]
        ancestry = np.array(header.split("\t")[1:])
    A = len(ancestry)
    
    fb_df = pd.read_csv(path_to_fb, sep="\t", skiprows=[0])

    samples = [s.split(":::")[0] for s in fb_df.columns[4::A*2]]
    
    # Probabilities in snp space
    fb = np.array(fb_df)[:,4:]
    C, AN = fb.shape
    N = AN//A
    fb_reshaped = fb.reshape(C, N, A)      # (C, N, A)
    proba = np.swapaxes(fb_reshaped, 0, 1) # (N, C, A)
    
    # Probabilities in window space
    if n_wind is not None:
        gen_pos = np.array(fb_df['genetic_position'])
        w_cM = np.arange(gen_pos[0], gen_pos[-1], step = gen_pos[-1]/n_wind)
        f = interp1d(gen_pos, np.arange(C), fill_value=(0, C), bounds_error=False) 
        w_idx = f(w_cM).astype(int)
        proba = proba[:,w_idx,:]
    
    return proba