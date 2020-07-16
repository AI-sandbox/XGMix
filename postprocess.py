import allel
from collections import Counter
import gzip
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import sys

def read_vcf(vcf_file, verbose=True):
    """
    Reads vcf files into a dictionary
    fields="*" extracts more information, take out if ruled unecessary
    """
    if vcf_file[-3:]==".gz":
        with gzip.open(vcf_file, 'rb') as vcf:
            data = allel.read_vcf(vcf) #, fields="*")
    else: 
        data = allel.read_vcf(vcf_file) #, fields="*")

    if verbose:    
        chmlen, n, _ = data["calldata/GT"].shape
        print("File read:", chmlen, "SNPs for", n, "individuals")

    return data

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

def snp_intersection(pos1, pos2, verbose=False):
    
    # find indicese of intersection
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

    try:
        len_ratio2 = len(intersection)/len(pos2)
    except ZeroDivisionError:
        print("Error: No SNPs of specified chromosome found in query file.")
        print("Exiting...")
        sys.exit(0)

    if verbose:
        print("Number of SNPs from model:", len(pos1))
        print("Number of SNPs from file:", len(pos2))
        print("Number of intersecting SNPs:", len(intersection))
        print("Ratio of matching SNPs from file:", round(len_ratio2,4))

    return idx1, idx2


def vcf_to_npy(vcf_fname, chm, snp_pos_fmt=None, miss_fill=2, verbose=True):
    """
    Converts vcf file to numpy matrix. If SNP position format is specified, then
    accompany that format by filling in values of missing positions and ignoring
    additional positions.
    """
    
    # unzip and read vcf
    vcf_data = read_vcf(vcf_fname, verbose)
    chm_idx = vcf_data['variants/CHROM']==str(chm)
    
    # convert to np array 
    chm_data = vcf_data["calldata/GT"][chm_idx,:,:]
    chm_len, nout, _ = chm_data.shape
    mat_vcf_2d = chm_data.reshape(chm_len,nout*2).T
    
    # matching SNP positions with format
    vcf_pos = None
    if snp_pos_fmt is not None:
        vcf_pos = vcf_data['variants/POS'][chm_idx]
        fmt_idx, vcf_idx = snp_intersection(snp_pos_fmt, vcf_pos, verbose=verbose)
        fill = np.full((nout*2, len(snp_pos_fmt)), miss_fill)
        fill[:,fmt_idx] = mat_vcf_2d[:,vcf_idx]
        mat_vcf_2d = fill
        effective_vcf_pos = vcf_data['variants/POS'][chm_idx][vcf_idx]

    return mat_vcf_2d, vcf_pos, effective_vcf_pos, fmt_idx, vcf_idx, vcf_data['samples']

def get_effective_pred(prediction, chm_len, window_size, model_idx):
    """
    Maps SNP indices to window number to find predictions for those SNPs
    """

    # expanding prediction
    ext = np.repeat(prediction, window_size, axis=1)

    # handling remainder
    rem_len = chm_len-ext.shape[1]
    ext_rem = np.tile(prediction[:,-1], [rem_len,1]).T
    ext = np.concatenate([ext, ext_rem], axis=1)

    # return relevant positions
    return ext[:, model_idx]


def get_msp_data(chm, pred_wind, model_pos, query_pos, n_wind, wind_size, genetic_map_file):

    model_chm_len = len(model_pos)
    
    gen_map_df = pd.read_csv(genetic_map_file, sep="\t", comment="#", header=None, dtype="str")
    gen_map_df.columns = ["chm", "pos", "pos_cm"]
    gen_map_df = gen_map_df.astype({'chm': str, 'pos': np.int64, 'pos_cm': np.float64})
    gen_map_df = gen_map_df[gen_map_df.chm == chm]
    
    # chm
    chm_array = [chm]*n_wind

    # start and end pyshical positions
    spos_idx = np.arange(0, model_chm_len, wind_size)[:-1]
    epos_idx = np.concatenate([np.arange(0, model_chm_len, wind_size)[1:-1],np.array([model_chm_len])])-1
    spos = model_pos[spos_idx]
    epos = model_pos[epos_idx]

    # start and end positions in cM (using linear interpolation)
    f = interp1d(gen_map_df.pos, gen_map_df.pos_cm, fill_value="extrapolate") 
    sgpos = np.round(f(spos),5)
    egpos = np.round(f(epos),5)

    # number of query snps in interval
    wind_index = [min(n_wind-1, np.where(q == sorted(np.concatenate([epos, [q]])))[0][0]) for q in query_pos]
    window_count = Counter(wind_index)
    n_snps = [window_count[w] for w in range(n_wind)]

    # Concat with prediction table
    meta = np.array([chm_array, spos, epos, sgpos, egpos, n_snps]).T
    msp_data = np.concatenate([meta, pred_wind.T], axis=1)
    msp_data = msp_data.astype(str)

    return msp_data
    
def write_msp_tsv(output_basename, msp_data, populations, query_samples):
    
    meta_col_names = ["chm", "spos", "epos", "sgpos", "egpos", "n snps"]
    
    with open("./"+output_basename+".msp.tsv", 'w') as f:
        # first line (comment)
        f.write("#Subpopulation order/codes: ")
        f.write("\t".join([str(pop)+"="+str(i) for i, pop in enumerate(populations)])+"\n")
        # second line (comment/header)
        f.write("#"+"\t".join(meta_col_names) + "\t")
        f.write("\t".join([str(s) for s in np.concatenate([[s+".0",s+".1"] for s in query_samples])])+"\n")
        # rest of the lines (data)
        for l in range(msp_data.shape[0]):
            f.write("\t".join(msp_data[l,:]))
            f.write("\n")