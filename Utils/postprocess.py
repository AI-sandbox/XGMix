import allel
from collections import Counter
import gzip
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import sys

from utils import read_vcf

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


def get_msp_data(chm, pred_labels, model_pos, query_pos, n_wind, wind_size, genetic_map_file):
    """
    Transforms the predictions on a window level to a .msp file format.
        - chm: chromosome number
        - pred_labels: labels or predictions on a window level
        - model_pos: physical positions of the model input SNPs in basepair units
        - query_pos: physical positions of the query input SNPs in basepair units
        - n_wind: number of windows in model
        - wind_size: size of each window in the model
        - genetic_map_file: the input genetic map file
    """

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

    # start and end positions in cM (using linear interpolation, truncate ends of map file)
    end_pts = tuple(np.array(gen_map_df.pos_cm)[[0,-1]])
    f = interp1d(gen_map_df.pos, gen_map_df.pos_cm, fill_value=end_pts, bounds_error=False) 
    sgpos = np.round(f(spos),5)
    egpos = np.round(f(epos),5)

    # number of query snps in interval
    wind_index = [min(n_wind-1, np.where(q == sorted(np.concatenate([epos, [q]])))[0][0]) for q in query_pos]
    window_count = Counter(wind_index)
    n_snps = [window_count[w] for w in range(n_wind)]

    # Concat with prediction table
    meta = np.array([chm_array, spos, epos, sgpos, egpos, n_snps]).T
    msp_data = np.concatenate([meta, pred_labels.T], axis=1)
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