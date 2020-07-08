import allel
import gzip
import numpy as np

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

    try:
        len_ratio2 = len(intersection)/len(pos2)
    except ZeroDivisionError:
        print("Error: No SNPs of specified chromosome found in query file.")
        print("Exiting...")
        exit()

    if verbose:
        print("Number of SNPs from model:", len(pos1))
        print("Number of SNPs from file:", len(pos2))
        print("Number of intersecting SNPs:", len(intersection))
        print("Ratio of matching SNPs from file:", round(len_ratio2,4))

    return idx1, idx2


def vcf_to_npy(vcf_fname, chm, snp_pos_fmt=None, miss_fill=2, verbose=True):
    
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

    return mat_vcf_2d, vcf_pos, effective_vcf_pos, fmt_idx, vcf_idx

def get_effective_pred(prediction, chm_len, window_size, model_idx):
    win_idx = np.concatenate([np.arange(0, chm_len, window_size)[:-1],np.array([chm_len-1])])
    query_window = [sum(win_idx < i) for i in model_idx]
    pred_eff = prediction[:,query_window]

    return pred_eff

def write_fb(output_basename, pred_eff, query_pos_eff, populations, chm):
    maternal, paternal = np.split(pred_eff, 2, axis=0)
    n_ind = maternal.shape[0]
    n_pos = len(query_pos_eff)
    n_col = 4 + 2*n_ind
    with open("./"+output_basename+".tsv", 'w') as f:
        f.write("#reference_panel_population: " + " ".join(populations)+"\n")
        f.write("\t".join(["col_" + str(c) for c in range(n_col)])+"\n") # TODO: What are the column names?
        for p, pos in enumerate(query_pos_eff):
            f.write(chm)
            f.write("\t" + str(pos))
            f.write("\t" + "-")
            f.write("\t" + "-")
            for ind in range(n_ind):
                f.write("\t" + str(maternal[ind, p]))
                f.write("\t" + str(paternal[ind, p]))
            f.write("\n")