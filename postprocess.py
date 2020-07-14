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
        exit()

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
    win_idx = np.concatenate([np.arange(0, chm_len, window_size)[1:-1],np.array([chm_len])])
    query_window = [sum(win_idx <= i) for i in model_idx]
    pred_eff = prediction[:,query_window]

    return pred_eff

def write_msp_tsv(output_basename, pred_eff, query_pos_eff, populations, chm, query_samples):
    """
    Writes out predictions for .msp.tsv file
    TODO:
        - vectorize for speed
    """
    with open("./"+output_basename+".msp.tsv", 'w') as f:
        f.write("#reference_panel_population: " + " ".join(populations)+"\n")
        f.write("chm \t pos \t pos_cM \t genetic_map_index \t")
        f.write("\t".join([str(s) for s in np.concatenate([[s+".0",s+".1"] for s in query_samples])])+"\n")
        for p, pos in enumerate(query_pos_eff):
            f.write(chm)
            f.write("\t" + str(pos))
            f.write("\t" + "-")
            f.write("\t" + "-")
            for ind in range(0,pred_eff.shape[0],2):
                f.write("\t" + str(pred_eff[ind, p]))
                f.write("\t" + str(pred_eff[ind+1, p]))
            f.write("\n")