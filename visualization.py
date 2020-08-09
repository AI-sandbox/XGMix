import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def plot_cm(cm, normalize=True, labels=None, figsize=(12,10)):
    plt.figure(figsize=figsize)
    
    # normalize w.r.t. number of samples from class
    if normalize:
        cm = cm/np.sum(cm, axis=0)
        cm = np.nan_to_num(cm, copy=False, nan=0.0)
        
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[1]))
    sns.set(font_scale=1.4) # for label size
    if labels is None:
        fig = sns.heatmap(df_cm, annot=False, annot_kws={"size": 16}) # font size
    else:
        fig = sns.heatmap(df_cm, xticklabels=labels, yticklabels=labels,
                   annot=False, annot_kws={"size": 16}) # font size
    
    plt.show()
    return fig

def plot_chm(sample_id, msp_df, img_name="chm_img"):
    
    # defining a color palette
    palette = sns.color_palette("colorblind").as_hex()
    
    # get the base of the tagore style dataframe
    nrows = msp_df.shape[0]
    default_params = pd.DataFrame({"feature": [0]*nrows, "size": [1]*nrows})
    tagore_base = msp_df[["#chm", "spos", "epos"]].join(default_params)
    tagore_base.columns = ["chm", "start", "stop", "feature", "size"]
    
    # adding data from the individual with that sample_id
    colors0 = [palette[i] for i in np.array(msp_df[sample_id+".0"])]
    colors1 = [palette[i] for i in np.array(msp_df[sample_id+".1"])]
    tagore0 = tagore_base.join(pd.DataFrame({"color": colors0, "chrCopy": 1}))
    tagore1 = tagore_base.join(pd.DataFrame({"color": colors1, "chrCopy": 2}))
    tagore_df = pd.concat([tagore0, tagore1])

    # plot the results
    tagore_df_fname = "./tagore.tsv"
    tagore_df.to_csv(tagore_df_fname, sep="\t", index=False, 
                     header = ['#chr','spos','epos','feature','size','color','ChrCopy'])
    os.system("tagore --i " + tagore_df_fname + " -p "+ img_name +  " --build hg37 -f")
    os.system("rm " + tagore_df_fname)