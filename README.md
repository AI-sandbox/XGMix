# XGMix: Local-Ancestry Inference With Stacked XGBoost

This repository includes a python implemenation of XGMix, a gradient boosting tree-based local-ancestry inference (ancestry deconvolution) method.

XGMIX.py loads and uses pre-trained XGMix models to predict the ancestry for a given query_file (VCF) and chromosome number. The models are trained on build 37 references from the following continents: *AFR AHG EAS EUR NAT OCE SAS WAS* and labels and predicts them as 0, 1, .., 7 respectively.

## Usage

The dependencies are listed in *requirements.txt*. Assuming pip is already installed, they can be installed via
```
$ pip install -r requirements.txt
```

For execution run:
```
$ python3 XGMIX.py <query_file> <genetic_map_file> <output_basename> <chr_nr> <path_to_model> 
```

where 
- *<query_file>* is a .vcf or .vcf.gz file containing the query sequences (see example in the **/demo** folder)
- *<genetic_map_file>* is the genetic map file (see example in the **/demo** folder)
- *<output_basename>*.msp.tsv. is where the predictions are written (see **Output** below)
- *<chr_nr>* is the chromosome number
- *<path_to_model>* is a path to the model used for predictions (see **Pre-trained Models** below)

## Output

The first line is a comment line, that specifies the order and encoding of populations: eg:
#Sub_population order/code: golden_retriever=0 labrador_retriever=1 poodle poodle_small=2

The second line specifies the column names, and every following line marks the genome position.

The first 6 columns specify
- the chromosome
- interval of genetic marker's physical position in basepair units (one column represents the starting point and one the end point)
- interval of genetic position in centiMorgans (one column represents the starting point and one the end point)
- number of *<query_file>* SNP positions that are included in interval

The remaining columns give the predicted reference panel population for the given interval. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes) and therefore the total number of columns in the file is 6 + 2*(number of genotypes)

## Pre-trained Models

Pre-trained models are available for download from [XGMix-models](https://github.com/AI-sandbox/XGMix-models).

When making predictions, the input to the model is an intersection of the pre-trained model SNP positions and the SNP positions from the <query_file>. That means that the set of positions that's only in the original training input is encoded as missing and the set of positions only in the <query_file> is discarded. When the script is executed, it will log the intersection-ratio as the performance will depend on how much of the original positions are missing. When the intersection is low, we recommend using a model trained with high percentage of missing data.

## Cite

#### When using this software, please cite: Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A., "XGMix: Local-Ancestry Inference With Stacked XGBoost," International Conference on Learning Representations Workshops (ICLR, 2020, Workshop AI4H).

https://www.biorxiv.org/content/10.1101/2020.04.21.053876v1

```
@article{kumar2020xgmix,
  title={XGMix: Local-Ancestry Inference With Stacked XGBoost},
  author={Kumar, Arvind and Montserrat, Daniel Mas and Bustamante, Carlos and Ioannidis, Alexander},
  journal={International Conference of Learning Representations Workshops, AI4H},
  year={2020}
}
```

#### You can also include its companion paper: Montserrat, D.M., Kumar, A., Bustamante, C. and Ioannidis, A., "Addressing Ancestry Disparities in Genomic Medicine: A Geographic-aware Algorithm," International Conference on Learning Representations Workshops (ICLR, 2020, Workshop AI4CC).

https://arxiv.org/pdf/2004.12053.pdf

```
@article{montserrat2020addressing,
  title={Addressing Ancestry Disparities in Genomic Medicine: A Geographic-aware Algorithm},
  author={Montserrat, Daniel Mas and Kumar, Arvind and Bustamante, Carlos and Ioannidis, Alexander},
  journal={International Conference of Learning Representations Workshops, AI4CC},
  year={2020}
}
```



