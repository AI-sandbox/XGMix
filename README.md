# XGMix: Local-Ancestry Inference With Stacked XGBoost

XGMIX.py loads and uses pre-trained XGMix models to predict reference panels for a given query_file and chromosome. The models are trained on build 37 references from the following continents: *AFR AHG EAS EUR NAT OCE SAS WAS* and labels and predicts them as 0, 1, .., 7 respectively.


## Requirements

To use this script install:
```
  $ pip install xgboost
  $ pip install scikit-learn
  $ pip install scikit-allel
```

## Usage

For execution run:
```
$ python3 XGMIX.py <query_file> <output_basename> <chr_nr>
```

where <query_file> is a .vcf or .vcf.gz reference file containing the query references and <chr_nr> is the chromosome number. The predictions are written in <output_basename>.msp.tsv .

## Output

The first line is a comment line, that specifies the order of subpopulations: eg:
#reference_panel_population: golden_retriever labrador_retriever poodle poodle_small

The second line specifies the column names, and every following line marks the genome position.

The first four columns specify
- the chromosome
- genetic marker's physical position in basepair units
- genetic position in centiMorgans (Currently missing)
- the genetic marker's numerical index in the genetic map file (Currently missing)

The remaining columns give the predicted reference panel population. A genotype has two haplotypes, so the number of predictions for a genotype is 2*(number of genotypes) and therefore the total number of columns in the file is 4 + 2*(number of genotypes)

## Pre-trained Models

The pre-trained models are trained on various SNP positions from a build 37 reference file. When making predictions, the input to the model is an intersection of the pre-trained model SNP positions and the SNP positions from the <query_file>. The set of positions that's only in the original training input is encoded as missing and the set of positions only in the <query_file> is discarded. When the script is executed, it will log the size of the intersection as the performance will depend on how much of the original positions are missing.

NOTE: TEMPORARILY ONLY OFFERING CHROMOSOME 21 and 22

## Cite

#### When using this software, please cite: Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A., XGMix: Local-Ancestry Inference With Stacked XGBoost, International Conference on Learning Representations (ICLR, 2020, Workshop AI4H).
