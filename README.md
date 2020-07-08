# XGMix-models

XGMIX.py loads and makes use of the pre-trained models for a given query_file and a chromosome.

## Usage

Usage:

$ python3 XGMIX.py <query_file> <output_basename> <chm>

## Output

The predictions are written in <output_basename>.tsv.

The first line is a comment line, that specifies the order of subpopulations: eg:
#reference_panel_population: golden_retriever labrador_retriever poodle poodle_small

The second line specifies the column names, and every following line marks the genome position.

The first four columns specify
- the chromosome
- genetic marker's physical position in basepair units
- genetic position in centiMorgans (Currently missing)
- the genetic marker's numerical index in the rfmix genetic map input file (Currently missing)
The remaining columns give the predicted reference panel population. A genotype has two haplotypes, so the number of probabilities for a genotype is 2*(number of reference panel populations). The number of columns in the file is 4 + 2*(number of genotypes)
