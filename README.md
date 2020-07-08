### Welcome to XGMix code.

XGMIX.py defines the class XGMIX which is used to train and predict. There is also a predict function - which takes as input the input chromosome and path to model file. The gives out prediction for each window. The parameters of the model are also available in the .pkl file (like window size, smoothing window size, etc...).

xgmix.ipynb walks through this process.

preprocess.py has some helper functions to load data, simulate missing values, etc... (preferred only for training).

old\_xgmix.py throws some snps away and also does not do padding - so, it has a lot of undesirable properties. 

##### When using this software, please cite: Kumar, A., Montserrat, D.M., Bustamante, C. and Ioannidis, A., XGMix: Local-Ancestry Inference With Stacked XGBoost, International Conference on Learning Representations (ICLR, 2020, Workshop AI4H).
