# General configuration
verbose = True     # logging
instance_name = "" # can be used for organizing simulation output and models

# Simulatiation configuration
run_simulation = True     # if you already simulated data, setting False will re-use the data
founders_ratios = [0.6, 0.10, 0.15, 0.15]
num_outs = [64, 10, 40, 40] # [320, 50, 80, 80]  # how many individuals in each set for each generation
generations = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]  # what generations to generate during simulation
rm_simulated_data = False     # remove the simulated data after training the model

# Model configuration
model_name = "model" # the complete name will be <model_name>_chm<chm>.pkl
window_size = 750    # window size of the XGMix model
smooth_size = 75     # how many windows to aggregate over in second phase
missing = 0.0        # fraction in [0,1) of how much to simulate missing data during training
retrain_base = True  # for retraining base models with [train1, train2] once the smoother has been trained with train2
calibrate = True
n_cores = 30         # how many units of cpu to use