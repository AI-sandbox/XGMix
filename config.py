# General configuration
verbose = True     # logging
instance_name = "" # can be used for organizing simulation output and models

# Simulatiation configuration
run_simulation = True     # if you already simulated data, setting False will re-use it
num_outs = [700, 200, 0]  # how many individuals in each sub-instance
generations = [2, 4, 6]   # what generations to generate during simulation
rm_simulated_data = False # remove the simulated data after training the model

# Model configuration
model_name = "model" # the complete name will be <model_name>_chm<chm>.pkl
window_size = 1000   # window size of the XGMix model
smooth_size = 75     # how many windows to aggregate over in second phase
missing = 0.0        # fraction in [0,1) of how much to simulate missing data during training
n_cores = 4          # how many units of cpu to use
smooth_lite = 10000  # how many (random) snps the smoother uses