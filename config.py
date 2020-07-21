# General arguments
instance_name = "" # can be used for organizing simulation output and models
verbose = True     # logging

# Simulatiation arguments
num_outs = [700, 200, 0]  # how many individuals in each sub-instance
generations = [2, 4, 6]   # what generations to generate during simulation
rm_simulated_data = False # remove the simulated data after training the model

# Model arguments
model_name = "model" # the complete name will be <model_name>_chm<chm>
window_size = 5000   # window size of the XGMix model
smooth_size = 75     # how many windows to aggregate over in second phase
missing = 0.0        # fraction in [0,1) of how much to simulate missing data during training
n_cores = 16         # how many cores to use
smooth_lite = 10000  # how many (random) snps the smoother uses