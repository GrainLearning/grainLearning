import os
import numpy as np
import glob
import re

# Directory containing the .npy files
data_dir = os.path.dirname(__file__)

# Pattern to match files
pattern = os.path.join(data_dir, "column_collapse_triax_Iter0_*_*_fields.npy")

# Find all matching files
file_list = glob.glob(pattern)

# Get the number of samples
num_samples = len(set(re.search(r'column_collapse_triax_Iter0_(\w+)_\d+_fields\.npy', os.path.basename(f)).group(1) for f in file_list))
print(f"Found {len(file_list)} files corresponding to {num_samples} samples.")

# Combine data per sample into a big dictionary, using time steps as keys
def combine_data_per_sample(sampleID, step_interval=1000):
    combined_data = {}
    # define pattern per sample
    pattern = os.path.join(data_dir, f"column_collapse_triax_Iter0_Sample{sampleID:02d}_*_fields.npy")
    # check if sampleID matches the pattern
    file_list = glob.glob(pattern)
    # sort file_list by timestep
    file_list = sorted(file_list, key=lambda x: int(re.search(r'_(\d+)_fields\.npy', os.path.basename(x)).group(1)))
    # check if time steps are continous and indeed increasing
    time_steps = [int(re.search(r'_(\d+)_fields\.npy', os.path.basename(f)).group(1)) for f in file_list]
    # increment between time steps should be step_interval
    assert all(t2 - t1 == step_interval for t1, t2 in zip(time_steps, time_steps[1:])), f"Time steps for sample {sampleID} are not increasing by {step_interval}: {time_steps}"
    assert time_steps == list(range(time_steps[0], time_steps[0] + step_interval * len(time_steps), step_interval)), f"Time steps for sample {sampleID} are not continuous: {time_steps}"
    # load and combine data
    for i, fname in enumerate(file_list):
        data = np.load(fname, allow_pickle=True)
        combined_data[i] = data
    # save combined data

    np.save(os.path.join(data_dir, f"column_collapse_{sampleID:02d}_CG_fields.npy"), combined_data)
    return combined_data

for sampleID in range(num_samples):
    combined_data = combine_data_per_sample(sampleID)
    print(f"Combined data for sample {sampleID} saved.")