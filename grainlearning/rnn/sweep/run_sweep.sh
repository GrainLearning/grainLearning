#!/bin/bash
#Set job requirements
#SBATCH -n 32
#SBATCH --partition=fat
#SBATCH --time=00:30:00
#SBATCH -o ./job_output/slurm-%j.out

module load 2021 Python/3.9.5-GCCcore-10.3.0 TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1 
module load h5py/3.2.1-foss-2021a matplotlib/3.4.2-foss-2021a plotly.py/5.1.0-GCCcore-10.3.0 

# Direct the output of this job to a file
sweep_file=./job_output/sweep_id-${SLURM_JOB_ID}.txt

# create a sweep using the configuration file that was passed in 2nd possition to the call 
#of this script (.yaml) 
wandb sweep $1 &> $sweep_file

# The previous command will print the sweep id assigned, this is going to be saved to sweep_file.
# We extract the sweep id:
sweep_id=`cat $sweep_file | grep agent | cut -d' ' -f8`

# run an agent of the sweep in this machine
srun wandb agent $sweep_id

