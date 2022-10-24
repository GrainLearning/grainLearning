#!/bin/bash
#Set job requirements
#SBATCH -n 32
#SBATCH --partition=fat
#SBATCH --time=00:30:00
#SBATCH -o ./job_output/slurm-%j.out

module load 2021 Python/3.9.5-GCCcore-10.3.0 TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1 h5py/3.2.1-foss-2021a matplotlib/3.4.2-foss-2021a plotly.py/5.1.0-GCCcore-10.3.0

# save wandb error message to file...
sweep_file=./job_output/sweep_id-${SLURM_JOB_ID}.txt
wandb sweep $1 &> $sweep_file
# ... and extract the sweep id
sweep_id=`cat $sweep_file | grep agent | cut -d' ' -f8`

srun wandb agent $sweep_id

