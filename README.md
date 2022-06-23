# sequences
Using recurrent neural networks to predict a sequence of macroscopic observables of a granular material undergoing a given input sequence of strains.
Trained on DEM simulations with YADE, using different contact parameters.

# How to use

## Installation

Clone this repo and go to the created directory
```
git clone git@github.com:GrainLearning/sequences.git
cd sequences
```

Create and activate a new environment with conda:
```
conda create --name sequences
conda activate sequences
```

Install with pip:
```
pip install -e .
```
For the M1 Mac, comment out tensorflow in the `setup.cfg` and install that separately following [this](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/).

To test if it installed properly, run (Actually this probably already requires a wandb account)
```
python train.py
```

## weights and biases

Create a free account on [wandb](https://wandb.ai/site) (easiest is to couple your github account). 

This can then be used to run parameter sweeps. A sweep's configuration is specified in a yaml file, like `example_sweep.yaml`.
These list the chosen hyperparameter possibilities. Make sure to change the `entity` to your account name.

A sweep is then created on the command line using 
```
wandb sweep example_sweep.yaml
```
This won't do any training yet, it will create a sweep ID, show a link where the results can be tracked, 
and output the command needed to run it, which is of the form
```
wandb agent <entity>/<project>/<sweep_id>
```
Running this command will start a training run with hyperparameters chosen according to the config file, and will keep starting new runs.

Models are saved both locally and also uploaded to wandb.

## Usage on Snellius

First clone the repo on your snellius directory
```
git clone https://github.com/GrainLearning/sequences.git
```
Manually copy the data into `sequences/data/sequences.hdf5`, and create a directory `job_output`.

To run the example sweep, run the `run_sweep.sh` job script:
```
sbatch run_sweep.sh example_sweep.yaml
```
This will store the slurm logs and the information of the weights and biases sweep,
and create a `wandb` folder containing all the wandb output.

## Predicting

In `predict.py` a sweep id is used to load the best model found in that sweep, and make predictions on test data.
