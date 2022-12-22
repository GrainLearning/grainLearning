# RNN module of grainLearning

This module uses recurrent neural networks (RNN) to predict a sequence of macroscopic observables (i.e. stress) of a granular material undergoing a given input sequence of strains.
Trained on DEM simulations with [YADE](http://yade-dem.org/), using different contact parameters.

## Installation

During the installation of grainLearning activate extra `rnn`:

`poetry install --extras "rnn"` or `pip install .[rnn]`

For MacOS with arm64 processor, we recommend to install  tensorflow following [this](https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/) and install grainLearning without `rnn` extra.

## How to use

There are three main usages of RNN module:

I. Train a RNN with your own data.

II. Make a prediction with a pre-trained model.

III. Use a trained RNN as "software" in the workflow of grainLearning calibration process.

IV. Train a RNN model during the grainLearning calibration process.

### I. Train a RNN with your own data

1. Your **data** should be a .hdf5 file with the structure given in `parse_data.py`. You can download our prepared dataset here to `grainLearning/rnn/data/`. Modify the entry `raw_data` in the config dictionary in `train.py` or if you're running a sweep in `sweep/example_sweep.yaml`.
2. Check and/or modify the hyperparameters values in the default config dictionary. See docs for more details.
3. In `train.py` you can decide to call:

    - `train` : saves model information to `wandb/latest-run`.
    - `train_without_wandb` : saves model information to `outputs/`.

4. Once you are happy with your trained model you can copy `config.yaml`, `train_stats.npy`, and `model-best.h5` to `trained_models` and use it in the future.

### II. Make a prediction with a pre-trained model

In `predict.py` you can make predictions on test data, loading:

- Best model found in a sweep, or
- A previously trained model (see examples in trained_models).

### III. Use RNN during grainLearning calibration process

In the framework of grainLearning, RNN is a Model and can be easily linked using `Model.callback` [implemented in python](https://grainlearning.readthedocs.io/en/latest/models.html#id1).

See examples in documentation #TODO

### IV. Train a RNN model during the grainLearning calibration process

#TODO

## Weights and Biases

[Weights a Biases](https://wandb.ai/site) is a platform used for tracking experiments and hyperparameter tuning.

Create a free account on [wandb](https://wandb.ai/site) (easiest is to couple your github account).

### Sweeps for hyperparameter tuning

This can then be used to run parameter sweeps. A sweep's configuration is specified in a yaml file, like `sweep/example_sweep.yaml`.
These list the chosen hyperparameter possibilities. Make sure to change the `entity` to your account name.

A sweep is then created on the command line using 

```bash
wandb sweep example_sweep.yaml
```

This won't do any training yet, it will create a sweep ID, show a link where the results can be tracked, and output the command needed to run it, which is of the form

```bash
wandb agent <entity>/<project>/<sweep_id>
```

Running this command will start a training run with hyperparameters chosen according to the config file, and will keep starting new runs.

Models are saved both locally and also uploaded to wandb.

## Usage on HPC

*Note:* This instructions assume that your HPC platform uses job scheduler slurm. `run_sweep.sh` configures the job and loads modules from **Snellius**, these can be different in other supercomputers.

1. First clone the grainLearning repo to your supercomputer directory and install it.

2. Copy the data into `grainLearning/rnn/data/my_data.hdf5`, and create a directory `job_output`.

3. Configure wandb for your own account.

    ```bash
    wand init
    ```

4. In `sweep/example_sweep.yaml` change the `entity` to your account name as well as the parameters ranges.

5. To run the example sweep, run the `run_sweep.sh` job script:

    ```bash
    sbatch run_sweep.sh example_sweep.yaml
    ```

This will store the slurm logs and the information of the weights and biases sweep (which includes a link to the sweep page) in the `job_output` directory,
and creates a `wandb` folder containing all the wandb output.

It will use a quarter of a node in the fat partition, which has 32 cores, and so it will run 32 agents in parallel in the same sweep.
