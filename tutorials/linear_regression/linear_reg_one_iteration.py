from grainlearning import CalibrationToolbox
from grainlearning.models import IOModel
from pathlib import Path
import os, sys

sys.path.append(os.path.join(os.getcwd(), "grainlearning"))

sim_data_dir = Path("./tests/data/linear_sim_data")
curr_iter = 0

calibration = CalibrationToolbox.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 0,
        "model": {
            "param_mins": [0.001, 0.001],
            "param_maxs": [1, 10],
            "obs_data_file": 'linearObs.dat',
            "obs_names": ['f'],
            "ctrl_name": 'u',
            "sim_name": 'linear',
            "param_data_file": f'{sim_data_dir}/iter{curr_iter}/smcTable0.txt',
            "sim_data_dir": sim_data_dir,
            "param_names": ['a', 'b'],
        },
        "calibration": {
            "inference": {"ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "cov_type": "full",
            },
        },
        "save_fig": 0,
        "model_type": IOModel,
    }
)

# %%
# reproduce the result with a given sigma value
calibration.load_and_run_one_iteration()
resampled_param_data = calibration.resample()

# %%
# write new parameter table to the simulation directory
calibration.model.write_to_table(calibration.curr_iter + 1)
