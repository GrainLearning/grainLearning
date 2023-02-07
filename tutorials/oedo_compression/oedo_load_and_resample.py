from grainlearning import BayesianCalibration
from grainlearning.models import IOModel

curr_iter = 1
sim_data_dir = "./tests/data/oedo_sim_data"

calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 0,
        "model": {
            "model_type": IOModel,
            "obs_data_file": 'obsdata.dat',
            "obs_names": ['p', 'q', 'n'],
            "ctrl_name": 'e_a',
            "sim_name": 'oedo',
            "sim_data_dir": sim_data_dir,
            "param_data_file": f'{sim_data_dir}/iter{curr_iter}/smcTable{curr_iter}.txt',
            "param_names": ['E', 'mu', 'k_r', 'mu_r'],
            "param_min": [100e9, 0.3, 0, 0.1],
            "param_max": [200e9, 0.5, 1e4, 0.5],
            "inv_obs_weight": [1, 1, 0.01],
        },
        "calibration": {
            "inference": {"ess_target": 0.2},
            "sampling": {
                "max_num_components": 10,
                "prior_weight": 0.01,
            },
            "proposal_data_file": f"iter{curr_iter - 1}/gmm_iter{curr_iter - 1}.pkl",
        },
        "save_fig": 0,
    }
)

# %%
# load the simulation data and run the inference for one iteration
# ~ calibration.load_and_run_one_iteration()
calibration.load_and_process(0.01)
_ = calibration.resample()

# %%
# plot the uncertainty evolution
calibration.plot_uq_in_time()
