"""
This tutorial shows how to use GrainLearning to load existing simulation data,
 estimate parameter distribution, and perform resampling to generate new parameter samples.
"""
import os
from grainlearning import BayesianCalibration
from grainlearning.dynamic_systems import IODynamicSystem
import seaborn as sns
import pandas as pd

sim_data_dir = os.path.abspath(os.path.join(__file__, "../../../../tests/data/oedo_sim_data"))
curr_iter = 1

calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": curr_iter,
        "num_iter": 0,
        "system": {
            "system_type": IODynamicSystem,
            "obs_data_file": sim_data_dir + '/obsdata.dat',
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
                "weight_concentration_prior": 0.01,
            },
            "proposal_data_file": f"gmm_iter{curr_iter - 1}.pkl",
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

# %%
# prepare the data in pandas.DataFrame format. The most important input is the trained Gaussian Mixture Model (gmm)
gmm = calibration.calibration.sampling.gmm
num = calibration.system.num_samples
samples, _ = gmm.sample(num)
samples *= calibration.calibration.sampling.max_params
df = pd.DataFrame(samples, columns=calibration.system.param_names)

# plot the parameter distribution in a scatter plot
fig = sns.PairGrid(df, diag_sharey=False)
fig.map_upper(sns.scatterplot, s=15)
fig.map_lower(sns.kdeplot)
fig.map_diag(sns.kdeplot, lw=2)
fig.savefig(f'{calibration.system.sim_name}_scatterplot.png')
