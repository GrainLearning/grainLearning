#%%

import numpy as np

from grainlearning import (
    SequentialMonteCarlo,
    IterativeBayesianFilter,
    GaussianMixtureModel,
    Model,
    Observations,
    Parameters,
    CalibrationToolbox
)

import matplotlib.pyplot as plt



x_obs = np.arange(10)

y_obs = 4 * x_obs + 5

plt.plot(x_obs, y_obs)

#%%
class MyModel(Model):
    parameters = Parameters(
        names=["m", "c"],
        mins=[0, 0],
        maxs=[10, 10],
    )
    observations = Observations(data=y_obs, ctrl=x_obs, names=["y"], ctrl_name="x")
    
    num_samples = 20
    
    iter_step = 0

    def __init__(self):
        self.parameters.generate_halton(self.num_samples)

    def run(self):
        # for each parameter calculate the spring force
        data = []
        print(f"calibration step {self.iter_step}")

        for params in self.parameters.data:
            y_sim = params[0] * self.observations.ctrl + params[1]
            data.append(np.array(y_sim, ndmin=2))
            plt.plot(x_obs, y_sim,label="simulation")

        self.data = np.array(data)
        
        self.iter_step +=1
        self.plot_data()
        # self.plot_params()
        
    def plot_data(self):
        plt.figure()
        plt.plot(x_obs,y_obs,"--",label="Observation")
        plt.title(f"Calibration iteration {self.iter_step}")
        for y_sims in self.data[:,0]:
            plt.plot(x_obs, y_sims,label="Simulation")

mymodel = MyModel()

smc_cls = SequentialMonteCarlo(
    ess_target=0.1, inv_obs_weight=[1], scale_cov_with_max=True
)
gmm_cls = GaussianMixtureModel(max_num_components=1) 

ibf_cls = IterativeBayesianFilter(inference=smc_cls, sampling=gmm_cls)

calibration = CalibrationToolbox(mymodel,ibf_cls)

# %%

calibration.run()

# %%
calibration.model.parameters.data
# %%
