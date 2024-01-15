"""
This module contains the Bayesian calibration class.
"""
from typing import Type, Dict
import os
from numpy import argmax
from grainlearning.dynamic_systems import DynamicSystem, IODynamicSystem
from grainlearning.iterative_bayesian_filter import IterativeBayesianFilter
from grainlearning.tools import plot_param_stats, plot_posterior, plot_param_data, plot_obs_and_sim, plot_pdf, \
    close_plots


class BayesianCalibration:
    """This is the Bayesian calibration class.

    A Bayesian calibration class consists of the inference class and the (re)sampling class.
    For instance, in GrainLearning, we have a calibration method called "iterative Bayesian filter"
    which consists of "sequential Monte Carlo" for model parameter estimation
    and "variational Gaussian mixture" for resampling.

    There are two ways of initializing a calibration toolbox class.

    Method 1 - dictionary style (recommended)

    .. highlight:: python
    .. code-block:: python

        bayesian_calibration = BayesianCalibration.from_dict(
            {
                "num_iter": 8,
                "system": {
                    "system_type": DynamicSystem,
                    "model_name": "test",
                    "param_names": ["a", "b"],
                    "param_min": [0, 0],
                    "param_max": [1, 10],
                    "num_samples": 10,
                    "obs_data": [2,4,8,16],
                    "ctrl_data": [1,2,3,4],
                    "callback": run_sim,
                },
                "calibration": {
                    "inference": {"ess_target": 0.3},
                    "sampling": {"max_num_components": 1},
                },
                "save_fig": -1,
            }
        )

    or

    Method 2 - class style

    .. highlight:: python
    .. code-block:: python

        bayesian_calibration = BayesianCalibration(
            num_iter = 10,
            system = DynamicSystem(...),
            calibration = IterativeBayesianFilter(...)
            save_fig = -1
        )

    :param system: A `dynamic system <https://en.wikipedia.org/wiki/Particle_filter#Approximate_Bayesian_computation_models>`_ whose observables and hidden states evolve dynamically over "time"
    :param calibration: An iterative Bayesian Filter that iteratively sample the parameter space
    :param num_iter: Number of iteration steps
    :param curr_iter: Current iteration step
    :param save_fig: Flag for skipping (-1), showing (0), or saving (1) the figures
    """
    def __init__(
        self,
        system: Type["DynamicSystem"],
        calibration: Type["IterativeBayesianFilter"],
        num_iter: int = 10,
        curr_iter: int = 0,
        save_fig: int = -1
    ):
        """Initialize the Bayesian calibration class"""


        self.num_iter = num_iter


        self.save_fig = save_fig

        self.system = system

        self.system.curr_iter = curr_iter
        
        self.curr_iter = curr_iter
        
        self.calibration = calibration


    def run(self):
        """ This is the main calibration loop which does the following steps
            1. First iteration of Bayesian calibration starts with a Halton sequence
            2. Iterations continue by resampling the parameter space until certain criteria are met.
        """
        # Move existing simulation data to the backup folder
        self.system.backup_sim_data()

        print(f"Bayesian calibration iter No. {self.curr_iter}")
        # First iteration
        self.run_one_iteration()

        # Bayesian calibration continue until curr_iter = num_iter or sigma_max < tolerance
        for _ in range(self.num_iter - 1):
            self.increase_curr_iter()
            print(f"Bayesian calibration iter No. {self.curr_iter}")
            self.run_one_iteration()
            if self.system.sigma_max < self.system.sigma_tol:
                self.num_iter = self.curr_iter + 1
                break

    def run_one_iteration(self, index: int = -1):
        """Run Bayesian calibration for one iteration.

        :param index: iteration step, defaults to -1
        """
        # Initialize the samples if it is the first iteration
        if self.curr_iter == 0:
            self.calibration.initialize(self.system)
        # Fetch the parameter values from a stored list
        self.system.param_data = self.calibration.param_data_list[index]
        self.system.num_samples = self.system.param_data.shape[0]

        # Run the model realizations
        self.system.run()

        # Load model data from disk
        self.load_system()

        # Estimate model parameters as a distribution
        self.calibration.solve(self.system, )
        self.calibration.sigma_list.append(self.system.sigma_max)

        # Generate some plots
        self.plot_uq_in_time()

    def load_system(self):
        """Load existing simulation data from disk into the dynamic system
        """
        if isinstance(self.system, IODynamicSystem):
            self.system.load_param_data()
            self.system.get_sim_data_files()
            self.system.load_sim_data()
        else:
            if self.system.param_data is None or self.system.sim_data is None:
                raise RuntimeError("The parameter and simulation data are not set up correctly.")

    def load_and_run_one_iteration(self):
        """Load existing simulation data and run Bayesian calibration for one iteration
           Note the maximum uncertainty sigma_max is solved to reach a certain effective sample size ess_target,
           unlike being assumed as an input for `load_and_process(...)`
        """
        self.load_system()
        self.calibration.add_curr_param_data_to_list(self.system.param_data)
        self.calibration.solve(self.system)
        self.system.write_params_to_table()
        self.calibration.sigma_list.append(self.system.sigma_max)
        self.plot_uq_in_time()

    def load_and_process(self, sigma: float = 0.1):
        """Load existing simulation data and compute the posterior distribution using an assumed sigma

        :param sigma: assumed uncertainty coefficient, defaults to 0.1
        """
        self.load_system()
        self.calibration.add_curr_param_data_to_list(self.system.param_data)
        self.calibration.load_proposal_from_file(self.system)
        self.calibration.inference.data_assimilation_loop(sigma, self.system)
        self.system.compute_estimated_params(self.calibration.inference.posteriors)

    def load_all(self):
        """Simply load all previous iterations of Bayesian calibration
        """
        self.load_system()
        self.calibration.add_curr_param_data_to_list(self.system.param_data)
        self.increase_curr_iter()
        while self.curr_iter < self.num_iter:
            print(f"Bayesian calibration iter No. {self.curr_iter}")
            self.load_system()
            self.calibration.add_curr_param_data_to_list(self.system.param_data)
            self.calibration.run_inference(self.system)
            self.calibration.sigma_list.append(self.system.sigma_max)
            self.plot_uq_in_time()
            self.increase_curr_iter()

    def resample(self):
        """Learn and resample from a proposal distribution
        todo this should be refactored

        :return: Combinations of resampled parameter values
        """
        self.calibration.posterior = self.calibration.inference.get_posterior_at_time()
        self.calibration.run_sampling(self.system, )
        resampled_param_data = self.calibration.param_data_list[-1]
        self.system.write_params_to_table()
        return resampled_param_data

    def plot_uq_in_time(self):
        """Plot the evolution of uncertainty moments and distribution over time
        """
        if self.save_fig < 0:
            return

        path = f'{self.system.sim_data_dir}/iter{self.curr_iter}' \
            if isinstance(self.system, IODynamicSystem) \
            else f'./{self.system.sim_name}/iter{self.curr_iter}'

        if not os.path.exists(path):
            os.makedirs(path)

        fig_name = f'{path}/{self.system.sim_name}'
        plot_param_stats(
            fig_name, self.system.param_names,
            self.system.estimated_params,
            self.system.estimated_params_cv,
            self.save_fig
        )

        plot_posterior(
            fig_name,
            self.system.param_names,
            self.system.param_data,
            self.calibration.inference.posteriors,
            self.save_fig
        )

        plot_param_data(
            fig_name,
            self.system.param_names,
            self.calibration.param_data_list,
            self.save_fig
        )

        plot_obs_and_sim(
            fig_name,
            self.system.ctrl_name,
            self.system.obs_names,
            self.system.ctrl_data,
            self.system.obs_data,
            self.system.sim_data,
            self.calibration.inference.posteriors,
            self.save_fig
        )

        plot_pdf(
            fig_name,
            self.system.param_names,
            self.calibration.param_data_list,
            self.save_fig,
        )

        close_plots(self.save_fig)

    def get_most_prob_params(self):
        """Return the most probable set of parameters

        :return: Estimated parameter values
        """
        most_prob = argmax(self.calibration.posterior)
        return self.system.param_data[most_prob]

    def increase_curr_iter(self):
        """Increase the current iteration step by one
        """
        self.system.curr_iter += 1
        self.curr_iter += 1

    @classmethod
    def from_dict(
        cls: Type["BayesianCalibration"],
        obj: Dict
    ):
        """An alternative constructor to allow choosing a system type (e.g., dynamic system or IO dynamic system)

        :param obj: a dictionary containing the keys and values to construct a BayesianCalibration object
        :return: a BayesianCalibration object
        """

        # Get the system class, defaults to `DynamicSystem`
        system_obj = obj["system"]
        system_type = system_obj.get("system_type", DynamicSystem)
        # if the dictionary has the key "system_type", then delete it to avoid passing it to the constructor
        system_obj.pop("system_type", None)

        # Create a system object
        system = system_type.from_dict(obj["system"])

        # Create a calibration object
        calibration = IterativeBayesianFilter.from_dict(obj["calibration"])

        return cls(
            system=system,
            calibration=calibration,
            num_iter=obj["num_iter"],
            curr_iter=obj.get("curr_iter", 0),
            save_fig=obj.get("save_fig", -1)
        )
