"""
This module contains the Bayesian calibration class.
"""
from typing import Type, Dict, Callable
import os
from glob import glob
from numpy import argmax
from grainlearning.dynamic_systems import DynamicSystem, IODynamicSystem
from grainlearning.iterative_bayesian_filter import IterativeBayesianFilter
from grainlearning.tools import plot_param_stats, plot_posterior, plot_param_data, plot_obs_and_sim, plot_pdf, \
    close_plots


class BayesianCalibration:
    """This is the Bayesian calibration class.

    A Bayesian calibration requires the following input:

    1. The :class:`.DynamicSystem` that encapsulates the observation data and simulation data,

    2. The inference method, for example, :class:`.IterativeBayesianFilter`,

    3. The number of iterations,

    4. The current iteration number if the user simply wants to process their data with GrainLearning for one iteration,

    5. the flag for skipping (-1), showing (0), or saving (1) the figures,
    
    6. and the tolerances for stopping the calibration based on the mean absolute percentage error and the ensemble percentage error.

    There are two ways of initializing a calibration toolbox class.

    Method 1 - dictionary style (recommended)

    .. highlight:: python
    .. code-block:: python

        bayesian_calibration = BayesianCalibration.from_dict(
            {
                "num_iter": 8,
                "callback": run_sim,
                "system": {
                    "system_type": DynamicSystem,
                    "model_name": "test",
                    "param_names": ["a", "b"],
                    "param_min": [0, 0],
                    "param_max": [1, 10],
                    "num_samples": 10,
                    "obs_data": [2,4,8,16],
                    "ctrl_data": [1,2,3,4],
                },
                "inference": {
                    "Bayes_filter": {"ess_target": 0.3},
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
            num_iter = 8,
            system = DynamicSystem(...),
            inference = IterativeBayesianFilter(...)
            save_fig = -1
        )

    :param system: A `dynamic system <https://en.wikipedia.org/wiki/Particle_filter#Approximate_Bayesian_computation_models>`_ whose observables and hidden states evolve dynamically over "time"
    :param inference: An inference method to determine unknown parameters in state-parameter space. Currently, only the iterative Bayesian Filter is available.
    :param num_iter: Number of iteration steps
    :param curr_iter: Current iteration step
    :param error_tol: Tolerance to check the sample with the smallest mean absolute percentage error
    :param gl_error_tol: Tolerance to check the GrainLearning ensemble percentage error
    :param save_fig: Flag for skipping (-1), showing (0), or saving (1) the figures
    :param callback: A callback function that runs the external software and passes the parameter sample to generate outputs
    """
    def __init__(
        self,
        system: Type["DynamicSystem"],
        inference: Type["IterativeBayesianFilter"],
        num_iter: int = 1,
        curr_iter: int = 0,
        error_tol: float = None,
        gl_error_tol: float = None,
        save_fig: int = -1,
        threads: int = 4,
        callback: Callable = None,
    ):
        """Initialize the Bayesian calibration class"""

        self.num_iter = num_iter

        self.save_fig = save_fig

        self.threads = threads

        self.system = system

        self.system.curr_iter = curr_iter

        self.curr_iter = curr_iter

        self.error_tol = error_tol

        self.gl_error_tol = gl_error_tol

        self.inference = inference

        self.callback = callback

        self.error_array = None

        self.gl_errors = []

    def run(self):
        """ This is the main calibration loop which does the following steps
            1. First iteration of Bayesian calibration starts with a Halton sequence
            2. Iterations continue by resampling the parameter space until certain criteria are met.
        """
        # Move existing simulation data to the backup folder
        self.system.backup_sim_data()

        # Bayesian calibration continue until curr_iter = num_iter or sigma_max < tolerance
        for _ in range(self.num_iter):
            print(f"Bayesian calibration iter No. {self.curr_iter}")
            stopping_criteria_met = self.run_one_iteration()
            if stopping_criteria_met:
                break

        # Print the errors after all iterations are done
        if not stopping_criteria_met:
            error_most_probable = min(self.error_array)
            gl_error = self.gl_errors[-1]
            print(f"\n"
                    f"Stopping criteria NOT met: \n"
                    f"sigma = {self.system.sigma_max},\n"
                    f"Smallest mean absolute percentage error = {error_most_probable: .3e},\n"
                    f"GrainLearning ensemble percentage error = {gl_error: .3e}\n\n"
                    f"Ending Bayesian calibration.")

    def run_one_iteration(self, index: int = -1):
        """Run Bayesian calibration for one iteration.

        :param index: iteration step, defaults to -1
        """
        # Initialize the samples if it is the first iteration
        if self.curr_iter == 0:
            self.inference.initialize(self.system)
        # Fetch the parameter values from a stored list
        self.system.param_data = self.inference.param_data_list[index]
        self.system.num_samples = self.system.param_data.shape[0]
        self.system.sim_data = None

        # Run the model realizations
        self.run_callback()

        # Load model data from disk if sim_data is not already set in the callback function
        if self.system.sim_data is None:
            self.load_system()

        # Estimate model parameters as a distribution
        self.inference.solve(self.system)
        self.inference.sigma_list.append(self.system.sigma_max)

        # Generate some plots
        self.plot_uq_in_time()

        self.compute_errors()

        # Defining stopping criterion
        error_most_probable = min(self.error_array)
        gl_error = self.gl_errors[-1]
        print(f"\n"
                f"sigma = {self.system.sigma_max},\n"
                f"Smallest mean absolute percentage error = {error_most_probable: .3e},\n"
                f"GrainLearning ensemble percentage error = {gl_error: .3e}\n")

        # If any stopping condition is met
        # Check stopping criteria
        normalized_sigma_met = self.system.sigma_max < self.system.sigma_tol
        most_probable_error_met = error_most_probable < self.error_tol if self.error_tol is not None else False
        ensemble_error_met = gl_error < self.gl_error_tol if self.gl_error_tol is not None else False

        if normalized_sigma_met or most_probable_error_met or ensemble_error_met:
            print(f"\nStopping criteria are met. Ending Bayesian calibration!\n")
            self.num_iter = self.curr_iter + 1
            return True
        else:
            self.increase_curr_iter()
            return False

    def run_callback(self):
        """
        Run the callback function
        """

        if self.callback is not None:
            if isinstance(self.system, IODynamicSystem):
                self.system.set_up_sim_dir(self.threads)
                self.callback(self)
                self.system.move_data_to_sim_dir()
            else:
                self.callback(self)
        else:
            raise ValueError("No callback function defined")

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
        self.inference.add_curr_param_data_to_list(self.system.param_data)
        self.inference.solve(self.system)
        self.system.write_params_to_table(self.threads)
        self.inference.sigma_list.append(self.system.sigma_max)
        self.plot_uq_in_time()

    def load_and_process(self, sigma: float = 0.1):
        """Load existing simulation data and compute the posterior distribution using an assumed sigma

        :param sigma: assumed uncertainty coefficient, defaults to 0.1
        """
        self.load_system()
        self.inference.add_curr_param_data_to_list(self.system.param_data)
        self.inference.load_proposal_from_file(self.system)
        self.inference.Bayes_filter.data_assimilation_loop(sigma, self.system)
        self.system.compute_estimated_params(self.inference.Bayes_filter.posteriors)

    def load_all(self):
        """Simply load all previous iterations of Bayesian calibration
        """
        self.load_system()
        self.inference.add_curr_param_data_to_list(self.system.param_data)
        self.increase_curr_iter()
        while self.curr_iter < self.num_iter:
            print(f"Bayesian calibration iter No. {self.curr_iter}")
            self.load_system()
            self.inference.add_curr_param_data_to_list(self.system.param_data)
            self.inference.run_inference(self.system)
            self.inference.sigma_list.append(self.system.sigma_max)
            self.plot_uq_in_time()
            self.increase_curr_iter()

    def resample(self):
        """Learn and resample from a proposal distribution
        todo this should be refactored

        :return: Combinations of resampled parameter values
        """
        self.inference.posterior = self.inference.Bayes_filter.get_posterior_at_time()
        self.inference.run_sampling(self.system, )
        resampled_param_data = self.inference.param_data_list[-1]
        self.system.write_params_to_table(self.threads)
        return resampled_param_data

    def plot_uq_in_time(self, verbose: bool = False):
        """Plot the evolution of uncertainty moments and distribution over time
        
        param verbose: plot also the detailed statistics, defaults to False
        """
        if self.save_fig < 0:
            return

        path = f'{self.system.sim_data_dir}/iter{self.curr_iter}' \
            if isinstance(self.system, IODynamicSystem) \
            else f'./{self.system.sim_name}/iter{self.curr_iter}'

        if not os.path.exists(path):
            os.makedirs(path)

        fig_name = f'{path}/{self.system.sim_name}'
        
        if verbose:
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
                self.inference.Bayes_filter.posteriors,
                self.save_fig
            )

            plot_param_data(
                fig_name,
                self.system.param_names,
                self.inference.param_data_list,
                self.save_fig
            )

        plot_obs_and_sim(
            fig_name,
            self.system.ctrl_name,
            self.system.obs_names,
            self.system.ctrl_data,
            self.system.obs_data,
            self.system.sim_data,
            self.inference.Bayes_filter.posteriors,
            self.save_fig
        )

        plot_pdf(
            fig_name,
            self.system.param_names,
            self.inference.param_data_list,
            self.save_fig,
        )

        close_plots(self.save_fig)

    def get_most_prob_params_id(self):
        """Return the most probable set of parameters

        :return: Estimated parameter values
        """
        return argmax(self.inference.posterior)

    def get_most_prob_params(self):
        """Return the most probable set of parameters

        :return: Estimated parameter values
        """
        most_prob = argmax(self.inference.posterior)
        return self.system.param_data[most_prob]

    def increase_curr_iter(self):
        """Increase the current iteration step by one
        """
        self.system.curr_iter += 1
        self.curr_iter += 1


    def compute_errors(self):
        """Compute the mean absolute percentage error per sample and the ensemble error
        """
        from sklearn.metrics import mean_absolute_error
        import numpy as np

        # compute mean absolute percentage error
        sim_data = self.system.sim_data
        num_samples = self.system.num_samples

        # compute GrainLearning errors
        self.error_array = np.zeros(num_samples)
        # loop over all samples to compute sample percentage errors
        # if observation data is a time series
        if self.system.num_steps == 1:
            obs_maxs = np.max(self.system.obs_data)
            obs_mins = np.min(self.system.obs_data)
        else:
            obs_maxs = np.max(self.system.obs_data, axis=1)
            obs_mins = np.min(self.system.obs_data, axis=1)
        obs_range = obs_maxs - obs_mins
        for i in range(num_samples):
            self.error_array[i] = mean_absolute_error(self.system.obs_data.T/obs_range, sim_data[i, :, :].T/obs_range)

        # compute the ensemble error
        self.gl_errors.append(np.dot(self.error_array, self.inference.posterior))

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

        # Create a inference object
        inference = IterativeBayesianFilter.from_dict(obj["inference"])

        return cls(
            system=system,
            inference=inference,
            num_iter=obj["num_iter"],
            curr_iter=obj.get("curr_iter", 0),
            error_tol=obj.get("error_tol", None),
            gl_error_tol=obj.get("gl_error_tol", None),
            save_fig=obj.get("save_fig", -1),
            threads=obj.get("threads", 4),
            callback=obj.get("callback", None)
        )
