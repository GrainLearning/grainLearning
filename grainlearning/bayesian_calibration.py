from typing import Type, Dict
from .models import Model, IOModel
from .iterative_bayesian_filter import IterativeBayesianFilter
from .tools import plot_param_stats, plot_posterior, plot_param_data, plot_obs_and_sim


class BayesianCalibration:
    """This is the Bayesian calibration class

    A Bayesian calibration class consists of the inference class and the (re)sampling class.
    For instance, in GrainLearning, we have a calibration method called "iterative Bayesian filter"
    which consists of "sequential Monte Carlo" for model parameter estimation
    and "variational Gaussian mixture" for resampling.

    There are two ways of initializing a calibration toolbox class.

    Method 1 - dictionary style (recommended)

    .. highlight:: python
    .. code-block:: python

        model_cls = BayesianCalibration.from_dict(
            {
                "num_iter": 8,
                "model": {
                    "model_type": Model,
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

        model_cls = BayesianCalibration(
            num_iter = 10,
            model = Model(...),
            calibration = IterativeBayesianFilter(...)
            save_fig = -1
        )

    :param model: A `state-space model <https://en.wikipedia.org/wiki/Particle_filter#Approximate_Bayesian_computation_models>`_ (a Model or IOModel object)
    :param calibration: An Iterative Bayesian Filter
    :param num_iter: Number of iteration steps
    :param curr_iter: Current iteration step
    :param save_fig: Flag for skipping (-1), showing (0), or saving (1) the figures
    """
    #: Model being calibrated on
    model: Type["Model"]

    #: Calibration method (e.g, Iterative Bayesian Filter)
    calibration: Type["IterativeBayesianFilter"]

    #: Number of iterations
    num_iter: int

    #: Current calibration step
    curr_iter: int = 0

    #: Flag to save figures
    save_fig: int = -1

    def __init__(
        self,
        model: Type["Model"],
        calibration: Type["IterativeBayesianFilter"],
        num_iter: int,
        curr_iter: int,
        save_fig: int
    ):
        self.model = model

        self.calibration = calibration

        self.num_iter = num_iter

        self.curr_iter = curr_iter

        self.save_fig = save_fig

    def run(self):
        """ This is the main calibration loop which does the following steps
            1. First iteration of Bayesian calibration starts with a Halton sequence
            2. Iterations continue by resampling the parameter space until certain criteria are met.
        """
        print(f"Bayesian calibration iter No. {self.curr_iter}")
        # First iteration
        self.run_one_iteration()

        # Bayesian calibration continue until num_iter is reached or sigma_max is smaller than the tolerance
        for i in range(self.num_iter - 1):
            self.curr_iter += 1
            print(f"Bayesian calibration iter No. {self.curr_iter}")
            self.run_one_iteration()
            if self.model.sigma_max < self.model.sigma_tol:
                self.num_iter = self.curr_iter + 1
                break

    def run_one_iteration(self, index: int = -1):
        """Run Bayesian calibration for one iteration.

        :param index: iteration step, defaults to -1
        """
        # Initialize the samples if it is the first iteration
        if self.curr_iter == 0:
            self.calibration.initialize(self.model)
        # Fetch the parameter values from a stored list
        self.model.param_data = self.calibration.param_data_list[index]
        self.model.num_samples = self.model.param_data.shape[0]

        # Run the model instances
        self.model.run(curr_iter=self.curr_iter)

        # Load model data from disk
        if type(self.model) is IOModel:
            self.load_model()

        # Estimate model parameters as a distribution
        self.calibration.solve(self.model)
        self.calibration.sigma_list.append(self.model.sigma_max)

        # Generate some plots
        self.plot_uq_in_time()

    def load_model(self):
        """Load existing simulation data from disk into the state-space model
        """
        self.model.load_param_data(self.curr_iter)
        self.model.get_sim_data_files(self.curr_iter)
        self.model.load_sim_data()

    def load_and_run_one_iteration(self):
        """Load existing simulation data and run Bayesian calibration for one iteration
           Note the maximum uncertainty sigma_max is solved to reach a certain effective sample size ess_target,
           unlike being assumed as an input for `load_and_process(...)`
        """
        self.load_model()
        self.calibration.add_curr_param_data_to_list(self.model.param_data)
        self.calibration.solve(self.model)
        self.calibration.sigma_list.append(self.model.sigma_max)
        self.plot_uq_in_time()

    def load_and_process(self, sigma: float = 0.1):
        """Load existing simulation data and compute the posterior distribution using an assumed sigma

        :param sigma: assumed uncertainty coefficient, defaults to 0.1
        """
        self.load_model()
        self.calibration.add_curr_param_data_to_list(self.model.param_data)
        self.calibration.load_proposal_from_file(self.model)
        self.calibration.inference.data_assimilation_loop(sigma, self.model)

    def resample(self):
        """Learn and resample from a proposal distribution
        todo this should be refactored

        :return: Combinations of resampled parameter values
        """
        self.calibration.posterior_ibf = self.calibration.inference.give_posterior()
        self.calibration.run_sampling(self.model)
        resampled_param_data = self.calibration.param_data_list[-1]
        self.model.write_to_table(self.curr_iter + 1)
        return resampled_param_data

    def plot_uq_in_time(self):
        """Plot the evolution of uncertainty moments and distribution over time
        """
        if self.save_fig < 0:
            return

        import os
        path = f'{self.model.sim_data_dir}/iter{self.curr_iter}' \
            if type(self.model) == IOModel \
            else f'./{self.model.sim_name}/iter{self.curr_iter}'

        if not os.path.exists(path):
            os.makedirs(path)

        fig_name = f'{path}/{self.model.sim_name}'
        plot_param_stats(
            fig_name, self.model.param_names,
            self.calibration.inference.ips,
            self.calibration.inference.covs,
            self.save_fig
        )

        plot_posterior(
            fig_name,
            self.model.param_names,
            self.model.param_data,
            self.calibration.inference.posteriors,
            self.save_fig
        )

        plot_param_data(
            fig_name,
            self.model.param_names,
            self.calibration.param_data_list,
            self.save_fig
        )

        plot_obs_and_sim(
            fig_name,
            self.model.ctrl_name,
            self.model.obs_names,
            self.model.ctrl_data,
            self.model.obs_data,
            self.model.sim_data,
            self.calibration.inference.posteriors,
            self.save_fig
        )

    def get_most_prob_params(self):
        """Return the most probable set of parameters

        :return: Estimated parameter values
        """
        from numpy import argmax
        most_prob = argmax(self.calibration.posterior_ibf)
        return self.model.param_data[most_prob]

    @classmethod
    def from_dict(
        cls: Type["BayesianCalibration"],
        obj: Dict
    ):
        """An alternative constructor to allow choosing a model type (e.g., Model or IOModel)
        :param obj: a dictionary containing the keys and values to construct a BayesianCalibration object
        :return: A BayesianCalibration object
        """

        # Get the model class, defaults to `Model`
        model_obj = obj["model"]
        model_type = model_obj.get("model_type", Model)
        # if the dictionary has the key "model_type", then delete it to avoid passing it to the constructor
        model_obj.pop("model_type", None)

        # Create a model object
        model = model_type.from_dict(obj["model"])

        # Create a calibration object
        calibration = IterativeBayesianFilter.from_dict(obj["calibration"])

        return cls(
            model=model,
            calibration=calibration,
            num_iter=obj["num_iter"],
            curr_iter=obj.get("curr_iter", 0),
            save_fig=obj.get("save_fig", -1)
        )
