
from ast import Param
from typing import Type, List, Dict
from .models import *
from .iterativebayesianfilter import IterativeBayesianFilter


class CalibrationToolbox:
    """This is the main calibration toolbox
  
    A Bayesian calibration method consists of the inference class and the (re)sampling class.
    For instance, in GrainLearning, we have a calibration method called "iterative Bayesian filter"
    which consists of "sequential Monte Carlo" for the inference and "variational Gaussian mixture" for the resampling

    There are two ways of initializing a calibration toolbox class.

    Method 1 - dictionary style (recommended)
    
    .. highlight:: python
    .. code-block:: python
    
        model_cls = CalibrationToolbox.from_dict(
            {
                "num_iter": 8,
                "model": {
                    "param_mins": [0, 0],
                    "param_maxs": [1, 10],
                    "num_samples": 14,
                    "obs_data": [2,4,8,16],
                    "ctrl_data": [1,2,3,4],
                    "callback": run_sim,
                },
                "calibration": {
                    "inference": {"ess_target": 0.3},
                    "sampling": {"max_num_components": 1},
                }
            }
        )

    or

    Method 2 - class style
    
    .. highlight:: python
    .. code-block:: python
    
        model_cls = CalibrationToolbox(
            num_iter = 8,
            model = Model(...),
            calibration = IterativeBayesianFilter(...)
        )

    :param model: Model class
    :param calibration: Iterative Bayesian Filter
    :param num_iter: Number of iteration steps
    """
    #: Model being calibrated on
    model: Type["Model"]

    #: Calibration method (e.g, Iterative Bayesian Filter)
    calibration: Type["IterativeBayesianFilter"]

    #: Number of iterations
    num_iter: int

    #: Current calibration step
    curr_iter: int = 0

    #: List of sigmas
    sigma_list: List = []

    def __init__(
        self,
        model: Type["Model"],
        calibration: Type["IterativeBayesianFilter"],
        num_iter: int,
        curr_iter: int
    ):
        self.model = model

        self.calibration = calibration

        self.num_iter = num_iter

        self.curr_iter = curr_iter


    def run(self):
        """Main calibration loop.
        1. The first iteration starts with a Halton sequence
        2. The following iterations continue by sampling parameter space, until certain criterion is met.
        """
        print(f"Calibration step {self.curr_iter}")
        self.calibration.initialize(self.model)
        self.run_one_iteration()

        for i in range(self.num_iter-1):
            self.curr_iter += 1
            print(f"Calibration step {self.curr_iter}")
            self.run_one_iteration()
            if self.model.sigma_max < self.model.sigma_tol:
                self.num_iter = self.curr_iter + 1
                break

    def run_one_iteration(self):
        """Run GrainLearning for one iteration.
        TODO: more docu needed
        """
        self.model.param_data = self.calibration.list_of_param_data[-1]
        self.model.run()
        self.calibration.solve(self.model)
        self.sigma_list.append(self.model.sigma_max)
    
    def load_and_run_one_iteration(self):
        """Load an existing dataset and run GrainLearning for one iteration
        """
        self.model.load_param_data(self.curr_iter)
        self.model.get_sim_data_files(self.curr_iter)
        self.model.load_sim_data()
        self.calibration.add_curr_param_data_to_list(self.model.param_data)
        self.calibration.solve(self.model)
        self.sigma_list.append(self.model.sigma_max)

    def load_and_process(self, sigma: float):
        """Load an existing dataset and compute posterior distribution with a given sigma
        """
        self.model.load_param_data(self.curr_iter)
        self.model.get_sim_data_files(self.curr_iter)
        self.model.load_sim_data()
        self.calibration.inference.data_assimilation_loop(sigma, self.model)

    def resample(self):
        """Learn and resample from a posterior (proposal) distribution
        """
        self.calibration.posterior_ibf = self.calibration.inference.give_posterior()
        self.calibration.run_sampling(self.model)
        resampled_param_data = self.calibration.list_of_param_data[-1]
        np.save(f'./{self.model.sim_name}_table_{self.curr_iter + 1}.npy',resampled_param_data)
        return resampled_param_data

    @classmethod
    def from_dict(
        cls: Type["CalibrationToolbox"],
        obj: Dict
    ):
        model_type = obj.get("model_type", Model)
        model = model_type.from_dict(obj["model"])

        calibration = IterativeBayesianFilter.from_dict(obj["calibration"])

        return cls(
            model=model,
            calibration=calibration,
            num_iter=obj["num_iter"],
            curr_iter=obj.get("curr_iter", 0)
        )
