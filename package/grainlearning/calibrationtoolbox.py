
from ast import Param
from typing import Type, List, Dict
from .models import Model
from .iterativebayesianfilter import IterativeBayesianFilter


class CalibrationToolbox:
    """This is the main calibration tooblox

  
    There are two ways of initializing the class:

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
    
    #: Number of iteratives
    num_iter: int
    
    #: Current calibration step
    curr_iter:int = 0
    
    #: List of sigmas
    sigma_list : List = []

    def __init__(
        self,
        model: Type["Model"],
        calibration: Type["IterativeBayesianFilter"],
        num_iter: int
    ):
        self.model = model
        
        self.calibration = calibration
        
        self.num_iter = num_iter


    def run(self):
        """Main calibration loop."""
        for self.curr_iter in range(self.num_iter):
            print(f"Calibrationg step {self.curr_iter}")
            self.model.run()
            self.calibration.solve(self.model)
            self.sigma_list.append( self.model.sigma_max)

    @classmethod
    def from_dict(
        cls: Type["CalibrationToolbox"],
        obj: Dict
    ):
        model = Model.from_dict(obj["model"])
        
        calibration = IterativeBayesianFilter.from_dict( obj["calibration"])
        
        return cls(
            model = model,
            calibration=calibration,
            num_iter = obj["num_iter"]
        )

