
from ast import Param
from typing import Type, List, Dict
from .models import Model
from .iterativebayesianfilter import IterativeBayesianFilter


class CalibrationToolbox:
    
    model: Type["Model"]

    calibration: Type["IterativeBayesianFilter"]
    
    num_iter: int
    
    curr_iter:int = 0
    
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
        
        
        obj["ibf"]["num_samples"] = model.num_samples
        
        calibration = IterativeBayesianFilter.from_dict( obj["ibf"])
        
        return cls(
            model = model,
            calibration=calibration,
            num_iter = obj["num_iter"]
        )

