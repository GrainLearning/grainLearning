
from ast import Param
from typing import Type, List, Dict
from .models import Model,FunctionModel
from .iterativebayesianfilter import IterativeBayesianFilter
from .observations import Observations
from .parameters import Parameters

class CalibrationToolbox:
    
    model: Type["Model"]

    calibration: Type["IterativeBayesianFilter"]
    
    sigma_list : List = []

    def __init__(
        self,
        model: Type["Model"],
        calibration: Type["IterativeBayesianFilter"],
    ):
        self.model = model
        
        self.calibration = calibration

        self.calibration.configure(self.model)

    def run(self):

        for _ in range(5):
            self.model.run()
            new_parameter = self.calibration.solve(self.model)
            self.model.parameters.data = new_parameter
            self.sigma_list.append( self.calibration.sigma_max)

    @classmethod
    def from_dict(
        cls: Type["CalibrationToolbox"],
        obj: Dict
    ):
        input_model = obj["model"]
        
        
        # if model is command line argument 
        if isinstance(input_model,str):
            
            print("command line argument not implemented yet")
            
        # if model is a python function
        elif callable(input_model):
            arguments = obj["arguments"].get("arguments", None)
            model = FunctionModel(input_model,arguments)

        else:
            model = input_model
        
        
        model.observations = Observations.from_dict(obj["observations"])
        model.parameters = Parameters.from_dict(obj["parameters"])

        if model.num_samples is None:
            model.num_samples = obj["num_samples"]
            
        if model.parameters.data is None:
            model.parameters.generate_halton(model)
            
        return cls(
            model = model,
            calibration=IterativeBayesianFilter.from_dict(obj["calibration"]),
        )

