import numpy as np
import typing as t
from .models import Model
from .iterativebayesianfilter import IterativeBayesianFilter


class CalibrationToolbox:
    
    model: t.Type["Model"]

    calibration: t.Type["IterativeBayesianFilter"]
    
    sigma_list : list = []

    def __init__(
        self,
        model: t.Type["Model"],
        calibration: t.Type["IterativeBayesianFilter"],
    ):
        self.model = model
        
        self.calibration = calibration

        self.calibration.configure(self.model)

    def run(self):

        for _ in range(10):
            self.model.run()
            new_parameter = self.calibration.solve(self.model)
            self.model.parameters.data = new_parameter
            self.sigma_list.append( self.calibration.sigma_max)


#     @classmethod
#     def from_dict(
#         cls: t.Type["CalibrationToolbox"],
#         obj: dict,
#         simulation_model: t.Type["Model"] = Model,
#         set_sim_obs: bool = True,
#     ):
#         simulations = simulation_model.from_dict(obj["simulations"])

#         observations = Observations.from_dict(obj["observations"])

#         if set_sim_obs:
#             simulations.set_observations(observations)

#         return cls(
#             simulations=simulations,
#             observations=observations,
#             calibration=IterativeBayesianFilter.from_dict(obj["calibration"]),
#         )

#     def set_model(self, model: t.Callable):
#         self.simulations.set_model(model)

