#%%
import numpy as np
from grainlearning import Model, Parameters, Observations

def test_create_run_model():
    """Test if a model can be created and run correctly.
    """
    class MyModel(Model):
        parameters = Parameters(
            names=["k", "t"],
            mins=[100, 0.1],
            maxs=[300, 10],
        )
        observations = Observations(
            data=[100, 200, 300], ctrl=[-0.1, -0.2, -0.3], names=["F"], ctrl_name=["x"]
        )
        num_samples = 10

        def __init__(self):
            self.parameters.generate_halton(self.num_samples)

        def run(self):
            # for each parameter calculate the spring force
            data = []
            
            for params in self.parameters.data:
                F = params[0]*params[1]*self.observations.ctrl
                data.append(np.array(F,ndmin=2))
                
            self.data = np.array(data)

    tst_model = MyModel()

    assert isinstance(tst_model, MyModel)


    tst_model.run()


    assert tst_model.parameters.data.shape ==(10,2)
    assert tst_model.observations.data.shape == (1,3)
    assert tst_model.data.shape == (10,1,3)
