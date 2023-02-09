import numpy as np

from grainlearning import (
    SMC,
    IterativeBayesianFilter,
    GaussianMixtureModel,
    DynamicSystem,
)


def test_init():
    """Test if the iterative bayesian filter is initialized correctly"""
    smc_cls = SMC(ess_target=0.1)
    gmm_cls = GaussianMixtureModel(max_num_components=5)
    ibf_cls = IterativeBayesianFilter(inference=smc_cls, sampling=gmm_cls)

    ibf_dct = IterativeBayesianFilter.from_dict(
        {
            "inference": {"ess_target": 0.1},
            "sampling": {"max_num_components": 5},
            "num_samples": 2,
        }
    )

    assert isinstance(ibf_cls, IterativeBayesianFilter)
    assert isinstance(gmm_cls, GaussianMixtureModel)

    raw_ibf_dct = ibf_dct.__dict__
    raw_ibf_cls = ibf_cls.__dict__

    raw_ibf_dct.pop("inference")
    raw_ibf_dct.pop("sampling")
    raw_ibf_cls.pop("sampling")
    raw_ibf_cls.pop("inference")

    np.testing.assert_equal(raw_ibf_dct, raw_ibf_cls)


def test_run_inference():
    """Test if the inference runs"""
    system_cls = DynamicSystem.from_dict({
        "param_min": [0, 10],
        "param_max": [10, 100],
        "obs_data": np.random.uniform(0, 100, (3, 4)),
        "num_samples": 3,
        "callback": None,
    })

    system_cls.sim_data = np.random.uniform(0, 100, (3, 3, 4))

    ibf_cls = IterativeBayesianFilter.from_dict(
        {
            "inference": {"ess_target": 0.1},
            "sampling": {"max_num_components": 5},
            "num_samples": system_cls.num_samples,
        }
    )
    print(system_cls.sigma_max)

    ibf_cls.initialize(system=system_cls)

    ibf_cls.run_inference(system=system_cls)

    assert True
