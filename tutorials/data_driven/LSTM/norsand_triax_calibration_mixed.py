import grainlearning.rnn.train as train_rnn
import grainlearning.rnn.predict as predict_rnn
from grainlearning.rnn import preprocessor
import numpy as np
from grainlearning import BayesianCalibration
from math import log


def norsand(x, params):
    # norsand parameters
    gamma = params[0]  # [0.9 - 1.4] 'altitude of CSL @ 1kPa'
    lambda_val = params[1]  # [0.01 - 0.07] 'slope CSL defined on natural log'
    M_tc = params[2]  # [1.2 - 1.5] 'critical state friction ratio triaxial compression'
    N = params[3]  # [0.2 - 0.5] 'volumetric coupling coefficient'
    H = params[4]  # [25 - 500] 'plastic hardening modulus for loading'
    Xim_tc = params[5]  # [2 - 5] 'relates maximum dilatancy to state variable (psi)'
    Ir = params[6]  # [100 - 600] 'shear rigidity'
    nu = params[7]  # [0.1 0.3] 'Poissons ratio'

    # Derived parameters
    eqp_inc = eqp_tot / (1e2 * lst)  # [-] 'increment applied plastic deviatoric strain'
    ratio = 1 / (1 - N) ** ((N - 1) / N)  # [-] 'ratio mean effective stress (p) and image stress (pim)'
    Xim = Xim_tc / (
        1 - Xim_tc * lambda_val / M_tc)  # [-] 'relationship between soil property Xi_tc to Norsand property Xi'

    # Declarations
    p = np.zeros(lst + 1)
    q = np.zeros(lst + 1)
    pim = np.zeros(lst + 1)
    ev = np.zeros(lst + 1)
    eq = np.zeros(lst + 1)
    e = np.zeros(lst + 1)
    psi = np.zeros(lst + 1)

    # Initial conditions
    p[0] = p0
    pim[0] = ratio * p0
    e[0] = e0
    psi[0] = e0 - (gamma - lambda_val * np.log(pim[0])) + lambda_val * np.log(pim[0] / p[0])

    # Loadstep cycle CD test
    for i in range(lst):
        # Update image state
        e[i + 1] = e0 - (1 + e0) * ev[i]
        psi[i + 1] = e[i + 1] - (gamma - lambda_val * np.log(pim[i])) + lambda_val * np.log(pim[i] / p[i])

        # Update image friction ratio
        Mim = M_tc + Xim * N * np.abs(psi[i + 1])

        # Apply hardening increment
        pim_max = p[i] * np.exp(-Xim * psi[i + 1] / M_tc)
        pim_inc = H * (pim_max - pim[i]) * eqp_inc
        pim[i + 1] = pim[i] + pim_inc

        # Calculate plastic volumetric strain increment
        Dp = Mim - q[i] / p[i]
        evp_inc = Dp * eqp_inc

        # Calculate bulk and shear modulus
        mu = Ir * p[i]
        K = mu * (2 * (1 + nu)) / (3 * (1 - 2 * nu))

        # Apply consistency condition to calculate stress ratio increment (drained)
        eta_inc = (pim_inc / pim[i]) / (
            1 / M_tc + 1 / (3.0 - q[i] / p[i]))  # Note: 3.0 is used instead of 3 in the denominator
        # Calculate mean effective stress
        p_inc = p[i] * eta_inc / (3.0 - q[i] / p[i])
        p[i + 1] = p[i] + p_inc

        # Calculate new stress ratio and shear stress
        eta = M_tc * (1 + np.log(pim[i + 1] / p[i + 1]))
        q[i + 1] = eta * p[i + 1]

        # Update volumetric and deviatoric strain increments
        eve_inc = p_inc / K
        eqe_inc = (q[i + 1] - q[i]) / (3 * mu)
        ev[i + 1] = ev[i] + eve_inc + evp_inc
        eq[i + 1] = eq[i] + eqe_inc + eqp_inc

    return np.stack([(q / p)[::int(lst / 100)], e[::int(lst / 100)]], axis=0)


# CD input
p0 = 130.0  # [kPa] 'inital pressure'
e0 = 0.60  # [-] 'init void ratio'
eqp_tot = 15.0  # [%] 'total applied plastic deviatoric strain'
lst = int(1e3)  # [-] 'loadsteps'

# ground truth parameter
gamma = 0.816  # [0.7 - 1.4] 'altitude of CSL @ 1kPa'
lambda_val = 0.015  # [0.01 - 0.07] 'slope CSL defined on natural log'
M_tc = 1.26  # [1.2 - 1.5] 'critical state friction ratio triaxial compression'
N = 0.4  # [0.2 - 0.5] 'volumetric coupling coefficient'
H = 200.0  # [25 - 500] 'plastic hardening modulus for loading'
Xim_tc = 3.8  # [2 - 5] 'relates maximum dilatancy to state variable (psi)'
Ir = 200.0  # [100 - 600] 'shear rigidity'
nu = 0.20  # [0.1 0.3] 'Poissons ratio'

# generate synthetic data
x_obs = np.arange(lst + 1) / lst * eqp_tot
x_obs = x_obs[::int(lst / 100)]
y_obs = norsand(x_obs, [gamma, lambda_val, M_tc, N, H, Xim_tc, Ir, nu])

# Define the configuration dictionary for the ML surrogate
my_config = {
    'input_data': None,
    'param_data': None,
    'output_data': None,
    'train_frac': 0.7,
    'val_frac': 0.2,
    'window_size': 10,
    'window_step': 1,
    'patience': 25,
    'epochs': 100,
    'learning_rate': 1e-4,
    'lstm_units': 128,
    'dense_units': 128,
    'batch_size': 64,
    'standardize_outputs': True,
    'save_weights_only': True
}


def run_sim_original(x, params):
    """Run different realizations of the original model.

    :param x: the input sequence
    :param params: the parameters
    """
    data = []
    for params in params:
        # Run the model
        y = norsand(x, params)
        data.append(np.array(y, ndmin=2))
    return np.array(data)


def run_sim(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    sim_data = run_sim_original(calib.system.ctrl_data, calib.system.param_data)
    calib.system.set_sim_data(sim_data)


def run_sim_surrogate(params_origin, output_origin, params_surrogate):
    """Train the ML surrogate and evaluate model output with the ML surrogate.

    :param params_origin: The parameter data used by the original model.
    :param output_origin: The output data produced by the original model.
    :param params_surrogate: The parameter data to be used by the ML surrogate.
    """
    # expend the parameter and output data
    my_config['param_data'] = np.vstack([my_config['param_data'], params_origin])
    my_config['output_data'] = np.vstack([my_config['output_data'], output_origin])

    preprocessor_lstm = preprocessor.PreprocessorLSTM.from_dict(my_config)
    _ = train_rnn.train_without_wandb(preprocessor_lstm, model=calibration.model, config=my_config)
    model, train_stats, config = predict_rnn.get_pretrained_model('outputs')
    calibration.model = model

    # run the surrogate for the second half of the samples
    data_inputs = preprocessor_lstm.prepare_input_data(params_surrogate)
    # make predictions with the trained model
    output_surrogate = predict_rnn.predict_batch(model, data_inputs, train_stats, config,
                                                 batch_size=params_surrogate.shape[0])
    # converting the predictions to GL format (temporal dimension at the end)
    output_surrogate = np.moveaxis(output_surrogate, 1, -1)

    return output_surrogate


# 3. Define the callback function using the ML surrogate
def run_sim_mixed(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    calib.system.num_steps = 101
    # if first iteration, run the original function
    if calib.curr_iter == 0:
        sim_data = run_sim_original(calib.system.ctrl_data, calib.system.param_data)
        calib.system.set_sim_data(sim_data)
        my_config['input_data'] = calib.system.ctrl_data
        my_config['param_data'] = calib.system.param_data
        my_config['output_data'] = calib.system.sim_data
        calibration.model = None
    else:
        # split samples into two subsets to be used with the original function and the ML surrogate
        np.random.seed()
        ids = np.random.permutation(len(calib.system.param_data))
        split_index = int(len(ids) * 0.5)
        ids_origin, ids_surrogate = ids[:split_index], ids[split_index:]
        calib.ids_origin, calib.ids_surrogate = ids_origin, ids_surrogate

        # run the original function for the first half of the samples
        param_data_origin = calib.system.param_data[ids_origin]
        sim_data_origin = run_sim_original(calib.system.ctrl_data, param_data_origin)

        # run the surrogate for the second half of the samples
        param_data_surrogate = calib.system.param_data[ids_surrogate]
        sim_data_surrogate = run_sim_surrogate(param_data_origin, sim_data_origin, param_data_surrogate)

        # put the two subsets of simulation data together according to the original order
        sim_data = np.zeros(
            [calib.system.num_samples, calib.system.num_obs, calib.system.num_steps - my_config['window_size']])
        sim_data[ids_surrogate] = sim_data_surrogate
        sim_data[ids_origin] = sim_data_origin[:, :, :-my_config['window_size']]

        # set `sim_data` to system
        calib.system.set_sim_data(sim_data)
        calib.system.num_steps = sim_data.shape[2]


num_cores = 18
param_names = ['gamma', 'lambda', 'M', 'N', 'H', 'Xim', 'Ir', 'nu']
num_samples = int(10 * len(param_names) * log(len(param_names)))

calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 10,
        "callback": run_sim_mixed,
        "system": {
            "param_min": [0.7, 0.01, 1.2, 0.2, 25, 2, 100, 0.1],
            "param_max": [1.4, 0.07, 1.5, 0.5, 500, 5, 600, 0.3],
            "param_names": param_names,
            "num_samples": num_samples,
            "obs_names": ['q/p', 'e'],
            "ctrl_name": 'u',
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "sim_name": 'triax',
            "sigma_tol": 0.01,
        },
        "calibration": {
            "inference": {
                "ess_target": 0.3,
                "scale_cov_with_max": True,
            },
            "sampling": {
                "max_num_components": 10,
                "n_init": 1,
                "random_state": 0,
                "slice_sampling": False,
            },
            "initial_sampling": "halton",
        },
        "save_fig": -1,
    }
)

calibration.run()

import matplotlib.pylab as plt
from grainlearning.tools import plot_posterior, plot_param_data, plot_pdf

plot_posterior('test', param_names, calibration.system.param_data[calibration.ids_surrogate],
               calibration.calibration.inference.posteriors[:, calibration.ids_surrogate])
plot_posterior('test', param_names, calibration.system.param_data[calibration.ids_origin],
               calibration.calibration.inference.posteriors[:, calibration.ids_origin])
plt.show()

plot_pdf('test', param_names, [calibration.system.param_data[calibration.ids_surrogate],
                               calibration.system.param_data[calibration.ids_origin]])
plt.show()