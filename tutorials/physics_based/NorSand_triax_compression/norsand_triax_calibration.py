import numpy as np
from grainlearning import BayesianCalibration
from math import log
from grainlearning.tools import plot_pdf
import matplotlib.pylab as plt


def norsand(x, e0, params):
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
e0 = [0.60, 0.80]  # [-] 'init void ratio'
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
y_obs_1 = norsand(x_obs, e0[0], [gamma, lambda_val, M_tc, N, H, Xim_tc, Ir, nu])
y_obs_2 = norsand(x_obs, e0[1], [gamma, lambda_val, M_tc, N, H, Xim_tc, Ir, nu])
y_obs = np.concatenate([y_obs_1, y_obs_2], axis=0)


def run_sim_original(x, params):
    """Run different realizations of the original model.

    :param x: the input sequence
    :param params: the parameters
    """
    data = []
    for params in params:
        y_i = []
        # Run the model
        for e in e0:
            y_i.append(norsand(x, e, params))
        data.append(np.concatenate(y_i, axis=0))
    return np.array(data)


def run_sim(calib):
    """This is the callback function that runs different realizations of the same model.

    :param calib: The calibration object.
    """
    sim_data = run_sim_original(calib.system.ctrl_data, calib.system.param_data)
    calib.system.set_sim_data(sim_data)


num_cores = 18
param_names = ['gamma', 'lambda', 'M', 'N', 'H', 'Xim', 'Ir', 'nu']
num_samples = int(5 * len(param_names) * log(len(param_names)))

calibration = BayesianCalibration.from_dict(
    {
        "num_iter": 10,
        "callback": run_sim,
        "system": {
            "param_min": [0.7, 0.01, 1.2, 0.2, 25, 2, 100, 0.1],
            "param_max": [1.4, 0.07, 1.5, 0.5, 500, 5, 600, 0.3],
            "param_names": param_names,
            "num_samples": num_samples,
            "obs_names": ['q/p (0.6)', 'e (0.6)', 'q/p (0.8)', 'e (0.8)'],
            "ctrl_name": 'u',
            "obs_data": y_obs,
            "ctrl_data": x_obs,
            "sim_name": 'triax',
            "sigma_tol": 0.01,
        },
        "inference": {
            "Bayes_filter": {
                "ess_target": 0.3,
                "scale_cov_with_max": True,
            },
            "sampling": {
                "max_num_components": 1,
                "random_state": 0,
                # FIXME slice sampling requires rejecting samples whose likelihood are low. However, this process becomes very slow if dimensionality is high.
                "slice_sampling": False,
            },
            "initial_sampling": "LH",
        },
        "save_fig": -1,
    }
)

calibration.run()

true_params = [gamma, lambda_val, M_tc, N, H, Xim_tc, Ir, nu]
plot_pdf('norsand_triax', param_names, calibration.inference.param_data_list, save_fig=0, true_params=true_params)
plt.show()
