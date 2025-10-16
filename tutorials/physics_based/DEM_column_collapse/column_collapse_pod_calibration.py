"""Bayesian calibration for the DEM column collapse scenario using reduced POD
coefficients A(t) as observables. After running YADE, train a lightweight
Parametric POD-SINDy ROM to obtain a robust POD basis and bounds, project both
reference and current runs to A(t), then calibrate on A(t)."""
import os
import glob
import numpy as np
from math import log
from grainlearning import BayesianCalibration, IODynamicSystem
from grainlearning.tools import write_dict_to_file

# ROM imports
from rom_pipeline import ParametricPodSindyROM
from rom_io import load_2d_trajectory_from_file
from rom_pod_ae import build_snapshots_from_list, transform


PATH = os.path.abspath(os.path.dirname(__file__))
executable = 'yade-batch'
yade_script = f'{PATH}/column_collapse.py'


def run_sim(calib):
    """Run YADE, learn POD basis (via ParametricPodSindyROM), project to A(t),
    and switch calibration I/O to use A(t)."""
    os.system(' '.join([executable, calib.system.param_data_file, yade_script]))
    sim_name = calib.system.sim_name
    iter_id = calib.system.curr_iter
    cwd = os.getcwd()
    pattern = os.path.join(cwd, f"{sim_name}_Iter{iter_id}_Sample*_CG_fields.npy")
    file_list = sorted(glob.glob(pattern))
    if not file_list:
        print(f"[WARN] No combined CG snapshot files found with pattern: {pattern}")
        return

    if getattr(calib.system, 'param_data', None) is None:
        raise RuntimeError("param_data not available on system; cannot train ROM.")
    u_list = [row for row in np.asarray(calib.system.param_data)]

    first = np.load(file_list[0], allow_pickle=True).item()
    time_steps = list(first.keys())
    T = len(time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])

    # Train Parametric POD-SINDy ROM on these runs (channels: choose scalar density 'rho')
    channels = ['rho', 'disp_x', 'disp_y']  # could be "disp", "vel", "scalar_rho_phi", etc
    tag = "POD-SINDy_Param"
    rom = ParametricPodSindyROM(normalization=True, energy=0.999, num_modes=3,
                                poly_deg_state=1, poly_deg_param=1, thresh=0.1,
                                diff="smoothed", tag=tag)
    rom.fit(file_list=file_list, u_list=u_list, channels=channels, dt=dt, t_max=T)

    ref_path = os.path.join(PATH, "column_collapse_DEM_test_run_CG_fields.npy")
    if not os.path.exists(ref_path):
        raise RuntimeError(f"Reference CG file not found at {ref_path}.")
    U_list_ref = load_2d_trajectory_from_file(ref_path, channels=channels, t_max=T)
    X_ref, _, _ = build_snapshots_from_list(U_list_ref, normalization=False)
    X_ref_n = transform(X_ref.copy(), rom.channel_bounds)
    A_ref = (X_ref_n.T @ rom.U_r_train).T  # shape (r, T)

    # Prepare names and time vector
    r = A_ref.shape[0]
    A_names = [f"A{i+1}" for i in range(r)]
    t_vec = np.arange(A_ref.shape[1]) * dt

    # Write reference obs file using GrainLearning helper: columns [A1..Ar, t]
    ref_txt = os.path.join(cwd, f"{sim_name}_A_reference.txt")
    ref_dict = {A_names[i]: A_ref[i, :].tolist() for i in range(r)}
    ref_dict['t'] = t_vec.tolist()
    write_dict_to_file(ref_dict, ref_txt)

    # Write per-sample sim and param files for current iteration
    num_samples = len(file_list)
    mag = int(np.floor(np.log10(max(1, num_samples - 1))) + 1)
    for i in range(num_samples):
        A_i = rom.A_list_train[i]  # (T, r)
        # sim file with columns [A1..Ar, t]
        sim_path = os.path.join(cwd, f"{sim_name}_Iter{iter_id}_Sample{str(i).zfill(mag)}_sim.txt")
        sim_dict = {A_names[j]: A_i[:, j].tolist() for j in range(r)}
        sim_dict['t'] = t_vec.tolist()
        write_dict_to_file(sim_dict, sim_path)

    # Point system to new obs file and names
    calib.system.obs_names = A_names
    calib.system.ctrl_name = 't'
    calib.system.obs_data_file = ref_txt
    calib.system.get_obs_data()

    # Move CG field snapshots to the simulation data directory
    for file in file_list:
        f_name = os.path.relpath(file, os.getcwd())
        os.replace(f'{file}', f'{calib.system.sim_data_sub_dir}/{f_name}')


# Choose parameters to calibrate; keep consistent with YADE's readParamsFromTable
param_names = ['kr', 'eta', 'mu']
num_samples = int(5 * len(param_names) * log(len(param_names)))

calibration = BayesianCalibration.from_dict(
    {
        "curr_iter": 0,
        "num_iter": 5,
        "error_tol": 0.01,
        "callback": run_sim,
        "system": {
            "system_type": IODynamicSystem,
            "param_min": [0.0, 0.0, 1.0],
            "param_max": [1.0, 1.0, 60.0],
            "param_names": param_names,
            "num_samples": num_samples,
            # Placeholder; will be overwritten in run_sim() after ROM training and projection
            "obs_data_file": PATH + '/column_collapse_A_reference.txt',
            "obs_names": ['A1', 'A2', 'A3'],
            "ctrl_name": 't',
            "sim_name": 'column_collapse',
            "sim_data_dir": PATH + '/sim_data/',
            "sim_data_file_ext": '.txt',
        },
        "inference": {
            "Bayes_filter": {"scale_cov_with_max": True,
                             "ess_target": 0.3},
            "sampling": {
                "max_num_components": 1,
                "slice_sampling": True,
            },
        },
        "save_fig": 1,
        "threads": 1,
    }
)

calibration.run()

most_prob_params = calibration.get_most_prob_params()
print(f'Most probable parameter values: {most_prob_params}')