"""Mixed Bayesian calibration for DEM column collapse using reduced POD
coefficients A(t) as observables.

- Maintains a global Parametric POD-SINDy ROM trained on all origin runs from
  every iteration.
- Iter > 0 evaluates a fraction of samples with the ROM (ids_surrogate) while
  running YADE only for the rest (ids_origin).
"""
import os
import glob
import numpy as np
from math import log
from grainlearning import BayesianCalibration, IODynamicSystem
from grainlearning.tools import write_dict_to_file
from rom_pipeline import ParametricPodSindyROM
from rom_io import load_2d_trajectory_from_file
from rom_pod_ae import build_snapshots_from_list, transform


PATH = os.path.abspath(os.path.dirname(__file__))
executable = 'yade-batch'
yade_script = f'{PATH}/column_collapse.py'


# -----------------------------
# Configuration and global ROM
# -----------------------------
ROM_CONFIG = {
    'surrogate_fraction': 0.5,              # fraction of samples evaluated via ROM at iter>0
    'channels': ['rho', 'disp_x', 'disp_y'],# channels used to build the POD basis
    'num_modes': 3,                         # number of POD modes for calibration
    'energy': 0.999,                        # POD energy threshold
    'poly_deg_state': 1,                    # SINDy state poly degree
    'poly_deg_param': 1,                    # SINDy parameter poly degree
    'thresh': 0.1,                          # STLSQ threshold
    'diff': 'smoothed',                     # differentiation method
}

GLOBAL_ROM = None
GLOBAL_FILE_LIST = []
GLOBAL_U_LIST = []
GLOBAL_CHANNELS = ROM_CONFIG['channels']
GLOBAL_DT = None
GLOBAL_T = None


def _compute_dt_and_T_from_file(npy_file: str) -> tuple[float, int]:
    """Infer dt and T from a CG .npy snapshot dict file using the 1.97e-5 scaling."""
    data = np.load(npy_file, allow_pickle=True).item()
    time_steps = list(data.keys())
    T = len(time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    return dt, T


def _project_A_from_file(npy_file: str, rom: ParametricPodSindyROM) -> np.ndarray:
    """Load channels from .npy, normalize with rom bounds, and project to POD coeffs A (r, T)."""
    U_list = load_2d_trajectory_from_file(npy_file, channels=GLOBAL_CHANNELS, t_max=GLOBAL_T)
    X, _, _ = build_snapshots_from_list(U_list, normalization=False)
    Xn = transform(X.copy(), rom.channel_bounds)
    A = (Xn.T @ rom.U_r_train).T  # (r,T)
    return A


def run_sim(calib):
    """Mixed evaluation callback with a persistent global parametric ROM.

    Iter 0: run YADE for all samples, train ROM, project to A(t), write sim/param files.
    Iter >0: split samples; run YADE for ids_origin, update ROM; evaluate ids_surrogate via ROM; write all files.
    """
    global GLOBAL_ROM, GLOBAL_FILE_LIST, GLOBAL_U_LIST, GLOBAL_DT, GLOBAL_T

    sim_name = calib.system.sim_name
    iter_id = calib.system.curr_iter
    cwd = os.getcwd()

    # 1) Split into origin/surrogate subsets
    num_total = calib.system.num_samples
    if iter_id == 0:
        ids_origin = np.arange(num_total)
        ids_surrogate = np.array([], dtype=int)
    else:
        rng = np.random.default_rng()
        perm = rng.permutation(num_total)
        frac = float(ROM_CONFIG.get('surrogate_fraction', 0.5))
        frac = min(max(frac, 0.0), 0.95)
        n_sur = int(round(num_total * frac))
        n_sur = min(max(n_sur, 1), num_total - 1)
        ids_surrogate = perm[:n_sur]
        ids_origin = perm[n_sur:]
    calib.ids_origin, calib.ids_surrogate = ids_origin, ids_surrogate
    mag = int(np.floor(np.log10(max(1, num_total - 1))) + 1)

    # 2) Run YADE for origin subset (write a subset table to drive YADE)
    if ids_origin.size > 0:
        from grainlearning.tools import write_to_table
        subset_param_table_name = f'{os.getcwd()}/{sim_name}_Iter{iter_id}_Subset_Samples.txt'
        write_to_table(
            sim_name,
            calib.system.param_data,
            calib.system.param_names,
            iter_id,
            threads=calib.threads,
            table_ids=ids_origin,
            table_name=subset_param_table_name,
        )
        os.system(' '.join([executable, subset_param_table_name, yade_script]))

    # 3) Collect newly produced CG field files for this iteration run, mapped to ids_origin order
    file_list_origin = []
    for i in ids_origin:
        fname = os.path.join(
            cwd,
            f"{sim_name}_Iter{iter_id}_Sample{str(int(i)).zfill(mag)}_CG_fields.npy"
        )
        if os.path.exists(fname):
            file_list_origin.append(fname)
        else:
            raise RuntimeError(f"[WARN] Expected CG file not found: {fname}")

    # 4) Update global training store and (re)fit the ROM
    if ids_origin.size != 0 and len(file_list_origin) != ids_origin.size:
        print(f"[WARN] Found {len(file_list_origin)} CG files but expected {ids_origin.size}.")
    if ids_origin.size > 0:
        GLOBAL_FILE_LIST.extend(file_list_origin)
        GLOBAL_U_LIST.extend([calib.system.param_data[i] for i in ids_origin])

        if GLOBAL_DT is None or GLOBAL_T is None:
            GLOBAL_DT, GLOBAL_T = _compute_dt_and_T_from_file(file_list_origin[0])

        if GLOBAL_ROM is None:
            GLOBAL_ROM = ParametricPodSindyROM(
                normalization=True,
                energy=float(ROM_CONFIG.get('energy', 0.999)),
                num_modes=int(ROM_CONFIG.get('num_modes', 3)),
                poly_deg_state=int(ROM_CONFIG.get('poly_deg_state', 1)),
                poly_deg_param=int(ROM_CONFIG.get('poly_deg_param', 1)),
                thresh=float(ROM_CONFIG.get('thresh', 0.1)),
                diff=str(ROM_CONFIG.get('diff', 'smoothed')),
                tag="POD-SINDy_Param_Global",
            )

        GLOBAL_ROM.fit(
            file_list=GLOBAL_FILE_LIST,
            u_list=[np.asarray(u) for u in GLOBAL_U_LIST],
            channels=GLOBAL_CHANNELS,
            dt=GLOBAL_DT,
            t_max=GLOBAL_T,
        )

    # 5) Update reference A(t) and wire system obs
    ref_path = os.path.join(PATH, "column_collapse_DEM_test_run_CG_fields.npy")
    if not os.path.exists(ref_path):
        raise RuntimeError(f"Reference CG file not found at {ref_path}.")
    dt_ref, t_ref = _compute_dt_and_T_from_file(ref_path)
    if GLOBAL_DT != dt_ref or GLOBAL_T != t_ref:
        raise RuntimeError(f"Reference dt/T ({dt_ref:.2e}, {t_ref}) do not match inferred GLOBAL_DT/T ({GLOBAL_DT:.2e}, {GLOBAL_T}).")
    A_ref = _project_A_from_file(ref_path, GLOBAL_ROM)
    r = A_ref.shape[0]
    A_names = [f"A{i+1}" for i in range(r)]
    t_vec = np.arange(A_ref.shape[1]) * GLOBAL_DT
    ref_txt = os.path.join(cwd, f"{sim_name}_A_reference.txt")
    ref_dict = {A_names[i]: A_ref[i, :].tolist() for i in range(r)}
    ref_dict['t'] = t_vec.tolist()
    write_dict_to_file(ref_dict, ref_txt)
    calib.system.obs_names = A_names
    calib.system.ctrl_name = 't'
    calib.system.obs_data_file = ref_txt
    calib.system.get_obs_data()

    # 6a) Origin subset: project each YADE CG output to A(t)
    for i, file in zip(ids_origin, file_list_origin):
        A_i = _project_A_from_file(file, GLOBAL_ROM).T  # (T,r)
        sim_path = os.path.join(cwd, f"{sim_name}_Iter{iter_id}_Sample{str(i).zfill(mag)}_sim.txt")
        sim_dict = {A_names[j]: A_i[:, j].tolist() for j in range(r)}
        sim_dict['t'] = t_vec.tolist()
        write_dict_to_file(sim_dict, sim_path)
        p = calib.system.param_data[i]
        p_dict = {name: float(val) for name, val in zip(calib.system.param_names, p)}
        p_path = os.path.join(cwd, f"{sim_name}_Iter{iter_id}_Sample{str(i).zfill(mag)}_param.txt")
        write_dict_to_file(p_dict, p_path)

    # 6b) Surrogate subset: rollout via ROM and write
    if ids_surrogate.size > 0:
        for i in ids_surrogate:
            u = calib.system.param_data[i]
            A0 = GLOBAL_ROM.gp_initial.predict(u.reshape(1, -1))[0]
            def u_fun(tau, u_const=u):
                tau = np.atleast_1d(tau)
                return np.tile(u_const, (tau.size, 1))
            A_pred = GLOBAL_ROM.model.simulate(A0, t_vec, u=u_fun)  # (T,r)
            sim_path = os.path.join(cwd, f"{sim_name}_Iter{iter_id}_Sample{str(int(i)).zfill(mag)}_sim.txt")
            sim_dict = {A_names[j]: A_pred[:, j].tolist() for j in range(r)}
            sim_dict['t'] = t_vec.tolist()
            write_dict_to_file(sim_dict, sim_path)
            p = calib.system.param_data[i]
            p_dict = {name: float(val) for name, val in zip(calib.system.param_names, p)}
            p_path = os.path.join(cwd, f"{sim_name}_Iter{iter_id}_Sample{str(int(i)).zfill(mag)}_param.txt")
            write_dict_to_file(p_dict, p_path)


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