import numpy as np
from rom_pod_ae import build_master_pod
from rom_sindy_gp import fit_sindycp_continuous, simulate_and_reconstruct_cp, fit_predict_gp_sklearn
from rom_io import create_gif_from_pngs, print_global_error

def POD_SINDy_parametric_time(file_list, u_list, t_list, channels="disp", energy=0.99, num_modes=10, t_max=-1):
    """
    End-to-end parametric ROM with POD + SINDy-CP.
    Inputs:
      - file_list: list of npy files with CG fields for different parameter settings
      - u_list: list of parameter vectors (or time series) corresponding to file_list
      - t_list: list of time arrays corresponding to file_list
      - channels: which CG fields to use ('disp', 'vel', 'scalar_rho_phi', etc)
      - energy: POD energy threshold for master basis
      - num_modes: number of POD modes to keep for SINDy-CP training
      - t_max: max number of time steps to load from each file (-1 for all)
    """
    # Build a common POD basis and per-run coefficients
    U_r, A_list_full, X_list, shapes = build_master_pod(file_list, channels=channels, energy=energy, t_max=t_max)
    r_full = U_r.shape[1]
    print(f"[POD] train: kept r = {r_full} modes (energy {energy*100:.1f}%)")

    # Truncate modes consistently across runs
    r_use = min(num_modes, r_full)
    A_list = [A[:, :r_use] for A in A_list_full]
    U_use  = U_r[:, :r_use]

    # Fit SINDy-CP (separate polynomial libraries for state and parameters)
    model = fit_sindycp_continuous(
        A_list=A_list,
        t_list=t_list,
        u_list=u_list,
        poly_deg_state=1,
        poly_deg_param=1,
        thresh=0.1,
        diff="smoothed"
    )
    model.print()
    return model, U_use, A_list, X_list, shapes

def evaluate_parametric_rom(model, U_use, A0_list, X_list, shapes, u_list, t_list, channels="disp", create_visual=False, every=10):
    """
    Evaluate the parametric ROM on multiple runs and optionally create GIFs.
    Inputs:
      - model: trained SINDy-CP model
      - U_use: POD basis used (D, r)
      - A0_list: list of initial POD coefficients per run (r,)
      - X_list: list of original snapshot matrices per run (D, T)
      - shapes: list of (nx, ny) shapes per run
      - u_list: list of parameter vectors (or time series) per run
      - t_list: list of time arrays per run
      - channels: which CG fields were used ('disp', 'vel', 'scalar_rho_phi', etc)
      - create_visual: whether to create GIFs of the field evolution
      - every: save every nth snapshot for GIF
    """
    avg_error = 0.0
    for i, (A0, X, shape) in enumerate(zip(A0_list, X_list, shapes)):
        u  = u_list[i]
        X_pred = simulate_and_reconstruct_cp(model, U_use, A0, t_eval=t_list, u=u)
        tag = f"[run {i}]"
        avg_error += print_global_error(X, X_pred, tag=tag)
        # Save evolution of the 2D field into a GIF
        if create_visual:
            create_gif_from_pngs(X, X_pred, shape, name=f"{channels}_Sample{i:02d}", every=every)
    print(f"Average error across all runs: {avg_error / len(A0_list):.4f}")

def ROM_parametric_time():
    import glob, os
    # Load datasets
    data_dir = 'Yade/column_collapse_sim_data/'
    # Get list of data files for different parameter settings
    file_list = glob.glob(os.path.join(data_dir, "column_collapse_*_CG_fields.npy"))
    # Load parameter values from file skipping the first row and the first two columns
    param_file = os.path.join(data_dir, "collapse_Iter0_Samples.txt")
    param_data = np.loadtxt(param_file, skiprows=1, usecols=range(3,6))
    u_list = list([data for data in param_data])
    # Check if the number of files matches the number of parameter sets
    assert len(file_list) == param_data.shape[0], "Number of data files and parameter sets do not match"
    # Extract time steps from the first file
    output0 = np.load(file_list[0], allow_pickle=True).item()
    dt = 1e3 * 1.97e-5
    t_max = 200
    t_list = (np.array(list(output0.keys())) + 1e4) * dt
    t_list = t_list[:t_max]
    # Prepare lists for trajectories
    channels = "disp"
    energy = 0.99    # retained energy for master POD
    num_modes = 10    # truncation for SINDy-CP training
    # Build and evaluate the parametric ROM
    model, U_use, A_list, X_list, shapes = POD_SINDy_parametric_time(file_list, u_list, t_list, channels=channels, energy=energy, num_modes=num_modes, t_max=t_max)
    A0_list = [A[0] for A in A_list]
    evaluate_parametric_rom(model, U_use, A0_list, X_list, shapes, u_list, t_list, channels=channels, create_visual=True, every=10)

    # --------- Split data into train/test runs and evaluate ----------
    mid = len(file_list) // 2
    file_list_train = file_list[:mid]
    file_list_test = file_list[mid:]
    u_list_train = u_list[:mid]
    u_list_test = u_list[mid:]
    # Build a GP to predict cofficients at time 0 for test runs
    model_train, U_use_train, A_list_train, X_list_train, shapes_train = POD_SINDy_parametric_time(
        file_list_train, u_list_train, t_list, channels=channels, energy=energy, num_modes=num_modes, t_max=t_max)
    A0_train = np.array([A[0] for A in A_list_train])
    A0_test = fit_predict_gp_sklearn(u_list_train, A0_train, u_list_test)
    
    # Evaluate on test runs
    evaluate_parametric_rom(model_train, U_use_train, A_list_test, X_list_test, shapes_test, u_list_test, t_list, channels=channels, create_visual=False)

if __name__ == "__main__":
    ROM_parametric_time()
