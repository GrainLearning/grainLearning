import os
import numpy as np
from rom_pod_ae import build_snapshots_from_list, center_snapshots, pod
from rom_sindy_gp import fit_sindy_continuous, simulate_and_reconstruct
from rom_io import print_error_metrics, check_errors

def main():
    # Load data
    output = np.load("Yade/column_collapse_sim_data/column_collapse_15_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    Rho = np.array([output[k].item()['scalars']['rho'] for k in time_steps])  # (T, nx, ny)

    # DEM base dt times an index stride (assuming uniform integer time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    visual_every = 10

    # Build snapshots with normalization
    X, shape, channel_bounds = build_snapshots_from_list([Rho])
    Xc, xbar = center_snapshots(X)
    U_r, A, Svals = pod(Xc, energy=0.99)
    r = U_r.shape[1]
    print(f"[POD] kept r = {r} modes (energy 99%)")

    # Fit SINDy
    num_modes = min(3, r)
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A[:, :num_modes], t, poly_degree=2, thresh=0.1, diff="smoothed")
    model.print()

    # Rollout from first state for the whole horizon
    A0 = A[0, :num_modes]
    X_pred = simulate_and_reconstruct(model, U_r[:, :num_modes], A0, t_eval=t, xbar=xbar)

    # Error metrics
    tag = "POD-SINDY"
    global_error, errors = print_error_metrics(X, X_pred, tag=tag)
    errors = np.insert(errors, 0, global_error)
    check_errors(tag, errors)

    tags = ['POD-SINDY_interpolate', 'POD-SINDY_extrapolate']
    X_train_list = []
    t_train_list = []
    # Select every nth snapshot for train/test split
    n = 5  # e.g., select every 5th snapshot
    X_train_list.append(X[:, ::n])
    t_train_list.append(t[::n])

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train_list.append(X[:, :mid])
    t_train_list.append(t[:mid])

    for i, (X_train, t_train, tag) in enumerate(zip(X_train_list, t_train_list, tags)):
        Xc_train, xbar_train = center_snapshots(X_train)
            
        # POD on training set
        U_r_train, A_train, Svals_train = pod(Xc_train, energy=0.99)
        r_train = U_r_train.shape[1]
        print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

        # Fit SINDy on training POD coefficients
        num_modes_train = min(3, r_train)
        model_train = fit_sindy_continuous(A_train[:, :num_modes_train], t_train, poly_degree=2, thresh=0.1, diff="smoothed")
        model_train.print()

        # Rollout on test set
        A0_test = A_train[0, :num_modes_train]
        X_test_pred = simulate_and_reconstruct(model_train, U_r_train[:, :num_modes_train], A0_test, t_eval=t, xbar=xbar_train)

        # Error metrics on test set
        print_error_metrics(X, X_test_pred, tag="")
        global_error, errors = print_error_metrics(X, X_pred, tag=tag)
        errors = np.insert(errors, 0, global_error)
        check_errors(tag, errors)

if __name__ == "__main__":
    np.random.seed(36)
    main()
