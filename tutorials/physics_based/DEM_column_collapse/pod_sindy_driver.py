import numpy as np
from rom_pod_ae import build_snapshots_from_list, center_snapshots, pod
from rom_sindy_gp import fit_sindy_continuous, simulate_and_reconstruct
from rom_io import visualize_2d_field_magnitude, visualize_2d_field, print_error_metrics

def main():
    # Load data
    output = np.load("collumn_collapse_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    # Occ = np.array([output[k]['scalars']['occ'] for k in time_steps])  # (T, nx, ny)
    Occ = np.array([output[k]['scalars']['rho'] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['scalars']['phi'] for k in time_steps])  # (T, nx, ny)
    # Ux = np.array([output[k]['vectors']['disp'][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['vectors']['disp'][1] for k in time_steps])  # (T, nx, ny)
    Ux = np.array([output[k]['vectors']['vel'][0]  for k in time_steps])  # (T, nx, ny)
    Uy = np.array([output[k]['vectors']['vel'][1]  for k in time_steps])  # (T, nx, ny)
    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)

    # DEM timestep size
    dt = 1.97e-5
    dt *= (time_steps[1] - time_steps[0])

    # Build centered snapshots and POD
    X, shape = build_snapshots_from_list([Occ, Ux, Uy])
    Xc, xbar = center_snapshots(X)
    U_r, A, Svals = pod(Xc, energy=0.99)
    r = U_r.shape[1]
    print(f"[POD] kept r = {r} modes (energy 99%)")

    # Fit SINDy
    num_modes = min(3, r)
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A[:, :num_modes], t, poly_degree=1, thresh=0.1, diff="smoothed")
    model.print()

    # Rollout from first state for the whole horizon
    A0 = A[0, :num_modes]
    X_pred = simulate_and_reconstruct(model, U_r[:, :num_modes], A0, t_eval=t, xbar=xbar)

    # Error metrics
    tag = "POD-SINDY"
    print_error_metrics(X, X_pred, tag=tag)

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t)):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, channels=[1, 2], name=tag+'vel_field_magnitude')
        visualize_2d_field(X, X_pred, shape, time_index=i, channel=0, name=tag+'rho_field')
    from rom_io import create_gif_from_pngs
    create_gif_from_pngs(name=tag+'vel_field_magnitude')
    create_gif_from_pngs(name=tag+'rho_field')

    # Select every nth snapshot for train/test split
    # n = 5  # e.g., select every 5th snapshot
    # X_train = X[:, ::n]
    # t_train = t[::n]

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train = X[:, :mid]
    t_train = t[:mid]

    # Center training data
    Xc_train, xbar_train = center_snapshots(X_train)
        
    # POD on training set
    U_r_train, A_train, Svals_train = pod(Xc_train, energy=0.99)
    r_train = U_r_train.shape[1]
    print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

    # Fit SINDy on training POD coefficients
    num_modes_train = min(3, r_train)
    model_train = fit_sindy_continuous(A_train[:, :num_modes_train], t_train, poly_degree=1, thresh=0.1, diff="smoothed")
    model_train.print()

    # Rollout on test set
    A0_test = A_train[0, :num_modes_train]
    X_test_pred = simulate_and_reconstruct(model_train, U_r_train[:, :num_modes_train], A0_test, t_eval=t, xbar=xbar_train)

    # Error metrics on test set
    print_error_metrics(X, X_test_pred, tag="")

    # Visualize the 2D field over time (10 snapshots)
    for i in range(0, len(t)):
        visualize_2d_field_magnitude(X, X_test_pred, shape, time_index=i, channels=[1, 2], name=tag+'test_vel_field_magnitude')
        visualize_2d_field(X, X_test_pred, shape, time_index=i, channel=0, name=tag+'test_rho_field')
    from rom_io import create_gif_from_pngs
    create_gif_from_pngs(name=tag+'test_vel_field_magnitude')
    create_gif_from_pngs(name=tag+'test_rho_field')

if __name__ == "__main__":
    main()
