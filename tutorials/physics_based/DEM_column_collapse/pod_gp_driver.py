import numpy as np
from rom_pod_ae import build_snapshots_from_list, center_snapshots, pod, inverse_transform
from rom_sindy_gp import simulate_and_reconstruct_gp
from rom_io import visualize_2d_field_magnitude, visualize_2d_field, print_error_metrics

def main():
    # Load data
    output = np.load("Yade/column_collapse_sim_data/column_collapse_15_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    # Choose the channels you want to compress:
    # Example (as in your snippet): use rho & phi as two “channels”
    # Occ = np.array([output[k]['scalars']['occ'] for k in time_steps])  # (T, nx, ny)
    Rho = np.array([output[k].item()['scalars']['rho'] for k in time_steps])  # (T, nx, ny)
    # For velocity instead, uncomment:
    # Ux = np.array([output[k]['vectors']['vel'][0] for k in time_steps])
    # Uy = np.array([output[k]['vectors']['vel'][1] for k in time_steps])
    Ux = np.array([output[k].item()['vectors']['disp'][0] for k in time_steps])
    Uy = np.array([output[k].item()['vectors']['disp'][1] for k in time_steps])
    # For stress instead, uncomment:
    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)

    # DEM base dt times an index stride (assuming uniform integer time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    visual_every = 10

    # Build snapshots with normalization
    X, shape, channel_bounds = build_snapshots_from_list([Rho, Ux, Uy])
    Xc, xbar = center_snapshots(X)
    U_r, A, Svals = pod(Xc, energy=0.99)
    r_full = U_r.shape[1]
    print(f"[POD] kept r = {r_full} modes (energy 99%)")

    # Simulate with Gaussian Process (sklearn)
    num_modes = min(10, r_full)
    T = A.shape[0]
    t_query = np.arange(T) * dt
    X_pred_gp = simulate_and_reconstruct_gp(U_r[:, :num_modes], A[:, :num_modes], t_query, t_query, xbar=xbar)

    # Error metrics
    tag = "POD-GP"
    global_error, errors = print_error_metrics(X, X_pred_gp, tag=tag)
    errors = np.insert(errors, 0, global_error)
    np.savetxt(f"{tag}_errors.txt", errors)

    # Visualize the 2D field over time
    for i in range(0, len(t_query), visual_every):
        visualize_2d_field(inverse_transform(X, channel_bounds),
                           inverse_transform(X_pred_gp, channel_bounds),
                           shape, time_index=i, channel=0, name='rho_field', tag=tag)
        visualize_2d_field_magnitude(inverse_transform(X, channel_bounds),
                                     inverse_transform(X_pred_gp, channel_bounds),
                                     shape, time_index=i, channels=[1, 2], name='vel_field_magnitude', tag=tag)
    from rom_io import create_gif_from_pngs
    create_gif_from_pngs(name=f'{tag}_rho_field')
    create_gif_from_pngs(name=f'{tag}_vel_field_magnitude')

    tags = ['POD-GP_interpolate', 'POD-GP_extrapolate']
    X_train_list = []
    t_train_list = []
    # Select every nth snapshot for train/test split
    n = 5  # e.g., select every 5th snapshot
    X_train_list.append(X[:, ::n])
    t_train_list.append(t_query[::n])

    # Select the first half for training
    mid = X.shape[1] // 2
    X_train_list.append(X[:, :mid])
    t_train_list.append(t_query[:mid])

    for i, (X_train, t_train, tag) in enumerate(zip(X_train_list, t_train_list, tags)):
        Xc_train, xbar_train = center_snapshots(X_train)
        U_r_train, A_train, Svals_train = pod(Xc_train, energy=0.99)
        r_train = U_r_train.shape[1]
        print(f"[POD] train: kept r = {r_train} modes (energy 99%)")

        # Fit GP on training POD coefficients
        num_modes = min(10, r_train)
        X_pred_gp = simulate_and_reconstruct_gp(U_r_train[:, :num_modes], A_train[:, :num_modes], t_train, t_query, xbar=xbar_train)

        # Error metrics on test set
        global_error, errors = print_error_metrics(X, X_pred_gp, tag=tag)
        errors = np.insert(errors, 0, global_error)
        np.savetxt(f"{tag}_errors.txt", errors)

        # Visualize the 2D field over time
        for i in range(0, len(t_query), visual_every):
            visualize_2d_field(inverse_transform(X, channel_bounds),
                                inverse_transform(X_pred_gp, channel_bounds),
                                shape, time_index=i, channel=0, name='test_rho_field', tag=tag)
            visualize_2d_field_magnitude(inverse_transform(X, channel_bounds),
                                          inverse_transform(X_pred_gp, channel_bounds),
                                          shape, time_index=i, channels=[1, 2], name='test_vel_field_magnitude', tag=tag)
        create_gif_from_pngs(name=f'{tag}_test_rho_field')
        create_gif_from_pngs(name=f'{tag}_test_vel_field_magnitude')

if __name__ == "__main__":
    np.random.seed(36)
    main()
