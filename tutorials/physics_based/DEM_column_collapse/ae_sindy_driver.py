import numpy as np
from rom_pod_ae import build_snapshots_from_list, train_autoencoder
from rom_sindy_gp import fit_sindy_continuous, simulate_and_reconstruct_autoencoder
from rom_io import visualize_2d_field_magnitude, visualize_2d_field, print_error_metrics, plot_ae_history

def main():
    # Load data
    output = np.load("collumn_collapse_CG_fields.npy", allow_pickle=True).item()
    time_steps = sorted(output.keys())
    Rho = np.array([output[k]['scalars']['rho'] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['scalars']['phi'] for k in time_steps])  # (T, nx, ny)
    # Occ = np.array([output[k]['scalars']['occ'] for k in time_steps])  # (T, nx, ny)
    # Ux = np.array([output[k]['vectors']['disp'][0] for k in time_steps], dtype=np.float64)
    # Uy = np.array([output[k]['vectors']['disp'][1] for k in time_steps], dtype=np.float64)
    Ux = np.array([output[k]['vectors']['vel'][0]  for k in time_steps], dtype=np.float64)
    Uy = np.array([output[k]['vectors']['vel'][1]  for k in time_steps], dtype=np.float64)
    # Ux = np.array([output[k]['tensors']['stress'][0][0] for k in time_steps])  # (T, nx, ny)
    # Uy = np.array([output[k]['tensors']['stress'][1][1] for k in time_steps])  # (T, nx, ny)

    # DEM timestep size
    dt = 1.97e-5
    dt *= (time_steps[1] - time_steps[0])

    # Build centered snapshots and Autoencoder
    X, shape = build_snapshots_from_list([Rho, Ux, Uy])
    epochs = 5000
    enc, dec, A, hist = train_autoencoder(X, latent_dim=2, epochs=epochs, batch_size=64, lr=1e-4, device="cuda", val_split=0.2, patience=200, print_every=50)

    # Save and check the loss curves
    plot_ae_history(hist, savepath="ae_train_val_loss.png")

    # Fit SINDy
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A, t, poly_degree=2, thresh=1, diff="smoothed")
    model.print()
    
    # Rollout from first state for the whole horizon
    A0 = A[0, :]
    X_pred = simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval=t, device="cuda")

    # decrease the polynomial degree if the rollout is unstable
    while X_pred.shape != X.shape:
        print(f"Warning: prediction shape {X_pred.shape} does not match original {X.shape}")
        model = fit_sindy_continuous(A, t, poly_degree=1, thresh=1, diff="smoothed")
        model.print()
        X_pred = simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval=t, device="cuda")

    # Error metrics
    tag = "AE-SINDy"
    print_error_metrics(X, X_pred, tag=tag)

    # Visualize the 2D field over time
    for i in range(0, len(t)):
        visualize_2d_field_magnitude(X, X_pred, shape, time_index=i, channels=[1, 2], name='vel_field_magnitude', tag=tag)
        visualize_2d_field(X, X_pred, shape, time_index=i, channel=0, name='rho_field', tag=tag)
    from rom_io import create_gif_from_pngs
    create_gif_from_pngs(name=f'{tag}_vel_field_magnitude')
    create_gif_from_pngs(name=f'{tag}_rho_field')

    tags = ['AE-SINDY_interpolate', 'AE-SINDY_extrapolate']
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
        # Build encoder/decoder on training set
        enc, dec, A_train, hist = train_autoencoder(X_train, latent_dim=2, epochs=epochs, batch_size=64, lr=1e-4, device="cuda", val_split=0.2, patience=200, print_every=50)
        plot_ae_history(hist, savepath="test_ae_train_val_loss.png")

        # Fit SINDy on training POD coefficients
        model_train = fit_sindy_continuous(A_train, t_train, poly_degree=2, thresh=1, diff="smoothed")
        model_train.print()

        # Rollout on test set
        A0_test = A_train[0, :]
        X_test_pred = simulate_and_reconstruct_autoencoder(model_train, dec, A0_test, t_eval=t, integrator="solve_ivp", device="cuda")

        # decrease the polynomial degree if the rollout is unstable
        while X_test_pred.shape != X.shape:
            print(f"Warning: test prediction shape {X_test_pred.shape} does not match original {X.shape}")
            model_train = fit_sindy_continuous(A_train, t_train, poly_degree=1, thresh=1, diff="smoothed")
            model_train.print()
            X_test_pred = simulate_and_reconstruct_autoencoder(model_train, dec, A0_test, t_eval=t, integrator="solve_ivp", device="cuda")

        # Error metrics on test set
        print_error_metrics(X, X_test_pred, tag=tag)

        # Visualize the 2D field over time
        for i in range(0, len(t)):
            visualize_2d_field_magnitude(X, X_test_pred, shape, time_index=i, channels=[1, 2], name='test_vel_field_magnitude', tag=tag)
            visualize_2d_field(X, X_test_pred, shape, time_index=i, channel=0, name='test_rho_field', tag=tag)
        create_gif_from_pngs(name=f'{tag}_test_vel_field_magnitude')
        create_gif_from_pngs(name=f'{tag}_test_rho_field')

if __name__ == "__main__":
    main()
