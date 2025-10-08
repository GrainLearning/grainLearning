import numpy as np
import torch
from rom_pod_ae import build_snapshots_from_list, train_autoencoder
from rom_sindy_gp import fit_sindy_continuous, simulate_and_reconstruct_autoencoder
from rom_io import print_error_metrics, check_errors

def main():
    # Load data
    output = np.load("Yade/column_collapse_sim_data/column_collapse_15_CG_fields.npy", allow_pickle=True).item()
    time_steps = list(output.keys())
    Rho = np.array([output[k].item()['scalars']['rho'] for k in time_steps])

    # DEM base dt times an index stride (assuming uniform integer time_steps)
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    visual_every = 10

    # Build snapshots with normalization
    X, shape, channel_bounds = build_snapshots_from_list([Rho])
    epochs = 5000
    enc, dec, A, hist = train_autoencoder(X, latent_dim=3, epochs=epochs, batch_size=64, lr=1e-4, device="cuda", val_split=0.2, patience=200, print_every=50)

    # Fit SINDy
    t = np.arange(A.shape[0]) * dt
    model = fit_sindy_continuous(A, t, poly_degree=2, thresh=0.1, diff="smoothed")
    model.print()
    
    # Rollout from first state for the whole horizon
    A0 = A[0, :]
    X_pred = simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval=t, device="cuda")

    # decrease the polynomial degree if the rollout is unstable
    while X_pred.shape != X.shape:
        print(f"Warning: prediction shape {X_pred.shape} does not match original {X.shape}")
        model = fit_sindy_continuous(A, t, poly_degree=1, thresh=0.1, diff="smoothed")
        model.print()
        X_pred = simulate_and_reconstruct_autoencoder(model, dec, A0, t_eval=t, device="cuda")

    # Error metrics
    tag = "AE-SINDy"
    _, errors = print_error_metrics(X, X_pred, tag=tag)
    check_errors(tag, errors)

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
        _, errors = print_error_metrics(X, X_test_pred, tag=tag)
        check_errors(tag, errors)

if __name__ == "__main__":
    np.random.seed(36)
    torch.manual_seed(0)
    main()
