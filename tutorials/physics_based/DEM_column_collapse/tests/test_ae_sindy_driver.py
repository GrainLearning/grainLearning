import os
import numpy as np
import torch
from rom_pipeline import AutoencoderSindyROM
from rom_io import check_errors


def main():
    data_file = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     "..",
                     "Yade",
                     "column_collapse_sim_data",
                     "column_collapse_15_CG_fields.npy"))
    output = np.load(data_file, allow_pickle=True).item()
    time_steps = list(output.keys())
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    T = len(time_steps)
    t = np.arange(T) * dt

    file = data_file
    channels = ["rho"]

    # Fit and evaluate reconstruction on full data (CPU for repeatability)
    rom = AutoencoderSindyROM(normalization=True,
                              latent_dim=3,
                              poly_degree=1,
                              thresh=0.1,
                              diff="smoothed",
                              epochs=5000,
                              batch_size=64,
                              lr=1e-4,
                              device="cuda",
                              val_split=0.2,
                              patience=200,
                              print_every=50,
                              tag="AE-SINDY")
    rom.fit(data_or_file=file, channels=channels, dt=dt)
    rom.evaluate(rom.channel_list, t, create_visual=False)
    # Check errors metric against baseline data
    errors = np.insert(np.array(rom.errors), 0, rom.global_error)
    check_errors("AE-SINDY", errors)

    # Test extrapolation and interpolation
    tags = ['AE-SINDY_interpolate', 'AE-SINDY_extrapolate']
    X_train_list = []
    # Select every nth snapshot for train/test split
    n = 5  # e.g., select every 5th snapshot
    X_train_list.append([channel[::n, :, :] for channel in rom.channel_list])
    # Select the first half for training
    mid = T // 2
    X_train_list.append([channel[:mid, :, :] for channel in rom.channel_list])
    # Set time steps for interpolation and extrapolation
    dt_list = [n * dt, dt]

    for dt, X_train, tag in zip(dt_list, X_train_list, tags):
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=dt)
        rom.evaluate(file, t, create_visual=False)
        # Check errors metric against baseline data
        errors = np.insert(np.array(rom.errors), 0, rom.global_error)
        check_errors(tag, errors)

if __name__ == "__main__":
    np.random.seed(36)
    torch.manual_seed(0)
    main()