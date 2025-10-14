import os
import numpy as np
from rom_pipeline import PodGpROM
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

    # Fit and evaluate reconstruction on training data
    rom = PodGpROM(normalization=True, energy=0.99, num_modes=10, tag="POD-GP")
    rom.fit(file, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    # Check errors metric against baseline data
    errors = np.insert(np.array(rom.errors), 0, rom.global_error)
    check_errors("POD-GP", errors)

    # Test extrapolation and interpolation
    tags = ['POD-GP_interpolate', 'POD-GP_extrapolate']
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
    main()