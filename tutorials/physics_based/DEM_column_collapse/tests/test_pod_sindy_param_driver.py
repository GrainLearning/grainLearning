import os
import glob
import numpy as np
from rom_pipeline import ParametricPodSindyROM
from rom_io import check_errors


def main():
    # Collect parametric dataset
    data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Yade", "column_collapse_sim_data"))
    file_list = sorted(glob.glob(os.path.join(data_dir, "column_collapse_*_CG_fields.npy")))
    # Load parameter values from file skipping the first row and the first two columns
    param_file = os.path.join(data_dir, "collapse_Iter0_Samples.txt")
    param_data = np.loadtxt(param_file, skiprows=1, usecols=range(3, 6))
    u_list = [row for row in param_data]

    # Sanity check counts (allow extra files by trimming below)
    if len(file_list) < len(u_list):
        u_list = u_list[:len(file_list)]
    elif len(file_list) > len(u_list):
        file_list = file_list[:len(u_list)]

    # Extract uniform dt and define time vector
    output0 = np.load(file_list[0], allow_pickle=True).item()
    time_steps = list(output0.keys())
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])

    # Cut to a manageable size for testing
    t_max = 200
    max_runs = 16
    file_list = file_list[:max_runs]
    u_list = u_list[:max_runs]
    T = min(t_max, len(time_steps))
    t = np.arange(T) * dt

    channels = ["rho"]
    tag = "POD-SINDy_Param"

    # 1) Train on all runs and evaluate (overall)
    rom = ParametricPodSindyROM(normalization=True, energy=0.99, num_modes=3,
                                poly_deg_state=1, poly_deg_param=1, thresh=0.1,
                                diff="smoothed", tag=tag)
    rom.fit(file_list=file_list, u_list=u_list, channels=channels, dt=dt, t_max=T)
    overall_errors = rom.evaluate_parametric(file_list=file_list, u_list_new=u_list, t=t, create_visual=False)
    check_errors(tag, overall_errors)

    # 2) Split runs into train/test and evaluate both
    mid = len(file_list) // 2
    file_list_train = file_list[:mid]
    file_list_test = file_list[mid:]
    u_list_train = u_list[:mid]
    u_list_test = u_list[mid:]

    rom_train = ParametricPodSindyROM(normalization=True, energy=0.99, num_modes=3,
                                      poly_deg_state=1, poly_deg_param=1, thresh=0.1,
                                      diff="smoothed", tag=tag)
    rom_train.fit(file_list=file_list_train, u_list=u_list_train, channels=channels, dt=dt, t_max=T)

    tag_train = tag + "_Train"
    train_errors = rom_train.evaluate_parametric(file_list=file_list_train, u_list_new=u_list_train, t=t, create_visual=False)
    check_errors(tag_train, train_errors)

    tag_test = tag + "_Test"
    test_errors = rom_train.evaluate_parametric(file_list=file_list_test, u_list_new=u_list_test, t=t, create_visual=False)
    check_errors(tag_test, test_errors)


if __name__ == "__main__":
    np.random.seed(36)
    main()
