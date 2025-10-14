import os
import numpy as np
import matplotlib.pyplot as plt
from rom_pipeline import PodGpROM, PodSindyROM, AutoencoderSindyROM
import matplotlib as mpl


def get_data_info():
    data_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "Yade",
            "column_collapse_sim_data",
            "column_collapse_15_CG_fields.npy",
        )
    )
    output = np.load(data_file, allow_pickle=True).item()
    time_steps = list(output.keys())
    dt = 1.97e-5 * (time_steps[1] - time_steps[0])
    T = len(time_steps)
    t = np.arange(T) * dt
    return data_file, dt, T, t


def run_pod_gp(file, dt, T, t, channels):
    curves = {}
    # Reconstruction on training data
    rom = PodGpROM(normalization=True, energy=0.99, num_modes=10, tag="POD-GP")
    rom.fit(file, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["POD-GP (recon)"] = np.array(rom.errors)

    # Interpolation for n in [5, 10, 20]
    n_values = [5, 10, 20]
    # Preload full data once for slicing
    full_channels = rom.channel_list
    for n in n_values:
        tag = "POD-GP_interpolate" if n == 5 else f"POD-GP_interpolate_n{n}"
        X_train = [ch[::n, :, :] for ch in full_channels]
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=n * dt)
        rom.evaluate(file, t, create_visual=False)
        curves[f"POD-GP (interp n={n})"] = np.array(rom.errors)

    # Extrapolation (train on first half, evaluate on full)
    tag = "POD-GP_extrapolate"
    mid = T // 2
    X_train = [ch[:mid, :, :] for ch in full_channels]
    rom.tag = tag
    rom.fit(X_train, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["POD-GP (extrap)"] = np.array(rom.errors)
    return curves


def run_pod_sindy(file, dt, T, t, channels):
    curves = {}
    # Reconstruction on training data
    rom = PodSindyROM(
        normalization=True,
        energy=0.99,
        num_modes=3,
        poly_degree=2,
        thresh=0.1,
        diff="smoothed",
        tag="POD-SINDY",
    )
    rom.fit(data_or_file=file, channels=channels, dt=dt)
    # Match original baseline evaluation using preloaded channel_list
    rom.evaluate(rom.channel_list, t, create_visual=False)
    curves["POD-SINDY (recon)"] = np.array(rom.errors)

    # Interpolation for n in [5, 10, 20]
    n_values = [5, 10, 20]
    full_channels = rom.channel_list
    for n in n_values:
        tag = "POD-SINDY_interpolate" if n == 5 else f"POD-SINDY_interpolate_n{n}"
        X_train = [ch[::n, :, :] for ch in full_channels]
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=n * dt)
        rom.evaluate(file, t, create_visual=False)
        curves[f"POD-SINDY (interp n={n})"] = np.array(rom.errors)

    # Extrapolation (train on first half, evaluate on full)
    tag = "POD-SINDY_extrapolate"
    mid = T // 2
    X_train = [ch[:mid, :, :] for ch in full_channels]
    rom.tag = tag
    rom.fit(X_train, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["POD-SINDY (extrap)"] = np.array(rom.errors)
    return curves


def run_ae_sindy(file, dt, T, t, channels, device: str):
    curves = {}
    # Reconstruction on training data (use CPU by default for portability)
    rom = AutoencoderSindyROM(
        normalization=True,
        latent_dim=3,
        poly_degree=1,
        thresh=0.1,
        diff="smoothed",
        epochs=5000,
        batch_size=64,
        lr=1e-4,
    device=device,
        val_split=0.2,
        patience=200,
        print_every=50,
        tag="AE-SINDY",
    )
    rom.fit(data_or_file=file, channels=channels, dt=dt)
    # Match original baseline evaluation using preloaded channel_list
    rom.evaluate(rom.channel_list, t, create_visual=False)
    curves["AE-SINDY (recon)"] = np.array(rom.errors)

    # Interpolation for n in [5, 10, 20]
    n_values = [5, 10, 20]
    full_channels = rom.channel_list
    for n in n_values:
        tag = f"AE-SINDY_interpolate_n{n}"
        X_train = [ch[::n, :, :] for ch in full_channels]
        rom.tag = tag
        rom.fit(X_train, channels=channels, dt=n * dt)
        rom.evaluate(file, t, create_visual=False)
        curves[f"AE-SINDY (interp n={n})"] = np.array(rom.errors)

    # Extrapolation (train on first half, evaluate on full)
    tag = "AE-SINDY_extrapolate"
    mid = T // 2
    X_train = [ch[:mid, :, :] for ch in full_channels]
    rom.tag = tag
    rom.fit(X_train, channels=channels, dt=dt)
    rom.evaluate(file, t, create_visual=False)
    curves["AE-SINDY (extrap)"] = np.array(rom.errors)
    return curves


def plot_errors(curves, t, outfile="rom_errors_over_time.png", group_key=None,
                size=(9.6, 5.4), dpi=150, legend_cols=2, save_pdf=False):
    """
    Plot multiple error curves vs time, slide-friendly (16:9), with consistent colors per ROM.
    Interpolation lines for each ROM use the same base color, with decreasing intensity as n increases.
    Extrapolation uses the same color as the first interpolation for each ROM.
    """

    # Define ROM base colors
    rom_colors = {
        "POD-GP": "#1f77b4",      # blue
        "POD-SINDY": "#2ca02c",   # green
        "AE-SINDY": "#d62728",    # red
    }
    # For each ROM, define interpolation n values (must match code above)
    interp_ns = [5, 10, 20]

    # Helper to extract ROM name from label
    def get_rom_name(label):
        for rom in rom_colors:
            if label.startswith(rom):
                return rom
        # fallback: try to match by substring
        for rom in rom_colors:
            if rom in label:
                return rom
        return None

    # For each ROM, collect all relevant curve keys
    rom_curve_keys = {rom: [] for rom in rom_colors}
    for k in curves:
        rom = get_rom_name(k)
        if rom:
            rom_curve_keys[rom].append(k)

    # For each ROM, assign colors for interpolation/extrapolation
    color_map = {}
    for rom, base_color in rom_colors.items():
        # Use a colormap to generate lighter shades for higher n
        cmap = mpl.colormaps.get_cmap("Blues" if rom == "POD-GP" else
                                      "Greens" if rom == "POD-SINDY" else
                                      "Reds")
        # Assign colors for interpolation n=5,10,20 (darker to lighter)
        interp_colors = [cmap(0.7), cmap(0.5), cmap(0.3)]
        # Map keys to colors
        for i, n in enumerate(interp_ns):
            for k in rom_curve_keys[rom]:
                if f"(interp n={n})" in k or f"_interpolate_n{n}" in k or (n == 5 and "_interpolate" in k and "n" not in k):
                    color_map[k] = interp_colors[i]
        # Extrapolation: use color of n=5
        for k in rom_curve_keys[rom]:
            if "(extrap)" in k or "_extrapolate" in k:
                color_map[k] = interp_colors[0]
        # Recon: use base color
        for k in rom_curve_keys[rom]:
            if "(recon)" in k:
                color_map[k] = base_color

    plt.figure(figsize=size)
    for label, err in curves.items():
        if len(err) != len(t):
            m = min(len(err), len(t))
            x = t[:m]
            y = err[:m]
        else:
            x = t
            y = err
        color = color_map.get(label, None)
        plt.plot(x, y, label=label, lw=1.6, color=color)
    plt.xlabel("time")
    plt.ylabel("relative error")
    if group_key:
        plt.title(f"ROM errors over time - {group_key}")
    else:
        plt.title("ROM errors over time")
    plt.yscale("log")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=8, ncol=legend_cols, frameon=False)
    plt.tight_layout()
    plt.margins(x=0)
    plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        root, _ = os.path.splitext(outfile)
        plt.savefig(root + ".pdf", bbox_inches="tight")
    plt.close()


np.random.seed(36)
# Torch is optional; set seed if available and determine device
try:
    import torch  # type: ignore
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

file, dt, T, t = get_data_info()
channels = ["rho"]

curves_gp = run_pod_gp(file, dt, T, t, channels)
curves_ps = run_pod_sindy(file, dt, T, t, channels)
curves_ae = run_ae_sindy(file, dt, T, t, channels, device)

# Combined dict across all ROMs
combined = {**curves_gp, **curves_ps, **curves_ae}

# Plot interpolation-only (across all ROMs)
interp_all = {k: v for k, v in combined.items() if "(interp" in k}
if interp_all:
    plot_errors(
        interp_all,
        t,
        outfile="rom_errors_interpolation.png",
        group_key="Interpolation (all ROMs)",
        size =(5.5, 4)
    )

# Plot extrapolation-only (across all ROMs)
extrap_all = {k: v for k, v in combined.items() if "(extrap)" in k}
if extrap_all:
    plot_errors(
        extrap_all,
        t,
        outfile="rom_errors_extrapolation.png",
        group_key="Extrapolation (all ROMs)",
        size =(5.5, 4)
    )