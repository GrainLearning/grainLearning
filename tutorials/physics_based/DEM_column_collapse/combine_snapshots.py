import glob
import importlib
import os
import re

import h5py
import numpy as np


DATA_DIR = 'Periodic_simple_shear'
STEP_INTERVAL = 5000

# Output controls
SAVE_COMBINED_NPY = False
SAVE_HDF5_OUTPUT = True
MAKE_MACRO_PLOT = True
VERIFY_EVERY_N_SNAPSHOTS = 100
IS_SINGLE_RUN = False

# Filename marker between prefix and sample id, e.g. "_Sample00".
# Change this to any text marker you use such as "test_run"
SAMPLE_MARKER = "Sample"

SNAPSHOT_GLOB = f"*_{SAMPLE_MARKER}*.npy"

SNAPSHOT_FILE_RE = re.compile(
    rf"^(?P<prefix>.+?)_{re.escape(SAMPLE_MARKER)}(?:_?(?P<sample>\d+))?_(?P<step>\d+)_(?P<suffix>.+)\.npy$"
)
COMBINED_FILE_RE = re.compile(
    rf"^(?P<prefix>.+?)_{re.escape(SAMPLE_MARKER)}(?:_?(?P<sample>\d+))?_(?P<suffix>.+)\.npy$"
)
MACRO_TXT_FILE_RE = re.compile(
    rf"^(?P<prefix>.+?)_{re.escape(SAMPLE_MARKER)}(?:_?(?P<sample>\d+))?(?:_(?P<suffix>.+))?\.txt$"
)

# Optional filters (set to a string to restrict what gets processed)
SNAPSHOT_PREFIX_FILTER = "friction"
SNAPSHOT_SUFFIX_FILTER = "CG_fields"
MACRO_PREFIX_FILTER = SNAPSHOT_PREFIX_FILTER
MACRO_SUFFIX_FILTER = "sim"


def _unwrap(obj):
    if isinstance(obj, np.ndarray) and obj.shape == ():
        try:
            return obj.item()
        except Exception:
            return obj
    if hasattr(obj, "item") and not isinstance(obj, dict):
        try:
            candidate = obj.item()
            if candidate is not obj:
                return candidate
        except Exception:
            pass
    return obj


def _sorted_time_keys(output_dict):
    def _sort_key(k):
        try:
            return (0, float(k))
        except Exception:
            return (1, str(k))

    return sorted(output_dict.keys(), key=_sort_key)


def _add_scalar_channels(snapshot, channels, scalar_keys):
    scalars = snapshot.get("scalars", {})
    if not isinstance(scalars, dict):
        return
    for name in scalar_keys:
        if name not in scalars:
            raise ValueError(f"Missing scalar key '{name}' in snapshot.")
        arr = np.asarray(scalars[name])
        if arr.ndim == 2:
            channels.append((f"scalars/{name}", arr))


def _add_vector_channels(snapshot, channels, vector_keys):
    vectors = snapshot.get("vectors", {})
    if not isinstance(vectors, dict):
        return

    component_labels = ["x", "y"]
    for name in vector_keys:
        if name not in vectors:
            raise ValueError(f"Missing vector key '{name}' in snapshot.")
        arr = np.asarray(vectors[name])
        if arr.ndim == 3:
            if arr.shape[0] != 2:
                raise ValueError(
                    f"Vector field '{name}' expected 2 components (x,y), got {arr.shape[0]}"
                )
            for i in range(arr.shape[0]):
                channels.append((f"vectors/{name}_{component_labels[i]}", arr[i]))


def _add_tensor_channels(snapshot, channels, tensor_keys):
    tensors = snapshot.get("tensors", {})
    if not isinstance(tensors, dict):
        return

    component_labels = ["x", "y"]
    for name in tensor_keys:
        if name not in tensors:
            raise ValueError(f"Missing tensor key '{name}' in snapshot.")
        arr = np.asarray(tensors[name])
        if arr.ndim == 4:
            if arr.shape[0] != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"Tensor field '{name}' expected shape (2,2,x,y), got {arr.shape}"
                )
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    channels.append((f"tensors/{name}_{component_labels[i]}{component_labels[j]}", arr[i, j]))
        elif arr.ndim == 3:
            if arr.shape[0] != 4:
                raise ValueError(
                    f"Tensor field '{name}' componentized form expected 4 components, got {arr.shape[0]}"
                )
            component_order = ["xx", "xy", "yx", "yy"]
            for i in range(arr.shape[0]):
                channels.append((f"tensors/{name}_{component_order[i]}", arr[i]))


def _extract_channels(snapshot, scalar_keys, vector_keys, tensor_keys):
    channels = []
    _add_scalar_channels(snapshot, channels, scalar_keys)
    _add_vector_channels(snapshot, channels, vector_keys)
    _add_tensor_channels(snapshot, channels, tensor_keys)
    return channels


def _build_4d_array(sim_data):
    time_keys = _sorted_time_keys(sim_data)
    if not time_keys:
        raise ValueError("Empty simulation dictionary.")

    first_snapshot = _unwrap(sim_data[time_keys[0]])
    if not isinstance(first_snapshot, dict):
        raise ValueError("Snapshot format is not a dictionary.")

    scalar_dict = first_snapshot.get("scalars", {})
    vector_dict = first_snapshot.get("vectors", {})
    tensor_dict = first_snapshot.get("tensors", {})

    scalar_keys = list(scalar_dict.keys()) if isinstance(scalar_dict, dict) else []
    vector_keys = list(vector_dict.keys()) if isinstance(vector_dict, dict) else []
    tensor_keys = list(tensor_dict.keys()) if isinstance(tensor_dict, dict) else []

    channels0 = _extract_channels(first_snapshot, scalar_keys, vector_keys, tensor_keys)
    if not channels0:
        raise ValueError("No scalar/vector/tensor channels found in snapshot.")

    channel_names = [name for name, _ in channels0]
    ref_shape = channels0[0][1].shape

    for name, arr in channels0:
        if arr.shape != ref_shape:
            raise ValueError(f"Inconsistent channel shape in first snapshot: {name} -> {arr.shape}")

    t_count = len(time_keys)
    c_count = len(channel_names)
    x_size, y_size = ref_shape
    data = np.empty((t_count, c_count, x_size, y_size), dtype=np.float32)

    for t_idx, tk in enumerate(time_keys):
        snapshot = _unwrap(sim_data[tk])
        if not isinstance(snapshot, dict):
            raise ValueError(f"Snapshot at time key {tk!r} is not a dictionary.")

        channels = _extract_channels(snapshot, scalar_keys, vector_keys, tensor_keys)
        names = [n for n, _ in channels]
        if names != channel_names:
            raise ValueError(
                "Channel set changed across time steps. "
                f"Expected {channel_names}, got {names} at key {tk!r}."
            )

        for c_idx, (_, arr) in enumerate(channels):
            if arr.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch at time key {tk!r}, channel {channel_names[c_idx]}: "
                    f"expected {ref_shape}, got {arr.shape}."
                )
            data[t_idx, c_idx] = arr.astype(np.float32, copy=False)

    time_vals = []
    for i, k in enumerate(time_keys):
        try:
            time_vals.append(float(k))
        except Exception:
            time_vals.append(float(i))
    time_array = np.array(time_vals, dtype=np.float64)
    metadata = {
        "scalar_keys": scalar_keys,
        "vector_keys": vector_keys,
        "tensor_keys": tensor_keys,
    }
    return data, channel_names, time_array, metadata

def _parse_snapshot_file(path):
    """Parse file names of the form <prefix>_SampleXX_[<step>_] <suffix>.npy"""
    basename = os.path.basename(path)
    match = SNAPSHOT_FILE_RE.match(basename)
    has_step = True
    if not match:
        match = COMBINED_FILE_RE.match(basename)
        has_step = False
    if not match:
        return None

    prefix = match.group("prefix")
    suffix = match.group("suffix")
    if SNAPSHOT_PREFIX_FILTER is not None and prefix != SNAPSHOT_PREFIX_FILTER:
        return None
    if SNAPSHOT_SUFFIX_FILTER is not None and suffix != SNAPSHOT_SUFFIX_FILTER:
        return None

    sample_str = match.group("sample")
    return {
        "path": path,
        "basename": basename,
        "prefix": prefix,
        "suffix": suffix,
        "sample_id": int(sample_str) if sample_str is not None else 0,
        "sample_width": len(sample_str) if sample_str is not None else 0,
        "step": int(match.group("step")) if has_step else None,
        "has_step": has_step,
    }


def _format_sample_token(sample_id, sample_width):
    if sample_width > 0:
        return f"{SAMPLE_MARKER}{sample_id:0{sample_width}d}"
    return SAMPLE_MARKER


def _collect_snapshot_records(data_dir):
    files = glob.glob(os.path.join(data_dir, SNAPSHOT_GLOB))
    records = []
    for f in files:
        parsed = _parse_snapshot_file(f)
        if parsed is not None:
            records.append(parsed)
    return records


def _discover_sample_ids(data_dir):
    records = _collect_snapshot_records(data_dir)
    if IS_SINGLE_RUN:
        print(f"Found {len(records)} matching snapshot files in single-run mode.")
        return [0]
    sample_ids = sorted({r["sample_id"] for r in records})
    print(f"Found {len(records)} matching snapshot files corresponding to {len(sample_ids)} samples.")
    return sample_ids


def combine_data_per_sample(sample_id, step_interval=STEP_INTERVAL):
    """Combine per-step snapshot files for one sample into an indexed dict."""
    combined_data = {}
    all_records = _collect_snapshot_records(DATA_DIR)
    records = all_records if IS_SINGLE_RUN else [r for r in all_records if r["sample_id"] == sample_id]
    records_with_step = sorted([r for r in records if r["has_step"]], key=lambda r: r["step"])
    records_without_step = [r for r in records if not r["has_step"]]

    if not records_with_step and not records_without_step:
        raise FileNotFoundError(f"No snapshots found for sample {sample_id} in {DATA_DIR}")

    if records_with_step and records_without_step:
        raise ValueError(
            f"Sample {sample_id} has mixed per-step and pre-combined files. "
            f"Please keep only one format in {DATA_DIR}."
        )

    if records_without_step:
        if len(records_without_step) > 1:
            raise ValueError(
                f"Sample {sample_id} has multiple pre-combined files: "
                f"{[r['basename'] for r in records_without_step]}"
            )
        record = records_without_step[0]
        loaded = _unwrap(np.load(record["path"], allow_pickle=True))
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid pre-combined file format in {record['path']}")
        return loaded, [record["path"]]

    records = records_with_step

    format_groups = {(r["prefix"], r["suffix"], r["sample_width"]) for r in records}
    if len(format_groups) > 1:
        raise ValueError(
            f"Sample {sample_id} has mixed filename formats (prefix/suffix/width): {sorted(format_groups)}"
        )

    source_files = [r["path"] for r in records]
    time_steps = [r["step"] for r in records]
    if not all(t2 - t1 == step_interval for t1, t2 in zip(time_steps, time_steps[1:])):
        raise ValueError(
            f"Time steps for sample {sample_id} are not increasing by {step_interval}: {time_steps}"
        )
    expected = list(range(time_steps[0], time_steps[0] + step_interval * len(time_steps), step_interval))
    if time_steps != expected:
        raise ValueError(f"Time steps for sample {sample_id} are not continuous: {time_steps}")

    for i, fname in enumerate(source_files):
        data = np.load(fname, allow_pickle=True)
        combined_data[i] = data
    if SAVE_COMBINED_NPY:
        first = records[0]
        sample_token = _format_sample_token(sample_id, first["sample_width"])
        stem = f"{first['prefix']}_{sample_token}_{first['suffix']}"
        np.save(os.path.join(DATA_DIR, f"{stem}.npy"), combined_data)
    return combined_data, source_files


def save_combined_hdf5(sample_id, combined_data, source_files):
    data, channel_names, time_array, metadata = _build_4d_array(combined_data)
    first = _parse_snapshot_file(source_files[0])
    if first is None:
        out_name = f"sample_{sample_id}_combined.hdf5"
    else:
        sample_token = _format_sample_token(sample_id, first["sample_width"])
        stem = f"{first['prefix']}_{sample_token}_{first['suffix']}"
        out_name = f"{stem}.hdf5"
    out_path = os.path.join(DATA_DIR, out_name)
    with h5py.File(out_path, "w") as h5f:
        data_ds = h5f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        h5f.create_dataset("time", data=time_array)
        h5f.create_dataset(
            "source_files",
            data=np.array([os.path.basename(f) for f in source_files], dtype=h5py.string_dtype("utf-8")),
        )
        h5f.create_dataset("channel_names", data=np.array(channel_names, dtype=h5py.string_dtype("utf-8")))

        str_dtype = h5py.string_dtype("utf-8")
        data_ds.attrs["scalar_keys"] = np.array(metadata["scalar_keys"], dtype=str_dtype)
        data_ds.attrs["vector_keys"] = np.array(metadata["vector_keys"], dtype=str_dtype)
        data_ds.attrs["tensor_keys"] = np.array(metadata["tensor_keys"], dtype=str_dtype)

    # verification: re-read raw source files from disk (sampled in time)
    with h5py.File(out_path, "r") as h5f:
        got_data = h5f["data"]
        got_time = h5f["time"][...]
        got_names = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h5f["channel_names"][...]]

        if got_data.shape != data.shape:
            raise ValueError(
                f"Shape mismatch for sample {sample_id}: h5 {got_data.shape} vs expected {data.shape}"
            )

        if not np.array_equal(got_time, time_array):
            raise ValueError(f"Time mismatch for sample {sample_id}")
        if got_names != channel_names:
            raise ValueError(f"Channel name mismatch for sample {sample_id}")

        t_count = got_data.shape[0]
        verify_stride = max(1, int(VERIFY_EVERY_N_SNAPSHOTS))
        verify_indices = sorted(set(range(0, t_count, verify_stride)) | {t_count - 1})

        first_src_meta = _parse_snapshot_file(source_files[0])
        is_precombined = not first_src_meta["has_step"]

        scalar_keys = metadata["scalar_keys"]
        vector_keys = metadata["vector_keys"]
        tensor_keys = metadata["tensor_keys"]

        loaded = None
        raw_time_keys = None
        if is_precombined:
            loaded = _unwrap(np.load(source_files[0], allow_pickle=True))
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid pre-combined file format in {source_files[0]}")
            raw_time_keys = _sorted_time_keys(loaded)
            if len(raw_time_keys) != t_count:
                raise ValueError(
                    f"Pre-combined file time count mismatch for sample {sample_id}: "
                    f"raw {len(raw_time_keys)} vs h5 {t_count}"
                )
        elif len(source_files) != t_count:
            raise ValueError(
                f"Source file count mismatch for sample {sample_id}: "
                f"files {len(source_files)} vs h5 time steps {t_count}"
            )

        def _load_raw_snapshot(t_idx):
            if is_precombined:
                return _unwrap(loaded[raw_time_keys[t_idx]])
            return _unwrap(np.load(source_files[t_idx], allow_pickle=True))

        max_abs_diff = 0.0
        for t_idx in verify_indices:
            raw_snapshot = _load_raw_snapshot(t_idx)

            if not isinstance(raw_snapshot, dict):
                raise ValueError(f"Invalid snapshot format at t={t_idx} for sample {sample_id}")

            channels = _extract_channels(raw_snapshot, scalar_keys, vector_keys, tensor_keys)
            names = [n for n, _ in channels]
            if names != channel_names:
                raise ValueError(
                    f"Channel mismatch at t={t_idx} for sample {sample_id}: "
                    f"expected {channel_names}, got {names}"
                )

            exp_snapshot = np.stack([arr.astype(np.float32, copy=False) for _, arr in channels], axis=0)
            diff = float(np.max(np.abs(got_data[t_idx] - exp_snapshot)))
            if diff > max_abs_diff:
                max_abs_diff = diff

    if max_abs_diff != 0.0:
        raise ValueError(f"Value mismatch for sample {sample_id:02d}, max_abs_diff={max_abs_diff}")

    print(
        f"Saved and verified HDF5: {out_path} "
        f"(shape={data.shape}, checked={len(verify_indices)}/{data.shape[0]} snapshots, max_abs_diff=0.0)"
    )

def plot_macro_measures(data_dir):
    txt_candidates = glob.glob(os.path.join(data_dir, f"*_{SAMPLE_MARKER}*.txt"))
    txt_records = []
    for path in txt_candidates:
        m = MACRO_TXT_FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        prefix = m.group("prefix")
        suffix = m.group("suffix") or ""
        if MACRO_PREFIX_FILTER is not None and prefix != MACRO_PREFIX_FILTER:
            continue
        if MACRO_SUFFIX_FILTER is not None and suffix != MACRO_SUFFIX_FILTER:
            continue
        if IS_SINGLE_RUN:
            txt_records.append((0, path))
        else:
            sample_str = m.group("sample")
            if sample_str is None:
                continue
            txt_records.append((int(sample_str), path))

    txt_records = sorted(txt_records, key=lambda x: x[0])
    if not txt_records:
        print("No macro trajectory text files found. Skipping plot.")
        return

    plt = importlib.import_module("matplotlib.pyplot")
    def _load_macro_txt_with_header(path):
        header_names = []
        skiprows = 0
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line.startswith("#"):
            header_names = first_line[1:].strip().split()
            skiprows = 1

        data = np.atleast_2d(np.loadtxt(path, skiprows=skiprows))
        if header_names and len(header_names) != data.shape[1]:
            print(
                f"Header/data column mismatch in {os.path.basename(path)}: "
                f"header has {len(header_names)} names, data has {data.shape[1]} columns. "
                f"Falling back to generic names."
            )
            header_names = []
        return data, header_names

    loaded = []
    for sample_id, fname in txt_records:
        data, header_names = _load_macro_txt_with_header(fname)
        if data.shape[1] < 2:
            print(f"Skipping {os.path.basename(fname)}: fewer than 2 columns.")
            continue
        loaded.append((sample_id, fname, data, header_names))

    if not loaded:
        print("No valid macro trajectory text files found (need at least 2 columns).")
        return

    n_cols_set = sorted({data.shape[1] for _, _, data, _ in loaded})
    if len(n_cols_set) > 1:
        print(f"Macro files have varying column counts {n_cols_set}; plotting available columns per file.")

    max_cols = max(data.shape[1] for _, _, data, _ in loaded)
    n_dep = max_cols - 1  # plot columns 1..N-1 vs column 0
    ncols = min(3, n_dep)
    nrows = int(np.ceil(n_dep / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    axs_flat = axs.ravel()

    ref_headers = []
    for _sample_id, _fname, data, header_names in loaded:
        if data.shape[1] == max_cols and header_names:
            ref_headers = header_names
            break

    for sample_id, _fname, data, _header_names in loaded:
        x = data[:, 0]
        label = f"SampleID_{sample_id:02d}"
        for j in range(1, data.shape[1]):
            ax = axs_flat[j - 1]
            ax.plot(x, data[:, j], label=label)

    x_label = ref_headers[0] if len(ref_headers) >= 1 else "col_0"
    for j in range(1, max_cols):
        ax = axs_flat[j - 1]
        y_label = ref_headers[j] if len(ref_headers) > j else f"col_{j}"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

    for k in range(n_dep, len(axs_flat)):
        axs_flat[k].axis("off")

    if loaded:
        axs_flat[0].legend(fontsize=8)

    fig.tight_layout()
    out_name = (
        f"{SNAPSHOT_PREFIX_FILTER}_macro_measures.png"
        if SNAPSHOT_PREFIX_FILTER
        else "macro_measures.png"
    )
    plt.savefig(os.path.join(data_dir, out_name))
    plt.close(fig)


def main():
    if SAVE_HDF5_OUTPUT:
        sample_ids = _discover_sample_ids(DATA_DIR)
        for sample_id in sample_ids:
            combined_data, source_files = combine_data_per_sample(sample_id)
            save_combined_hdf5(sample_id, combined_data, source_files)
            print(f"Combined data for sample {sample_id} saved.")

    if MAKE_MACRO_PLOT:
        plot_macro_measures(DATA_DIR)


if __name__ == "__main__":
    main()