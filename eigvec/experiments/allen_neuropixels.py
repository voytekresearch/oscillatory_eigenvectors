"""Allen Neuropixels LFP helpers for stationary row-ensemble experiments."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import SignalMatrix


def patch_allensdk_pandas_grouped_uniques() -> None:
    """Patch AllenSDK's array-valued metadata assignment for newer pandas.

    AllenSDK initializes ``ecephys_structure_acronyms`` as an integer column and
    then writes arrays into it. Recent pandas versions reject that dtype change.
    Creating the destination as object dtype keeps AllenSDK metadata loading
    usable without modifying site-packages.
    """

    import allensdk.brain_observatory.ecephys.ecephys_project_cache as epc

    def get_grouped_uniques(this, other, foreign_key, field_key, unique_key, inplace=False):
        if not inplace:
            this = this.copy()

        uniques = other.groupby(foreign_key).apply(
            lambda grp: pd.DataFrame(grp)[field_key].dropna().astype(str).unique()
        )
        this[unique_key] = pd.Series([tuple()] * len(this), index=this.index, dtype="object")
        for index, values in uniques.items():
            if index in this.index:
                this.at[index, unique_key] = tuple(values)

        if not inplace:
            return this

    epc.get_grouped_uniques = get_grouped_uniques


def read_allen_metadata(cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read Allen project metadata CSVs directly from the local cache."""

    cache_dir = Path(cache_dir)
    tables = {}
    for name in ("sessions", "probes", "channels", "units"):
        path = cache_dir / f"{name}.csv"
        if path.exists():
            tables[name] = pd.read_csv(path)
    required = {"sessions", "probes", "channels"}
    missing = sorted(required.difference(tables))
    if missing:
        raise FileNotFoundError(f"Missing Allen metadata tables in {cache_dir}: {missing}")
    return tables


def discover_lfp_assets(cache_dir: str | Path, metadata: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Return locally available probe-level LFP NWBs with session/probe metadata."""

    cache_dir = Path(cache_dir)
    rows = []
    for path in sorted(cache_dir.glob("session_*/probe_*_lfp.nwb")):
        session_match = re.search(r"session_(\d+)", str(path))
        probe_match = re.search(r"probe_(\d+)_lfp\.nwb$", path.name)
        if not session_match or not probe_match:
            continue
        rows.append(
            {
                "session_id": int(session_match.group(1)),
                "probe_id": int(probe_match.group(1)),
                "lfp_path": path,
                "lfp_size_gb": path.stat().st_size / 1e9,
            }
        )

    assets = pd.DataFrame(rows)
    if assets.empty:
        return assets

    if metadata is not None:
        probes = metadata.get("probes", pd.DataFrame()).copy()
        sessions = metadata.get("sessions", pd.DataFrame()).copy()
        if not probes.empty:
            probe_cols = ["id", "name", "lfp_sampling_rate", "ecephys_session_id", "has_lfp_data"]
            probe_cols = [col for col in probe_cols if col in probes.columns]
            assets = assets.merge(
                probes[probe_cols],
                left_on="probe_id",
                right_on="id",
                how="left",
                suffixes=("", "_probe"),
            )
        if not sessions.empty:
            session_cols = ["id", "session_type", "genotype", "sex", "age_in_days", "has_nwb"]
            session_cols = [col for col in session_cols if col in sessions.columns]
            assets = assets.merge(
                sessions[session_cols],
                left_on="session_id",
                right_on="id",
                how="left",
                suffixes=("", "_session"),
            )
    return assets


def discover_session_nwb_assets(cache_dir: str | Path) -> pd.DataFrame:
    """Return locally available session NWBs, which contain stimulus tables."""

    cache_dir = Path(cache_dir)
    rows = []
    for path in sorted(cache_dir.glob("session_*/session_*.nwb")):
        match = re.search(r"session_(\d+)\.nwb$", path.name)
        if not match:
            continue
        rows.append(
            {
                "session_id": int(match.group(1)),
                "session_nwb_path": path,
                "session_nwb_size_gb": path.stat().st_size / 1e9,
            }
        )
    return pd.DataFrame(rows)


def read_lfp_electrode_table(lfp_path: str | Path) -> pd.DataFrame:
    """Read the electrode rows that are actually present in a probe LFP NWB."""

    lfp_path = Path(lfp_path)
    with h5py.File(lfp_path, "r") as h5:
        data_path = _find_lfp_data_path(h5)
        lfp_group = h5[data_path].parent
        electrode_region = np.asarray(lfp_group["electrodes"][:], dtype=int)
        electrode_group = h5["general/extracellular_ephys/electrodes"]

        def col(name):
            values = electrode_group[name][:]
            if values.dtype.kind == "O":
                return [_decode(value) for value in values]
            return values

        electrode_table = pd.DataFrame(
            {
                "nwb_electrode_row": np.arange(len(electrode_group["id"][:])),
                "channel_id": col("id"),
                "local_index": col("local_index"),
                "probe_vertical_position": col("probe_vertical_position"),
                "probe_horizontal_position": col("probe_horizontal_position"),
                "region": col("location"),
            }
        )

    table = electrode_table.iloc[electrode_region].copy().reset_index(drop=True)
    table.insert(0, "lfp_column", np.arange(len(table)))
    table["region"] = table["region"].replace("", np.nan)
    return table


def choose_lfp_channels(
    electrode_table: pd.DataFrame,
    *,
    target_region: str | None = None,
    min_channels: int = 8,
    max_channels: int | None = None,
    allow_region_fallback: bool = True,
) -> pd.DataFrame:
    """Choose a local contiguous channel set from one probe LFP electrode table."""

    table = electrode_table.copy()
    region = target_region
    if region is not None:
        selected = table[table["region"].eq(region)].copy()
        if len(selected) < min_channels and allow_region_fallback:
            region = None
        elif len(selected) < min_channels:
            raise RuntimeError(f"Only {len(selected)} LFP channels found for region {target_region!r}.")

    if region is None:
        counts = table.dropna(subset=["region"])["region"].value_counts()
        counts = counts[counts >= min_channels]
        if counts.empty:
            raise RuntimeError(f"No region has at least {min_channels} LFP channels.")
        region = str(counts.index[0])
        selected = table[table["region"].eq(region)].copy()

    selected = selected.sort_values("probe_vertical_position").reset_index(drop=True)
    if max_channels is not None and len(selected) > max_channels:
        start = (len(selected) - max_channels) // 2
        selected = selected.iloc[start:start + max_channels].copy().reset_index(drop=True)
    selected["selected_region"] = region
    return selected


def continuous_time_block_conditions(
    *,
    first=(0.05, 0.45),
    second=(0.55, 0.95),
    labels=("early_continuous", "late_continuous"),
) -> list[dict]:
    """Two broad recording fractions for LFP-only stationarity checks."""

    return [
        {"condition": labels[0], "start_fraction": first[0], "stop_fraction": first[1]},
        {"condition": labels[1], "start_fraction": second[0], "stop_fraction": second[1]},
    ]


def read_stimulus_presentations(session_nwb_path: str | Path) -> pd.DataFrame:
    """Read Allen stimulus presentations from a local session NWB."""

    from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

    session = EcephysSession.from_nwb_path(str(session_nwb_path))
    presentations = session.stimulus_presentations.copy()
    presentations = presentations.reset_index()
    if "id" in presentations.columns and "stimulus_presentation_id" not in presentations.columns:
        presentations = presentations.rename(columns={"id": "stimulus_presentation_id"})
    return presentations


def stimulus_blocks_from_presentations(
    presentations: pd.DataFrame,
    *,
    stimulus_names: list[str] | tuple[str, ...] | None = None,
    max_gap_seconds: float = 1.0,
    min_block_seconds: float = 0.0,
) -> pd.DataFrame:
    """Merge adjacent presentations of the same stimulus into longer blocks."""

    required = {"stimulus_name", "start_time", "stop_time"}
    missing = required.difference(presentations.columns)
    if missing:
        raise ValueError(f"Stimulus table is missing required columns: {sorted(missing)}")

    df = presentations.dropna(subset=["stimulus_name", "start_time", "stop_time"]).copy()
    df["stimulus_name"] = df["stimulus_name"].astype(str)
    if stimulus_names is not None:
        stimulus_names = [str(name) for name in stimulus_names]
        df = df[df["stimulus_name"].isin(stimulus_names)]
    df = df.sort_values("start_time")

    blocks = []
    current = None
    for _, row in df.iterrows():
        name = str(row["stimulus_name"])
        start = float(row["start_time"])
        stop = float(row["stop_time"])
        if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
            continue

        if current is not None and name == current["stimulus_name"] and start <= current["stop_time"] + max_gap_seconds:
            current["stop_time"] = max(current["stop_time"], stop)
            current["n_presentations"] += 1
            continue

        if current is not None:
            blocks.append(current)
        current = {
            "stimulus_name": name,
            "condition": name,
            "block_index": 0,
            "start_time": start,
            "stop_time": stop,
            "n_presentations": 1,
        }

    if current is not None:
        blocks.append(current)
    if not blocks:
        return pd.DataFrame(columns=["stimulus_name", "condition", "block_index", "start_time", "stop_time", "n_presentations", "duration"])

    blocks = pd.DataFrame(blocks)
    blocks["duration"] = blocks["stop_time"] - blocks["start_time"]
    blocks = blocks[blocks["duration"] >= min_block_seconds].copy()
    blocks["block_index"] = blocks.groupby("stimulus_name").cumcount()
    return blocks.reset_index(drop=True)


def load_continuous_lfp_condition_matrices(
    lfp_assets: pd.DataFrame,
    *,
    target_region: str | None = "CA1",
    conditions: list[dict] | None = None,
    window_seconds: float = 1.0,
    step_seconds: float = 0.5,
    min_channels: int = 8,
    max_channels: int | None = 32,
    max_rows_per_condition: int = 30_000,
    random_state: int = 0,
    allow_region_fallback: bool = True,
    dtype=np.float32,
) -> dict[str, SignalMatrix]:
    """Build condition matrices from continuous probe LFP without filtering.

    Conditions are recording-time ranges by default. This is intentionally a
    stationarity/circulant-structure test, not a stimulus-locked evoked analysis.
    """

    if conditions is None:
        conditions = continuous_time_block_conditions()
    rng = np.random.default_rng(random_state)
    lfp_assets = lfp_assets.copy()
    if lfp_assets.empty:
        raise RuntimeError("No local LFP assets were provided.")

    blocks = {condition["condition"]: [] for condition in conditions}
    rows = {condition["condition"]: [] for condition in conditions}
    fs_seen = []

    for asset in lfp_assets.itertuples(index=False):
        lfp_path = Path(asset.lfp_path)
        electrode_table = read_lfp_electrode_table(lfp_path)
        selected_channels = choose_lfp_channels(
            electrode_table,
            target_region=target_region,
            min_channels=min_channels,
            max_channels=max_channels,
            allow_region_fallback=allow_region_fallback,
        )
        channel_columns = selected_channels["lfp_column"].to_numpy(dtype=int)

        with h5py.File(lfp_path, "r") as h5:
            data_path = _find_lfp_data_path(h5)
            data = h5[data_path]
            timestamps = h5[str(Path(data_path).parent / "timestamps")][:]
            fs = float(1.0 / np.median(np.diff(timestamps[: min(len(timestamps), 10_000)])))
            fs_seen.append(fs)
            window_samples = int(round(window_seconds * fs))
            step_samples = int(round(step_seconds * fs))
            if window_samples <= 0 or step_samples <= 0:
                raise ValueError("window_seconds and step_seconds must be positive.")

            n_samples = data.shape[0]
            for condition in conditions:
                label = condition["condition"]
                start_sample = int(round(float(condition["start_fraction"]) * n_samples))
                stop_sample = int(round(float(condition["stop_fraction"]) * n_samples))
                stop_sample = min(stop_sample, n_samples)
                starts = np.arange(start_sample, stop_sample - window_samples + 1, step_samples)
                if len(starts) == 0:
                    continue

                max_windows = max(1, max_rows_per_condition // max(1, len(channel_columns) * len(lfp_assets)))
                if len(starts) > max_windows:
                    starts = np.sort(rng.choice(starts, size=max_windows, replace=False))

                X_parts = []
                row_parts = []
                for window_index, start in enumerate(starts):
                    stop = int(start + window_samples)
                    # HDF5 indexing keeps channel columns in LFP-file order.
                    segment = np.asarray(data[start:stop, channel_columns], dtype=dtype).T
                    X_parts.append(segment)
                    meta = selected_channels.copy()
                    meta["session_id"] = int(asset.session_id)
                    meta["probe_id"] = int(asset.probe_id)
                    meta["probe_name"] = getattr(asset, "name", "")
                    meta["lfp_path"] = str(lfp_path)
                    meta["condition"] = label
                    meta["window_index"] = int(window_index)
                    meta["window_start_sample"] = int(start)
                    meta["window_stop_sample"] = int(stop)
                    meta["window_start_time"] = float(timestamps[start])
                    meta["window_stop_time"] = float(timestamps[stop - 1])
                    meta["window_seconds"] = window_samples / fs
                    meta["step_seconds"] = step_samples / fs
                    meta["fs"] = fs
                    row_parts.append(meta)

                blocks[label].append(np.vstack(X_parts))
                rows[label].append(pd.concat(row_parts, ignore_index=True))

    matrices = {}
    if not fs_seen:
        raise RuntimeError("No LFP matrices could be built.")
    fs = float(np.median(fs_seen))
    for label, parts in blocks.items():
        if not parts:
            continue
        X = np.vstack(parts).astype(dtype, copy=False)
        row_table = pd.concat(rows[label], ignore_index=True)
        row_table["matrix_row"] = np.arange(len(row_table))
        matrices[label] = SignalMatrix(X, row_table, fs=fs, condition=label)
    if not matrices:
        raise RuntimeError("No LFP matrices could be built for the requested conditions.")
    return matrices


def load_stimulus_lfp_condition_matrices(
    lfp_assets: pd.DataFrame,
    session_nwb_assets: pd.DataFrame,
    *,
    stimulus_names: list[str] | tuple[str, ...],
    target_region: str | None = "VISam",
    window_seconds: float = 0.5,
    step_seconds: float = 0.5,
    onset_offset_seconds: float = 0.0,
    max_gap_seconds: float = 1.0,
    min_channels: int = 8,
    max_channels: int | None = 32,
    max_rows_per_condition: int = 30_000,
    random_state: int = 0,
    allow_region_fallback: bool = True,
    dtype=np.float32,
) -> dict[str, SignalMatrix]:
    """Build condition matrices from local LFP windows grouped by stimulus type."""

    if lfp_assets.empty:
        raise RuntimeError("No local LFP assets were provided.")
    if session_nwb_assets.empty:
        raise RuntimeError("Stimulus mode requires local session NWBs with stimulus tables.")

    rng = np.random.default_rng(random_state)
    session_nwb_by_id = {
        int(row.session_id): Path(row.session_nwb_path)
        for row in session_nwb_assets.itertuples(index=False)
    }
    lfp_assets = lfp_assets[lfp_assets["session_id"].isin(session_nwb_by_id)].copy()
    if lfp_assets.empty:
        raise RuntimeError("No local LFP assets have matching local session NWBs.")

    stimulus_names = [str(name) for name in stimulus_names]
    blocks_by_condition = {name: [] for name in stimulus_names}
    rows_by_condition = {name: [] for name in stimulus_names}
    presentations_by_session = {}
    fs_seen = []

    for asset in lfp_assets.itertuples(index=False):
        session_id = int(asset.session_id)
        if session_id not in presentations_by_session:
            presentations_by_session[session_id] = read_stimulus_presentations(session_nwb_by_id[session_id])
        stimulus_blocks = stimulus_blocks_from_presentations(
            presentations_by_session[session_id],
            stimulus_names=stimulus_names,
            max_gap_seconds=max_gap_seconds,
            min_block_seconds=window_seconds + onset_offset_seconds,
        )
        if stimulus_blocks.empty:
            continue

        lfp_path = Path(asset.lfp_path)
        electrode_table = read_lfp_electrode_table(lfp_path)
        selected_channels = choose_lfp_channels(
            electrode_table,
            target_region=target_region,
            min_channels=min_channels,
            max_channels=max_channels,
            allow_region_fallback=allow_region_fallback,
        )
        channel_columns = selected_channels["lfp_column"].to_numpy(dtype=int)

        with h5py.File(lfp_path, "r") as h5:
            data_path = _find_lfp_data_path(h5)
            data = h5[data_path]
            timestamps = h5[str(Path(data_path).parent / "timestamps")][:]
            fs = float(1.0 / np.median(np.diff(timestamps[: min(len(timestamps), 10_000)])))
            fs_seen.append(fs)
            window_samples = int(round(window_seconds * fs))
            step_time = float(step_seconds)
            if window_samples <= 0 or step_time <= 0:
                raise ValueError("window_seconds and step_seconds must be positive.")

            for condition in stimulus_names:
                condition_blocks = stimulus_blocks[stimulus_blocks["condition"].eq(condition)]
                candidates = []
                for block in condition_blocks.itertuples(index=False):
                    start_time = float(block.start_time) + onset_offset_seconds
                    stop_time = float(block.stop_time) - window_seconds
                    if stop_time < start_time:
                        continue
                    for window_start_time in np.arange(start_time, stop_time + 1e-9, step_time):
                        start_sample = int(np.searchsorted(timestamps, window_start_time, side="left"))
                        stop_sample = start_sample + window_samples
                        if stop_sample > data.shape[0]:
                            continue
                        candidates.append(
                            {
                                "start_sample": start_sample,
                                "stop_sample": stop_sample,
                                "window_start_time": float(timestamps[start_sample]),
                                "window_stop_time": float(timestamps[stop_sample - 1]),
                                "stimulus_block_index": int(block.block_index),
                                "stimulus_block_start_time": float(block.start_time),
                                "stimulus_block_stop_time": float(block.stop_time),
                                "stimulus_block_duration": float(block.duration),
                                "stimulus_presentations": int(block.n_presentations),
                            }
                        )
                if not candidates:
                    continue

                max_windows = max(1, max_rows_per_condition // max(1, len(channel_columns) * len(lfp_assets)))
                if len(candidates) > max_windows:
                    candidate_index = np.sort(rng.choice(np.arange(len(candidates)), size=max_windows, replace=False))
                    candidates = [candidates[index] for index in candidate_index]

                X_parts = []
                row_parts = []
                for window_index, candidate in enumerate(candidates):
                    segment = np.asarray(
                        data[candidate["start_sample"]:candidate["stop_sample"], channel_columns],
                        dtype=dtype,
                    ).T
                    X_parts.append(segment)
                    meta = selected_channels.copy()
                    meta["session_id"] = session_id
                    meta["probe_id"] = int(asset.probe_id)
                    meta["probe_name"] = getattr(asset, "name", "")
                    meta["lfp_path"] = str(lfp_path)
                    meta["condition"] = condition
                    meta["stimulus_name"] = condition
                    meta["window_index"] = int(window_index)
                    meta["window_start_sample"] = int(candidate["start_sample"])
                    meta["window_stop_sample"] = int(candidate["stop_sample"])
                    meta["window_start_time"] = candidate["window_start_time"]
                    meta["window_stop_time"] = candidate["window_stop_time"]
                    meta["window_seconds"] = window_samples / fs
                    meta["step_seconds"] = step_time
                    meta["fs"] = fs
                    for key in (
                        "stimulus_block_index",
                        "stimulus_block_start_time",
                        "stimulus_block_stop_time",
                        "stimulus_block_duration",
                        "stimulus_presentations",
                    ):
                        meta[key] = candidate[key]
                    row_parts.append(meta)

                blocks_by_condition[condition].append(np.vstack(X_parts))
                rows_by_condition[condition].append(pd.concat(row_parts, ignore_index=True))

    if not fs_seen:
        raise RuntimeError("No stimulus-locked LFP matrices could be built.")
    fs = float(np.median(fs_seen))
    matrices = {}
    for condition in stimulus_names:
        parts = blocks_by_condition.get(condition, [])
        if not parts:
            continue
        X = np.vstack(parts).astype(dtype, copy=False)
        row_table = pd.concat(rows_by_condition[condition], ignore_index=True)
        row_table["matrix_row"] = np.arange(len(row_table))
        matrices[condition] = SignalMatrix(X, row_table, fs=fs, condition=condition)
    if not matrices:
        raise RuntimeError("No stimulus-locked LFP matrices could be built for the requested stimuli.")
    return matrices


def summarize_lfp_matrices(matrices: dict[str, SignalMatrix]) -> pd.DataFrame:
    """Compact matrix/provenance summary."""

    rows = []
    for condition, matrix in matrices.items():
        meta = matrix.rows
        rows.append(
            {
                "condition": condition,
                "rows": matrix.X.shape[0],
                "columns": matrix.X.shape[1],
                "seconds": matrix.X.shape[1] / matrix.fs,
                "fs": matrix.fs,
                "sessions": meta["session_id"].nunique(),
                "probes": meta["probe_id"].nunique(),
                "regions": ", ".join(sorted(meta["region"].dropna().astype(str).unique())),
                "channels": meta["channel_id"].nunique(),
                "windows": meta[["session_id", "probe_id", "window_index"]].drop_duplicates().shape[0],
            }
        )
    return pd.DataFrame(rows)


def _find_lfp_data_path(h5: h5py.File) -> str:
    candidates = []
    for key in h5.get("acquisition", {}):
        path = f"acquisition/{key}"
        group = h5[path]
        if isinstance(group, h5py.Group) and "data" in group and "timestamps" in group:
            candidates.append(f"{path}/data")
        if isinstance(group, h5py.Group):
            for subkey in group:
                subpath = f"{path}/{subkey}"
                subgroup = h5[subpath]
                if isinstance(subgroup, h5py.Group) and "data" in subgroup and "timestamps" in subgroup:
                    candidates.append(f"{subpath}/data")
    if not candidates:
        raise RuntimeError("Could not find LFP data/timestamps in NWB acquisition group.")
    candidates = sorted(candidates, key=len)
    return candidates[0]


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
