"""Hippocampal theta loading helpers."""

from __future__ import annotations

import re
from math import gcd
from pathlib import Path
from xml.etree import ElementTree

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from urllib.request import urlretrieve

from .data import SignalMatrix, standardize_rows


COOLING_STATES = {
    0: "Cooling off",
    1: "Pre-Cooling",
    2: "Cooling on",
    3: "Post-Cooling",
}
TRIAL_DIRECTIONS = {1: "Left", 2: "Right"}
DANDI_001607_ASSET_ID = "a2e0ec66-b963-4640-a785-ec82eb8ca3a8"
DANDI_001607_DOWNLOAD_URL = f"https://api.dandiarchive.org/api/assets/{DANDI_001607_ASSET_ID}/download/"
DANDI_ASSET_DOWNLOAD_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"

BAD_SESSION_PATTERNS = (
    "!!!",
    "amp switched",
    "artifact",
    "too few",
    "few trials",
    "not good",
    "no temperature",
    "pure noise",
    "noisy",
    "weak positive ripple",
    "electrodes position too high",
    "low temp",
)
GOOD_REGION_PATTERNS = (
    "sharp wave reversal clear",
    "very good",
    "good",
    "ok",
)


def discover_crcns_lfp_sessions(
    crcns_root: str | Path,
    sheet_path: str | Path,
    processed_root: str | Path,
    *,
    fs: float = 1250.0,
    exclude_bad_notes: bool = True,
    require_good_region_note: bool = True,
) -> pd.DataFrame:
    """Find local raw ``.lfp`` sessions and their theta-shank channels."""

    crcns_root = Path(crcns_root)
    sheet = pd.read_csv(sheet_path)
    sheet = sheet[pd.to_numeric(sheet["ID"], errors="coerce").notna()].copy()
    sheet["ID"] = pd.to_numeric(sheet["ID"]).astype(int)

    records = []
    for lfp_path in sorted(crcns_root.rglob("*.lfp")):
        session = lfp_path.stem
        matches = sheet[sheet["Session name"].eq(session)]
        if matches.empty:
            continue
        session_dir = lfp_path.parent
        xml_path = session_dir / f"{session}.xml"
        trials_path = session_dir / f"{session}.trials.behavior.mat"
        behavior_path = session_dir / f"{session}.animal.behavior.mat"
        if not (xml_path.exists() and trials_path.exists() and behavior_path.exists()):
            continue

        group_name, reference_id = _theta_shank(_processed_nwb_for(processed_root, session))
        raw_channels = _xml_channels_on_shank(xml_path, group_name)
        row = matches.iloc[0]
        if exclude_bad_notes and _has_bad_session_note(row):
            continue
        region_text = str(row.get("GammaThetaPhaseCoupling Notes", "")).lower()
        if require_good_region_note and not any(pattern in region_text for pattern in GOOD_REGION_PATTERNS):
            continue
        xml = ElementTree.parse(xml_path).getroot()
        n_channels = int(xml.findtext("acquisitionSystem/nChannels"))
        lfp_fs = float(xml.findtext("fieldPotentials/lfpSamplingRate"))
        n_frames = lfp_path.stat().st_size // (np.dtype("int16").itemsize * n_channels)

        with h5py.File(trials_path, "r") as trials_file, h5py.File(behavior_path, "r") as behavior_file:
            last_trial_index = int(trials_file["trials/end"][:, 0].max()) - 1
            last_trial_time = float(behavior_file["animal/time"][last_trial_index, 0])
        if lfp_fs != fs or n_frames / fs < last_trial_time:
            raise ValueError(f"Incomplete or unexpected LFP file: {lfp_path}")

        records.append(
            {
                "animal": row["Animal"],
                "session": session,
                "sheet_id": int(row["ID"]),
                "manipulation": row["Manipulation"],
                "arena": row["Arena"],
                "brain_region": row["Brain regions"],
                "session_notes": row["Notes"],
                "lfp_path": lfp_path,
                "trials_path": trials_path,
                "behavior_path": behavior_path,
                "n_channels": n_channels,
                "n_frames": n_frames,
                "fs": lfp_fs,
                "theta_reference_id": reference_id,
                "theta_shank": group_name,
                "raw_channels": raw_channels,
            }
        )

    return pd.DataFrame(records)


def load_or_build_crcns_raw_cache(
    session_table: pd.DataFrame,
    cache_h5: str | Path,
    cache_trials_csv: str | Path,
    *,
    fs: float = 1250.0,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Create/read a window-invariant raw trial cache from local ``.lfp`` files."""

    cache_h5 = Path(cache_h5)
    cache_trials_csv = Path(cache_trials_csv)
    discovered_sessions = set(session_table["session"])
    needs_rebuild = rebuild or not (cache_h5.exists() and cache_trials_csv.exists())
    if not needs_rebuild:
        with h5py.File(cache_h5, "r") as cache:
            cached_sessions = set(cache.keys())
        needs_rebuild = discovered_sessions - cached_sessions

    if needs_rebuild:
        _build_crcns_raw_cache(session_table, cache_h5, cache_trials_csv, fs=fs)

    trial_index = pd.read_csv(cache_trials_csv)
    trial_index = trial_index[trial_index["session"].isin(discovered_sessions)].reset_index(drop=True)
    with h5py.File(cache_h5, "r") as cache:
        if float(cache.attrs["fs"]) != fs:
            raise ValueError(f"Cache fs={cache.attrs['fs']} does not match requested fs={fs}.")
        missing = set(trial_index["session"].unique()) - set(cache.keys())
        if missing:
            raise ValueError(f"Cache is missing sessions: {sorted(missing)}")
    return trial_index


def load_crcns_condition_matrices(
    cache_h5: str | Path,
    trial_index: pd.DataFrame,
    *,
    conditions: dict[str, str],
    window_seconds: float,
    step_seconds: float | None = None,
    fs: float = 1250.0,
    manipulation: str | None = "Cooling",
    arena_contains: str | None = "Theta maze",
) -> dict[str, SignalMatrix]:
    """Materialize one signal matrix per cooling-state condition."""

    matrices = {}
    for condition, cooling_state in conditions.items():
        criteria = {"cooling_state": cooling_state}
        if manipulation is not None:
            criteria["manipulation"] = manipulation
        rows = _select_trials(trial_index, **criteria)
        if arena_contains is not None:
            rows = rows[rows["arena"].fillna("").str.contains(arena_contains, case=False, regex=False)]
        matrices[condition] = _materialize_windows(
            cache_h5,
            rows,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            fs=fs,
            condition=condition,
        )
    return matrices


def summarize_signal_matrices(matrices: dict[str, SignalMatrix]) -> pd.DataFrame:
    """Return a compact inventory for condition matrices."""

    rows = []
    for condition, matrix in matrices.items():
        rows.append(
            {
                "condition": condition,
                "shape": matrix.X.shape,
                "rats": matrix.rows["animal"].nunique(),
                "sessions": matrix.rows["session"].nunique(),
                "session_electrodes": matrix.rows[["session", "raw_channel"]].drop_duplicates().shape[0],
                "physical_windows": matrix.rows[["session", "trial_id", "window_in_trial"]].drop_duplicates().shape[0],
                "errors_retained": int(matrix.rows["error"].sum()),
            }
        )
    return pd.DataFrame(rows)


def combine_condition_matrices(*matrix_sets: dict[str, SignalMatrix]) -> dict[str, SignalMatrix]:
    """Concatenate condition matrices that already share sampling and columns."""

    combined = {}
    conditions = sorted({condition for matrices in matrix_sets for condition in matrices})
    for condition in conditions:
        matrices = [matrices[condition] for matrices in matrix_sets if condition in matrices]
        if not matrices:
            continue
        fs_values = {float(matrix.fs) for matrix in matrices}
        n_cols = {matrix.X.shape[1] for matrix in matrices}
        if len(fs_values) != 1:
            raise ValueError(f"Cannot combine {condition!r}; mixed sampling rates {sorted(fs_values)}.")
        if len(n_cols) != 1:
            raise ValueError(f"Cannot combine {condition!r}; mixed column counts {sorted(n_cols)}.")
        combined[condition] = SignalMatrix(
            np.vstack([matrix.X for matrix in matrices]),
            pd.concat([matrix.rows for matrix in matrices], ignore_index=True),
            matrices[0].fs,
            condition,
        )
    return combined


def dandi_000059_candidate_sessions(
    asset_manifest: str | Path,
    sheet_path: str | Path,
    *,
    min_trials: int = 100,
    manipulation: str = "Cooling",
    arena_contains: str = "Theta maze",
    require_public: bool = True,
    require_processed: bool = True,
    require_behavior_lfp_temp: bool = True,
    require_good_region_note: bool = True,
) -> pd.DataFrame:
    """Return DANDI 000059 raw NWB sessions annotated with cleaning decisions."""

    import json

    asset_manifest = Path(asset_manifest)
    sheet = pd.read_csv(sheet_path)
    assets = json.loads(asset_manifest.read_text())
    asset_rows = []
    for asset in assets:
        path = asset.get("path", "")
        session = _session_from_dandi_path(path)
        asset_rows.append(
            {
                "session": session,
                "animal": _animal_from_dandi_path(path),
                "path": path,
                "asset_id": asset.get("asset_id"),
                "size_gb": asset.get("size", 0) / 1e9,
                "asset_kind": "raw" if "desc-raw_ecephys" in path else "processed" if "desc-processed_behavior+ecephys" in path else "other",
            }
        )
    assets_df = pd.DataFrame(asset_rows)
    raw = assets_df[assets_df["asset_kind"].eq("raw")].rename(
        columns={"asset_id": "raw_asset_id", "path": "raw_path", "size_gb": "raw_size_gb"}
    )
    processed = assets_df[assets_df["asset_kind"].eq("processed")].rename(
        columns={"asset_id": "processed_asset_id", "path": "processed_path", "size_gb": "processed_size_gb"}
    )
    table = raw.merge(
        processed[["session", "processed_asset_id", "processed_path", "processed_size_gb"]],
        on="session",
        how="left",
    )
    sheet_columns = [
        "Animal",
        "Session name",
        "Manipulation",
        "Arena",
        "Trial count",
        "Public",
        "Processed",
        "Behavior/LFP/Temp analysis",
        "Brain regions",
        "Notes",
        "GammaThetaPhaseCoupling Channel,Shank",
        "GammaThetaPhaseCoupling Notes",
    ]
    table = table.merge(sheet[sheet_columns], left_on="session", right_on="Session name", how="left")
    table["animal"] = table["animal"].fillna(table["Animal"])
    table["exclude_reason"] = table.apply(
        _dandi_candidate_exclude_reason,
        axis=1,
        min_trials=min_trials,
        manipulation=manipulation,
        arena_contains=arena_contains,
        require_public=require_public,
        require_processed=require_processed,
        require_behavior_lfp_temp=require_behavior_lfp_temp,
        require_good_region_note=require_good_region_note,
    )
    table["include"] = table["exclude_reason"].eq("")
    return table.sort_values(["include", "raw_size_gb"], ascending=[False, True]).reset_index(drop=True)


def load_or_build_dandi_downsampled_lfp_cache(
    candidate_table: pd.DataFrame,
    cache_h5: str | Path,
    cache_trials_csv: str | Path,
    processed_root: str | Path,
    raw_download_dir: str | Path,
    *,
    target_fs: float = 1250.0,
    max_assets: int | None = None,
    max_total_gb: float | None = None,
    download_missing: bool = False,
    rebuild: bool = False,
    delete_raw_after_downsample: bool = True,
) -> pd.DataFrame:
    """Build/read a compact DANDI 000059 cache from raw NWBs.

    The raw NWB is only a temporary source: selected theta-region electrodes
    are trialized, downsampled to ``target_fs``, written to ``cache_h5``, and
    optionally deleted immediately.
    """

    cache_h5 = Path(cache_h5)
    cache_trials_csv = Path(cache_trials_csv)
    processed_root = Path(processed_root)
    raw_download_dir = Path(raw_download_dir)
    allowed_sessions = set(candidate_table.loc[candidate_table["include"], "session"].dropna())
    selected = candidate_table[candidate_table["include"]].sort_values("raw_size_gb").copy()
    if max_total_gb is not None:
        selected = selected[selected["raw_size_gb"].cumsum() <= max_total_gb]
    if max_assets is not None:
        selected = selected.head(max_assets)

    if selected.empty:
        existing_trials = _read_existing_trial_index(cache_h5, cache_trials_csv, target_fs)
        if not existing_trials.empty:
            existing_trials = existing_trials[existing_trials["session"].isin(allowed_sessions)].reset_index(drop=True)
        return existing_trials

    cache_h5.parent.mkdir(parents=True, exist_ok=True)
    existing_trials = _read_existing_trial_index(cache_h5, cache_trials_csv, target_fs)
    if not existing_trials.empty:
        existing_trials = existing_trials[existing_trials["session"].isin(allowed_sessions)].reset_index(drop=True)
    completed_sessions = set(existing_trials["session"]) if not existing_trials.empty else set()

    trial_tables = [existing_trials[~existing_trials["session"].isin(set(selected["session"]))]] if not existing_trials.empty else []
    with h5py.File(cache_h5, "a") as cache:
        if "fs" in cache.attrs and float(cache.attrs["fs"]) != target_fs:
            raise ValueError(f"Cache fs={cache.attrs['fs']} does not match target_fs={target_fs}.")
        cache.attrs["fs"] = target_fs
        cache.attrs["units"] = "downsampled selected-electrode raw ephys"
        for spec in selected.to_dict("records"):
            session = spec["session"]
            if session in completed_sessions and not rebuild and session in cache:
                trial_tables.append(existing_trials[existing_trials["session"].eq(session)])
                continue
            if not download_missing:
                continue

            processed_path = processed_root / spec["processed_path"]
            raw_path = raw_download_dir / spec["raw_path"]
            _download_dandi_asset(spec["processed_asset_id"], processed_path)
            _download_dandi_asset(spec["raw_asset_id"], raw_path)

            if session in cache:
                del cache[session]
            session_trials = _append_dandi_raw_session_to_cache(
                cache,
                raw_path,
                processed_path,
                spec,
                target_fs=target_fs,
            )
            trial_tables.append(session_trials)
            if delete_raw_after_downsample and raw_path.exists():
                raw_path.unlink()

    if trial_tables:
        trial_index = pd.concat(trial_tables, ignore_index=True)
        trial_index.to_csv(cache_trials_csv, index=False)
        return trial_index
    return _read_existing_trial_index(cache_h5, cache_trials_csv, target_fs)


def _session_from_dandi_path(path: str) -> str | None:
    match = re.search(r"ses-([^/]+?)_desc", path)
    return match.group(1).replace("-", "_") if match else None


def _animal_from_dandi_path(path: str) -> str | None:
    match = re.search(r"sub-([^/]+)", path)
    return match.group(1) if match else None


def _dandi_candidate_exclude_reason(
    row: pd.Series,
    *,
    min_trials: int,
    manipulation: str,
    arena_contains: str,
    require_public: bool,
    require_processed: bool,
    require_behavior_lfp_temp: bool,
    require_good_region_note: bool,
) -> str:
    reasons = []
    if pd.isna(row.get("Session name")):
        reasons.append("not in spreadsheet")
    if pd.isna(row.get("processed_asset_id")):
        reasons.append("missing processed companion")
    if manipulation and row.get("Manipulation") != manipulation:
        reasons.append(f"not {manipulation}")
    arena = str(row.get("Arena", ""))
    if arena_contains and arena_contains.lower() not in arena.lower():
        reasons.append(f"arena not {arena_contains}")
    if require_public and str(row.get("Public", "")).lower() != "yes":
        reasons.append("not public")
    if require_processed and str(row.get("Processed", "")).lower() != "yes":
        reasons.append("not processed")
    if require_behavior_lfp_temp and str(row.get("Behavior/LFP/Temp analysis", "")).lower() != "yes":
        reasons.append("behavior/lfp/temp not yes")
    trial_count = pd.to_numeric(row.get("Trial count"), errors="coerce")
    if not np.isfinite(trial_count) or trial_count < min_trials:
        reasons.append(f"< {min_trials} trials")

    bad_note = _matched_bad_session_note(row)
    if bad_note:
        reasons.append(bad_note)
    region_text = str(row.get("GammaThetaPhaseCoupling Notes", "")).lower()
    if require_good_region_note and not any(pattern in region_text for pattern in GOOD_REGION_PATTERNS):
        reasons.append("no positive region-quality note")

    return "; ".join(dict.fromkeys(reasons))


def _has_bad_session_note(row: pd.Series) -> bool:
    return bool(_matched_bad_session_note(row))


def _matched_bad_session_note(row: pd.Series) -> str | None:
    text = " ".join(
        str(row.get(column, ""))
        for column in [
            "Notes",
            "Behavior/LFP/Temp analysis",
            "GammaThetaPhaseCoupling Notes",
        ]
    ).lower()
    for pattern in BAD_SESSION_PATTERNS:
        if pattern in text:
            return pattern
    return None


def _read_existing_trial_index(cache_h5: Path, cache_trials_csv: Path, target_fs: float) -> pd.DataFrame:
    if not (cache_h5.exists() and cache_trials_csv.exists()):
        return pd.DataFrame()
    with h5py.File(cache_h5, "r") as cache:
        if "fs" in cache.attrs and float(cache.attrs["fs"]) != target_fs:
            raise ValueError(f"Cache fs={cache.attrs['fs']} does not match target_fs={target_fs}.")
    return pd.read_csv(cache_trials_csv)


def _download_dandi_asset(asset_id: str, output_path: Path) -> None:
    if output_path.exists():
        return
    if not asset_id or pd.isna(asset_id):
        raise ValueError(f"Missing DANDI asset id for {output_path}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    urlretrieve(DANDI_ASSET_DOWNLOAD_URL.format(asset_id=asset_id), tmp_path)
    tmp_path.replace(output_path)


def _append_dandi_raw_session_to_cache(
    cache: h5py.File,
    raw_nwb_path: Path,
    processed_nwb_path: Path,
    spec: dict,
    *,
    target_fs: float,
) -> pd.DataFrame:
    trials = _processed_nwb_trials(processed_nwb_path, spec)
    if trials.empty:
        raise RuntimeError(f"No trials found in {processed_nwb_path}.")
    selected_electrodes, theta_shank, theta_reference_id = _processed_theta_electrodes(processed_nwb_path)
    selected_electrodes = set(int(value) for value in selected_electrodes)

    trial_blocks = []
    trial_rows = []
    cache_offset = 0
    with h5py.File(raw_nwb_path, "r") as raw_nwb:
        series = _find_raw_electrical_series(raw_nwb)
        source_fs, source_start_time = _series_rate_and_start(series)
        electrode_ids = _series_electrode_ids(raw_nwb, series)
        selected_positions = [idx for idx, electrode_id in enumerate(electrode_ids) if int(electrode_id) in selected_electrodes]
        if not selected_positions:
            raise RuntimeError(
                f"No raw ElectricalSeries electrodes match theta shank {theta_shank} in {raw_nwb_path.name}."
            )
        selected_ids = [int(electrode_ids[idx]) for idx in selected_positions]
        data = series["data"]
        channels_first = data.shape[0] == len(electrode_ids)
        n_time = data.shape[1] if channels_first else data.shape[0]
        source_duration = n_time / source_fs
        fallback_offset = 0.0
        if trials["trial_stop_s"].max() - source_start_time > source_duration and source_start_time == 0:
            fallback_offset = float(trials["trial_start_s"].min())

        for trial in trials.itertuples(index=False):
            start_time = float(trial.trial_start_s) - fallback_offset
            stop_time = float(trial.trial_stop_s) - fallback_offset
            first_sample = max(0, int(round((start_time - source_start_time) * source_fs)))
            last_sample = min(n_time, int(round((stop_time - source_start_time) * source_fs)))
            if last_sample <= first_sample:
                continue
            block = _read_series_block(data, first_sample, last_sample, selected_positions, channels_first=channels_first)
            block = _resample_block(block, source_fs, target_fs)
            if block.shape[0] == 0:
                continue
            trial_blocks.append(block.astype(np.float32, copy=False))
            trial_rows.append(
                {
                    **trial._asdict(),
                    "cache_start": cache_offset,
                    "cache_stop": cache_offset + block.shape[0],
                    "n_trial_samples": block.shape[0],
                    "fs": target_fs,
                    "source_fs": source_fs,
                    "units": "downsampled raw ephys",
                    "theta_shank": theta_shank,
                    "theta_reference_id": theta_reference_id,
                    "raw_channels": selected_ids,
                }
            )
            cache_offset += block.shape[0]

    if not trial_blocks:
        raise RuntimeError(f"No downsampled trial blocks were extracted for {spec['session']}.")
    samples = np.concatenate(trial_blocks, axis=0)
    session_group = cache.create_group(spec["session"])
    session_group.create_dataset("samples", data=samples, dtype="float32", compression="gzip", compression_opts=4)
    session_group.create_dataset("raw_channels", data=np.asarray(selected_ids, dtype=int))
    session_group.attrs["source"] = "DANDI:000059 raw NWB downsampled cache"
    session_group.attrs["source_fs"] = source_fs
    session_group.attrs["target_fs"] = target_fs
    session_group.attrs["theta_shank"] = theta_shank
    return pd.DataFrame(trial_rows)


def _processed_nwb_trials(processed_nwb_path: Path, spec: dict) -> pd.DataFrame:
    with h5py.File(processed_nwb_path, "r") as nwb:
        trials = nwb["intervals/trials"]
        start = np.asarray(trials["start_time"][:], dtype=float)
        stop = np.asarray(trials["stop_time"][:], dtype=float)
        ids = np.asarray(trials["id"][:], dtype=int) if "id" in trials else np.arange(len(start), dtype=int)
        cooling = _decode_h5_array(trials["cooling state"][:]) if "cooling state" in trials else np.array([""] * len(start))
        direction = _decode_h5_array(trials["condition"][:]) if "condition" in trials else np.array(["Unknown"] * len(start))
        error = np.asarray(trials["error"][:], dtype=bool) if "error" in trials else np.zeros(len(start), dtype=bool)

    rows = []
    for trial_id, trial_start, trial_stop, cooling_state, trial_direction, is_error in zip(
        ids,
        start,
        stop,
        cooling,
        direction,
        error,
    ):
        rows.append(
            {
                "animal": spec.get("animal"),
                "session": spec.get("session"),
                "sheet_id": np.nan,
                "manipulation": spec.get("Manipulation"),
                "arena": spec.get("Arena"),
                "brain_region": spec.get("Brain regions"),
                "session_notes": spec.get("Notes"),
                "trial_id": int(trial_id) + 1,
                "trial_direction": str(trial_direction),
                "cooling_state": str(cooling_state),
                "error": bool(is_error),
                "trial_start_s": float(trial_start),
                "trial_stop_s": float(trial_stop),
                "dandi_raw_asset_id": spec.get("raw_asset_id"),
                "dandi_processed_asset_id": spec.get("processed_asset_id"),
                "dandi_raw_path": spec.get("raw_path"),
                "dandi_processed_path": spec.get("processed_path"),
            }
        )
    return pd.DataFrame(rows)


def _processed_theta_electrodes(processed_nwb_path: Path) -> tuple[list[int], str, int]:
    theta_shank, theta_reference_id = _theta_shank(processed_nwb_path)
    with h5py.File(processed_nwb_path, "r") as nwb:
        electrodes = nwb["general/extracellular_ephys/electrodes"]
        ids = np.asarray(electrodes["id"][:], dtype=int)
        group_names = _decode_h5_array(electrodes["group_name"][:])
        selected = ids[np.asarray(group_names) == theta_shank]
    return selected.astype(int).tolist(), theta_shank, theta_reference_id


def _find_raw_electrical_series(nwb: h5py.File) -> h5py.Group:
    candidates = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Group) or "data" not in obj:
            return
        data = obj["data"]
        if not isinstance(data, h5py.Dataset) or data.ndim != 2:
            return
        neurodata_type = obj.attrs.get("neurodata_type", b"")
        if isinstance(neurodata_type, bytes):
            neurodata_type = neurodata_type.decode()
        if "ElectricalSeries" in str(neurodata_type) or "ecephys" in name.lower() or "electrical" in name.lower():
            candidates.append((data.size, name))

    nwb.visititems(visit)
    if not candidates:
        raise RuntimeError("Could not find a 2D ElectricalSeries-like dataset in raw NWB.")
    _, name = max(candidates)
    return nwb[name]


def _series_rate_and_start(series: h5py.Group) -> tuple[float, float]:
    if "starting_time" in series:
        starting_time = series["starting_time"]
        rate = starting_time.attrs.get("rate")
        if rate is not None:
            return float(rate), float(starting_time[()])
    if "timestamps" in series:
        timestamps = np.asarray(series["timestamps"][: min(10_000, len(series["timestamps"]))], dtype=float)
        return float(1 / np.nanmedian(np.diff(timestamps))), float(timestamps[0])
    raise RuntimeError(f"ElectricalSeries {series.name} has neither starting_time/rate nor timestamps.")


def _series_electrode_ids(nwb: h5py.File, series: h5py.Group) -> np.ndarray:
    electrodes = _nwb_electrode_table(nwb)
    if "electrodes" not in series:
        return electrodes["id"].to_numpy(dtype=int)
    region = np.asarray(series["electrodes"][:], dtype=int)
    if len(region) and region.max() < len(electrodes):
        return electrodes.iloc[region]["id"].to_numpy(dtype=int)
    if "id" in electrodes.columns:
        known_ids = set(electrodes["id"].astype(int))
        if set(region.astype(int)).issubset(known_ids):
            return region.astype(int)
    return region.astype(int)


def _nwb_electrode_table(nwb: h5py.File) -> pd.DataFrame:
    group = nwb["general/extracellular_ephys/electrodes"]
    table = {}
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 1 and obj.shape[0] == group["id"].shape[0]:
            try:
                table[name] = _decode_h5_array(obj[:])
            except TypeError:
                pass
    return pd.DataFrame(table)


def _read_series_block(
    data: h5py.Dataset,
    first_sample: int,
    last_sample: int,
    selected_positions: list[int],
    *,
    channels_first: bool,
) -> np.ndarray:
    positions = np.asarray(selected_positions, dtype=int)
    if channels_first:
        return np.asarray(data[positions, first_sample:last_sample], dtype=np.float32).T
    return np.asarray(data[first_sample:last_sample, positions], dtype=np.float32)


def _resample_block(block: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
    if np.isclose(source_fs, target_fs):
        return block
    source_i = int(round(source_fs))
    target_i = int(round(target_fs))
    if not np.isclose(source_fs, source_i) or not np.isclose(target_fs, target_i):
        ratio = target_fs / source_fs
        target_n = int(round(block.shape[0] * ratio))
        if target_n <= 0:
            return np.empty((0, block.shape[1]), dtype=np.float32)
        from scipy.signal import resample

        return resample(block, target_n, axis=0).astype(np.float32, copy=False)
    common = gcd(source_i, target_i)
    up = target_i // common
    down = source_i // common
    return resample_poly(block, up=up, down=down, axis=0).astype(np.float32, copy=False)


def load_dandi_001607_theta_matrices(
    nwb_path: str | Path,
    *,
    channels: str | list[int] | tuple[int, ...] = (0,),
    condition_blocks: dict[str, tuple[float, float]] | None = None,
    window_seconds: float = 2.0,
    step_seconds: float | None = None,
    standardize: bool = False,
    max_windows_per_condition: int | None = None,
    cv_chunk_windows: int = 20,
    download_if_missing: bool = False,
) -> tuple[dict[str, SignalMatrix], pd.DataFrame]:
    """Load the small DANDI 001607 ecephys NWB used in notebook 06.

    The distributed NWB has one continuous signal object at
    ``scratch/filtered data/data``. Rows are windows crossed with selected
    electrodes. ``condition_blocks`` names contiguous fractions of that same
    trace; these are analysis blocks, not independent experimental conditions.
    """

    nwb_path = Path(nwb_path)
    if not nwb_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"{nwb_path} does not exist. Set download_if_missing=True to fetch "
                f"DANDI asset {DANDI_001607_ASSET_ID}."
            )
        nwb_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DANDI_001607_DOWNLOAD_URL, nwb_path)

    condition_blocks = condition_blocks or {"lineartrack_1pulse": (0.0, 1.0)}
    step_seconds = window_seconds if step_seconds is None else step_seconds
    if cv_chunk_windows < 1:
        raise ValueError("cv_chunk_windows must be positive.")

    matrices: dict[str, SignalMatrix] = {}
    with h5py.File(nwb_path, "r") as nwb:
        data_dset = nwb["scratch/filtered data/data"]
        timestamps = np.asarray(nwb["scratch/filtered data/timestamps"], dtype=float)
        fs = float(1 / np.nanmedian(np.diff(timestamps)))
        n_samples, n_available_channels = data_dset.shape
        channel_indices = _resolve_dandi_channels(channels, n_available_channels)
        electrodes = _dandi_electrode_table(nwb)

        subject = _decode_h5_scalar(nwb["general/subject/subject_id"][()])
        species = _decode_h5_scalar(nwb["general/subject/species"][()])
        session_id = _decode_h5_scalar(nwb["general/session_id"][()])
        session_description = _decode_h5_scalar(nwb["session_description"][()])
        epoch_tag = _active_dandi_epoch_tag(nwb, timestamps)

        summary = pd.DataFrame(
            [
                {
                    "dandiset": "DANDI:001607",
                    "asset_id": DANDI_001607_ASSET_ID,
                    "nwb_path": str(nwb_path),
                    "subject": subject,
                    "species": species,
                    "session_id": session_id,
                    "session_description": session_description,
                    "available_channels": n_available_channels,
                    "selected_channels": channel_indices,
                    "fs": fs,
                    "samples": n_samples,
                    "seconds": n_samples / fs,
                    "active_epoch_tag": epoch_tag,
                    "condition_note": "condition labels are contiguous time blocks within the one available scratch trace",
                }
            ]
        )

        window_samples = int(round(window_seconds * fs))
        step_samples = int(round(step_seconds * fs))
        if window_samples <= 0 or step_samples <= 0:
            raise ValueError("window_seconds and step_seconds must be positive.")

        for condition, (start_frac, stop_frac) in condition_blocks.items():
            if not (0 <= start_frac < stop_frac <= 1):
                raise ValueError(f"Invalid condition block for {condition!r}: {(start_frac, stop_frac)}")
            start_sample = int(round(start_frac * n_samples))
            stop_sample = int(round(stop_frac * n_samples))
            starts = np.arange(start_sample, stop_sample - window_samples + 1, step_samples, dtype=int)
            if max_windows_per_condition is not None and len(starts) > max_windows_per_condition:
                keep = np.linspace(0, len(starts) - 1, max_windows_per_condition, dtype=int)
                starts = starts[keep]
            if len(starts) == 0:
                raise RuntimeError(f"No complete windows for condition {condition!r}.")

            block = np.asarray(data_dset[start_sample:stop_sample, :], dtype=np.float32)
            block = block[:, channel_indices]
            signal_blocks = []
            rows = []
            for window_index, global_start in enumerate(starts):
                local_start = int(global_start - start_sample)
                local_stop = local_start + window_samples
                signal_blocks.append(block[local_start:local_stop].T)
                for channel_position, channel_id in enumerate(channel_indices):
                    electrode = electrodes.loc[channel_id].to_dict() if channel_id in electrodes.index else {}
                    rows.append(
                        {
                            "animal": subject,
                            "subject": subject,
                            "species": species,
                            "session": session_id,
                            "session_id": session_id,
                            "session_description": session_description,
                            "dandiset": "DANDI:001607",
                            "asset_id": DANDI_001607_ASSET_ID,
                            "condition": condition,
                            "time_block": condition,
                            "dandi_epoch_tag": epoch_tag,
                            "trial_id": int(window_index),
                            "window_index": int(window_index),
                            "window_in_trial": 0,
                            "window_chunk": f"{condition}_chunk_{window_index // cv_chunk_windows:03d}",
                            "window_start_s": float((timestamps[global_start] - timestamps[0])),
                            "window_stop_s": float((timestamps[global_start + window_samples - 1] - timestamps[0])),
                            "window_start_time": float(timestamps[global_start]),
                            "window_stop_time": float(timestamps[global_start + window_samples - 1]),
                            "window_seconds": window_samples / fs,
                            "step_seconds": step_samples / fs,
                            "raw_channel": int(channel_id),
                            "channel_id": int(channel_id),
                            "channel_position": int(channel_position),
                            "source_matrix_row": int(window_index * len(channel_indices) + channel_position),
                            "error": False,
                            **electrode,
                        }
                    )

            X = np.concatenate(signal_blocks, axis=0)
            if standardize:
                X = standardize_rows(X)
            matrices[condition] = SignalMatrix(X.astype(np.float32, copy=False), pd.DataFrame(rows), fs, condition)

    return matrices, summary


def _resolve_dandi_channels(channels: str | list[int] | tuple[int, ...], n_available_channels: int) -> list[int]:
    if channels == "all":
        return list(range(n_available_channels))
    channel_indices = [int(channel) for channel in channels]
    invalid = [channel for channel in channel_indices if channel < 0 or channel >= n_available_channels]
    if invalid:
        raise ValueError(f"Channels out of range [0, {n_available_channels - 1}]: {invalid}")
    return channel_indices


def _dandi_electrode_table(nwb: h5py.File) -> pd.DataFrame:
    group = nwb["general/extracellular_ephys/electrodes"]
    columns = [
        "id",
        "channel_id",
        "bad_channel",
        "group_name",
        "location",
        "probe_electrode",
        "probe_shank",
        "rel_x",
        "rel_y",
        "rel_z",
        "x",
        "y",
        "z",
        "imp",
        "ntrode_id",
    ]
    table = {}
    for column in columns:
        if column in group:
            table[column] = _decode_h5_array(group[column][:])
    electrodes = pd.DataFrame(table)
    if "channel_id" not in electrodes.columns:
        electrodes["channel_id"] = np.arange(len(electrodes), dtype=int)
    electrodes["channel_id"] = electrodes["channel_id"].astype(int)
    return electrodes.set_index("channel_id", drop=False)


def _active_dandi_epoch_tag(nwb: h5py.File, timestamps: np.ndarray) -> str:
    if "intervals" not in nwb or "epochs" not in nwb["intervals"]:
        return ""
    epochs = nwb["intervals/epochs"]
    starts = np.asarray(epochs["start_time"][:], dtype=float)
    stops = np.asarray(epochs["stop_time"][:], dtype=float)
    tags = _decode_h5_array(epochs["tags"][:]) if "tags" in epochs else np.array([""] * len(starts))
    midpoint = float(np.nanmedian(timestamps))
    matches = np.flatnonzero((starts <= midpoint) & (midpoint <= stops))
    if len(matches) == 0:
        return ""
    return str(tags[matches[-1]])


def _decode_h5_scalar(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


def _decode_h5_array(values):
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.asarray([item.decode() for item in values])
    if values.dtype.kind == "O":
        return np.asarray([item.decode() if isinstance(item, bytes) else item for item in values])
    return values


def _processed_nwb_for(processed_root: str | Path, session: str) -> Path:
    key = session.replace("_", "-")
    matches = list(Path(processed_root).rglob(f"*ses-{key}_desc-processed_behavior+ecephys.nwb"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one processed NWB for {session}; found {len(matches)}")
    return matches[0]


def _theta_shank(processed_nwb: Path) -> tuple[str, int]:
    with h5py.File(processed_nwb, "r") as nwb:
        electrodes = nwb["general/extracellular_ephys/electrodes"]
        is_reference = electrodes["theta_reference"][:].astype(bool)
        if is_reference.sum() != 1:
            raise ValueError(f"Expected one theta-reference electrode in {processed_nwb.name}")
        value = electrodes["group_name"][:][is_reference][0]
        group_name = value.decode() if isinstance(value, bytes) else str(value)
        reference_id = int(electrodes["id"][:][is_reference][0])
    return group_name, reference_id


def _xml_channels_on_shank(xml_path: Path, group_name: str) -> list[int]:
    shank_number = int(re.search(r"(\d+)$", group_name).group(1))
    xml = ElementTree.parse(xml_path).getroot()
    groups = xml.findall(".//anatomicalDescription/channelGroups/group")
    return [int(node.text) for node in groups[shank_number - 1].findall("channel")]


def _build_crcns_raw_cache(
    session_table: pd.DataFrame,
    cache_h5: Path,
    cache_trials_csv: Path,
    *,
    fs: float,
) -> None:
    cache_h5.parent.mkdir(parents=True, exist_ok=True)
    trial_metadata = []
    with h5py.File(cache_h5, "w") as cache:
        cache.attrs["fs"] = fs
        cache.attrs["units"] = "int16 ADC counts"
        for spec in session_table.to_dict("records"):
            with h5py.File(spec["trials_path"], "r") as trials_file, h5py.File(spec["behavior_path"], "r") as behavior_file:
                trials = trials_file["trials"]
                starts = trials["start"][:, 0].astype(int) - 1
                stops = trials["end"][:, 0].astype(int) - 1
                cooling_codes = trials["cooling"][:, 0].astype(int)
                directions = trials["stat"][:, 0].astype(int)
                error_trials = set(trials["error"][:, 0].astype(int))
                behavior_time = behavior_file["animal/time"][:, 0]

            lfp = np.memmap(
                spec["lfp_path"],
                dtype="int16",
                mode="r",
                shape=(int(spec["n_frames"]), int(spec["n_channels"])),
            )
            raw_channels = np.asarray(spec["raw_channels"], dtype=int)
            trial_blocks = []
            cache_offset = 0
            for trial_index, (start_index, stop_index) in enumerate(zip(starts, stops), start=1):
                trial_start = float(behavior_time[start_index])
                trial_stop = float(behavior_time[stop_index])
                first_sample = int(round(trial_start * fs))
                last_sample = int(round(trial_stop * fs))
                block = np.asarray(lfp[first_sample:last_sample, raw_channels], dtype=np.int16)
                trial_blocks.append(block)

                trial_metadata.append(
                    {
                        "animal": spec["animal"],
                        "session": spec["session"],
                        "sheet_id": spec["sheet_id"],
                        "manipulation": spec["manipulation"],
                        "arena": spec["arena"],
                        "brain_region": spec["brain_region"],
                        "session_notes": spec["session_notes"],
                        "theta_shank": spec["theta_shank"],
                        "theta_reference_id": spec["theta_reference_id"],
                        "trial_id": trial_index,
                        "trial_direction": TRIAL_DIRECTIONS.get(int(directions[trial_index - 1]), "Unknown"),
                        "cooling_code": int(cooling_codes[trial_index - 1]),
                        "cooling_state": COOLING_STATES[int(cooling_codes[trial_index - 1])],
                        "error": trial_index in error_trials,
                        "trial_start_s": trial_start,
                        "trial_stop_s": trial_stop,
                        "cache_start": cache_offset,
                        "cache_stop": cache_offset + len(block),
                        "n_trial_samples": len(block),
                        "fs": fs,
                        "units": "int16 ADC counts",
                    }
                )
                cache_offset += len(block)

            session_group = cache.create_group(spec["session"])
            session_group.create_dataset("samples", data=np.concatenate(trial_blocks, axis=0), dtype="int16")
            session_group.create_dataset("raw_channels", data=raw_channels)

    pd.DataFrame(trial_metadata).to_csv(cache_trials_csv, index=False)


def _select_trials(trials: pd.DataFrame, **criteria) -> pd.DataFrame:
    rows = trials.copy()
    for column, value in criteria.items():
        rows = rows[rows[column].eq(value)]
    return rows


def _materialize_windows(
    cache_h5: str | Path,
    trials: pd.DataFrame,
    *,
    window_seconds: float,
    step_seconds: float | None,
    fs: float,
    condition: str,
) -> SignalMatrix:
    step_seconds = window_seconds if step_seconds is None else step_seconds
    window_samples = int(round(window_seconds * fs))
    step_samples = int(round(step_seconds * fs))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Window and step must be positive.")

    signal_blocks = []
    metadata = []
    with h5py.File(cache_h5, "r") as cache:
        for trial in trials.itertuples(index=False):
            group = cache[trial.session]
            samples = group["samples"]
            raw_channels = group["raw_channels"][:].astype(int)
            starts = range(int(trial.cache_start), int(trial.cache_stop) - window_samples + 1, step_samples)
            for window_index, first in enumerate(starts):
                signal_blocks.append(samples[first:first + window_samples].T)
                window_start = float(trial.trial_start_s) + window_index * step_seconds
                for channel_position, raw_channel in enumerate(raw_channels):
                    metadata.append(
                        {
                            **trial._asdict(),
                            "raw_channel": int(raw_channel),
                            "channel_position_on_shank": channel_position,
                            "window_in_trial": window_index,
                            "window_start_s": window_start,
                            "window_stop_s": window_start + window_seconds,
                            "window_seconds": window_seconds,
                            "step_seconds": step_seconds,
                        }
                    )

    if not signal_blocks:
        raise RuntimeError(f"No windows materialized for {condition!r}.")
    X = np.concatenate(signal_blocks, axis=0).astype(np.int16, copy=False)
    rows = pd.DataFrame(metadata)
    if X.shape != (len(rows), window_samples):
        raise RuntimeError(f"Materialized matrix shape mismatch for {condition!r}.")
    return SignalMatrix(X, rows, fs, condition)
