"""OpenNeuro ds006897 rhythmic visual stimulation helpers."""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .data import SignalMatrix, standardize_rows


DATASET_ID = "ds006897"
S3_PREFIX = f"s3://openneuro.org/{DATASET_ID}"

DEFAULT_VISUAL_CHANNELS = (
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "O1",
    "Oz",
    "O2",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
)

DEFAULT_CONDITIONS = ("controle", "10hz", "20hz", "40hz")
METADATA_FILES = ("participants.tsv", "dataset_description.json", "README.md")
STIM_EVENT_BY_CONDITION = {
    "controle": "CNhz",
    "10hz": "10hz",
    "20hz": "20hz",
    "40hz": "40hz",
}


def normalize_subject(subject: str | int) -> str:
    """Return OpenNeuro subject ids as ``sub-012`` strings."""

    if isinstance(subject, int):
        return f"sub-{subject:03d}"
    subject = str(subject)
    return subject if subject.startswith("sub-") else f"sub-{int(subject):03d}"


def subject_eeg_stem(subject: str | int) -> str:
    """Return the BIDS stem for one ds006897 subject EEG recording."""

    subject = normalize_subject(subject)
    return f"{subject}/eeg/{subject}_task-eegsvr"


def resolve_ds006897_subjects(root: str | Path, subjects) -> list[str]:
    """Resolve ``subjects`` to concrete BIDS subject ids.

    Pass ``"all"`` to use every participant listed in ``participants.tsv``.
    """

    root = Path(root)
    if isinstance(subjects, str) and subjects.lower() == "all":
        participants = load_ds006897_participants(root)
        return participants["participant_id"].astype(str).tolist()
    if isinstance(subjects, str | int):
        return [normalize_subject(subjects)]
    return [normalize_subject(subject) for subject in subjects]


def download_ds006897_subjects(
    root: str | Path,
    subjects,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download selected raw BrainVision files from OpenNeuro's public S3 bucket.

    This fetches only the files needed to read each subject with MNE. It does not
    download the full 26 GB dataset unless ``subjects`` contains every subject.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for relative in METADATA_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            status = "cached"
        else:
            cmd = ["aws", "s3", "cp", "--no-sign-request", f"{S3_PREFIX}/{relative}", str(target)]
            subprocess.run(cmd, check=True)
            status = "downloaded"
        rows.append({"path": relative, "local_path": str(target), "status": status})

    subjects = resolve_ds006897_subjects(root, subjects)
    files = []
    for subject in subjects:
        stem = subject_eeg_stem(subject)
        files.extend(
            [
                f"{stem}_eeg.eeg",
                f"{stem}_eeg.vhdr",
                f"{stem}_eeg.vmrk",
                f"{stem}_eeg.json",
                f"{stem}_channels.tsv",
                f"{stem}_events.tsv",
            ]
        )

    for relative in files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            status = "cached"
        else:
            cmd = ["aws", "s3", "cp", "--no-sign-request", f"{S3_PREFIX}/{relative}", str(target)]
            subprocess.run(cmd, check=True)
            status = "downloaded"
        rows.append({"path": relative, "local_path": str(target), "status": status})

    return pd.DataFrame(rows)


def load_ds006897_participants(root: str | Path) -> pd.DataFrame:
    """Load participant metadata if it has already been downloaded."""

    root = Path(root)
    return pd.read_csv(root / "participants.tsv", sep="\t")


def load_ds006897_raw(
    root: str | Path,
    subject: str | int,
    *,
    preload: bool = True,
) -> mne.io.BaseRaw:
    """Read one subject's raw continuous BrainVision EEG without filtering."""

    root = Path(root)
    subject = normalize_subject(subject)
    vhdr = root / f"{subject}/eeg/{subject}_task-eegsvr_eeg.vhdr"
    if not vhdr.exists():
        raise FileNotFoundError(f"Missing {vhdr}. Run download_ds006897_subjects first.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Omitted .* annotation\\(s\\) that were outside data range.")
        return mne.io.read_raw_brainvision(vhdr, preload=preload, verbose=False)


def ds006897_sampling_rate_table(root: str | Path, subjects) -> pd.DataFrame:
    """Return native sampling rates for selected downloaded subjects."""

    root = Path(root)
    subjects = resolve_ds006897_subjects(root, subjects)
    rows = []
    for subject in subjects:
        raw = load_ds006897_raw(root, subject, preload=False)
        rows.append({"subject": subject, "fs": float(raw.info["sfreq"])})
        if hasattr(raw, "close"):
            raw.close()
    return pd.DataFrame(rows)


def load_ds006897_events(root: str | Path, subject: str | int) -> pd.DataFrame:
    """Load one subject's events table."""

    root = Path(root)
    subject = normalize_subject(subject)
    path = root / f"{subject}/eeg/{subject}_task-eegsvr_events.tsv"
    return pd.read_csv(path, sep="\t")


def load_ds006897_channels(root: str | Path, subject: str | int) -> pd.DataFrame:
    """Load one subject's channel table."""

    root = Path(root)
    subject = normalize_subject(subject)
    path = root / f"{subject}/eeg/{subject}_task-eegsvr_channels.tsv"
    return pd.read_csv(path, sep="\t")


def _available_good_channels(root: Path, subject: str, requested_channels) -> list[str]:
    channels = load_ds006897_channels(root, subject)
    good = channels.loc[channels["status"].fillna("good").eq("good"), "name"].tolist()
    return [channel for channel in requested_channels if channel in good]


def _trial_windows(
    events: pd.DataFrame,
    condition: str,
    *,
    fs: float,
    window_seconds: float,
    onset_offset_seconds: float,
    max_trials: int | None,
) -> pd.DataFrame:
    """Find fixed-length windows after rhythmic visual stimulation onset."""

    stim_event = STIM_EVENT_BY_CONDITION[condition]
    condition_events = events[
        events["Hz_condition"].astype(str).eq(condition)
        & events["trial_type"].astype(str).eq(stim_event)
    ].copy()
    if max_trials is not None:
        condition_events = condition_events.head(max_trials)

    if "onset" in condition_events:
        starts = np.round((condition_events["onset"].astype(float).to_numpy() + onset_offset_seconds) * fs).astype(int)
    else:
        starts = condition_events["sample"].astype(int).to_numpy() + int(round(onset_offset_seconds * fs))
    stops = starts + int(round(window_seconds * fs))
    out = condition_events[["onset", "sample", "trial_type", "Hz_condition", "temp_condition"]].copy()
    out["sample_start"] = starts
    out["sample_stop"] = stops
    out["window_start_s"] = starts / fs
    out["window_stop_s"] = stops / fs
    return out.reset_index(drop=True)


def load_ds006897_condition_matrices(
    root: str | Path,
    subjects,
    *,
    conditions=DEFAULT_CONDITIONS,
    channels=DEFAULT_VISUAL_CHANNELS,
    window_seconds: float = 3.0,
    onset_offset_seconds: float = 0.25,
    max_trials_per_condition: int | None = None,
    standardize: bool = False,
    target_fs: float | str | None = "mode",
) -> dict[str, SignalMatrix]:
    """Load raw EEG windows into one signal matrix per condition.

    Rows are subject x local-channel x trial windows. If ``target_fs`` is a
    number, or ``"mode"``, subjects whose native rate differs are resampled with
    MNE before window extraction. Event windows are indexed from BIDS onset
    times so resampled and native recordings align in seconds. Pass
    ``target_fs=None`` to require every subject to match and skip resampling.
    """

    root = Path(root)
    subjects = resolve_ds006897_subjects(root, subjects)
    fs_table = ds006897_sampling_rate_table(root, subjects)
    fs_counts = fs_table["fs"].value_counts()
    if target_fs is None:
        if len(fs_counts) > 1:
            raise RuntimeError(f"Mixed sampling rates are not supported: {sorted(fs_counts.index)}")
        selected_fs = float(fs_counts.index[0])
    elif isinstance(target_fs, str):
        if target_fs != "mode":
            raise ValueError("target_fs must be None, 'mode', or a numeric sampling rate.")
        selected_fs = float(fs_counts.idxmax())
    else:
        selected_fs = float(target_fs)

    blocks = {condition: [] for condition in conditions}
    row_tables = {condition: [] for condition in conditions}

    for subject in fs_table["subject"].tolist():
        raw = load_ds006897_raw(root, subject, preload=True)
        native_fs = float(raw.info["sfreq"])
        if not np.isclose(native_fs, selected_fs):
            raw.resample(selected_fs, npad="auto", verbose=False)
        fs = float(raw.info["sfreq"])

        channel_names = _available_good_channels(root, subject, channels)
        if not channel_names:
            raise RuntimeError(f"No requested good visual channels found for {subject}.")
        data = raw.get_data(picks=channel_names)
        events = load_ds006897_events(root, subject)

        for condition in conditions:
            windows = _trial_windows(
                events,
                condition,
                fs=fs,
                window_seconds=window_seconds,
                onset_offset_seconds=onset_offset_seconds,
                max_trials=max_trials_per_condition,
            )
            X_rows = []
            rows = []
            for trial_index, trial in windows.iterrows():
                start = int(trial["sample_start"])
                stop = int(trial["sample_stop"])
                if start < 0 or stop > data.shape[1]:
                    continue
                segment = data[:, start:stop]
                X_rows.append(segment)
                for channel_index, channel in enumerate(channel_names):
                    row = {
                        "subject": subject,
                        "channel": channel,
                        "channel_index": channel_index,
                        "trial_index": trial_index,
                        "condition": condition,
                        "native_fs": native_fs,
                        "analysis_fs": fs,
                        "resampled": not np.isclose(native_fs, fs),
                        "temp_condition": trial.get("temp_condition"),
                        "window_start_s": trial["window_start_s"],
                        "window_stop_s": trial["window_stop_s"],
                        "window_seconds": window_seconds,
                        "onset_offset_seconds": onset_offset_seconds,
                    }
                    rows.append(row)
            if X_rows:
                blocks[condition].append(np.vstack(X_rows))
                row_tables[condition].append(pd.DataFrame(rows))
        if hasattr(raw, "close"):
            raw.close()

    matrices = {}
    for condition in conditions:
        if not blocks[condition]:
            continue
        X = np.vstack(blocks[condition]).astype(np.float32, copy=False)
        if standardize:
            X = standardize_rows(X)
        rows = pd.concat(row_tables[condition], ignore_index=True)
        matrices[condition] = SignalMatrix(X, rows, selected_fs, condition)

    return matrices


def summarize_signal_matrices(matrices: dict[str, SignalMatrix]) -> pd.DataFrame:
    """Return row, subject, channel, and window counts for each condition."""

    rows = []
    for condition, matrix in matrices.items():
        rows.append(
            {
                "condition": condition,
                "rows": matrix.X.shape[0],
                "columns": matrix.X.shape[1],
                "seconds": matrix.X.shape[1] / matrix.fs,
                "fs": matrix.fs,
                "subjects": matrix.rows["subject"].nunique(),
                "channels": matrix.rows["channel"].nunique(),
                "trials": matrix.rows[["subject", "trial_index"]].drop_duplicates().shape[0],
            }
        )
    return pd.DataFrame(rows)


def read_eeg_sidecar(root: str | Path, subject: str | int) -> dict:
    """Read one subject EEG JSON sidecar."""

    root = Path(root)
    subject = normalize_subject(subject)
    path = root / f"{subject}/eeg/{subject}_task-eegsvr_eeg.json"
    with path.open() as f:
        return json.load(f)
