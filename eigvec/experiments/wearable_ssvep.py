"""THU wearable SSVEP dataset helpers."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .data import SignalMatrix, standardize_rows


WEARABLE_BASE_URL = "https://bci.med.tsinghua.edu.cn/upload/zhufangkun"
WEARABLE_ARCHIVE_GROUPS = (
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 90),
    (91, 102),
)
WEARABLE_CHANNELS = ("POz", "PO3", "PO4", "PO5", "PO6", "Oz", "O1", "O2")
WEARABLE_ELECTRODE_MODES = ("dry", "wet")
WEARABLE_FREQUENCIES = np.array(
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75],
    dtype=float,
)
WEARABLE_PHASES_PI = np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5], dtype=float)
WEARABLE_FS = 250.0


def wearable_condition_label(frequency: float) -> str:
    """Return stable labels such as ``11p25Hz`` for frequency conditions."""

    return f"{float(frequency):g}Hz".replace(".", "p")


def _archive_name(group: tuple[int, int]) -> str:
    return f"S{group[0]:03d}-S{group[1]:03d}.zip"


def _archive_groups(groups) -> tuple[tuple[int, int], ...]:
    if isinstance(groups, str):
        if groups.lower() != "all":
            raise ValueError("groups must be 'all' or an iterable of (start, stop) tuples.")
        return WEARABLE_ARCHIVE_GROUPS
    return tuple((int(start), int(stop)) for start, stop in groups)


def _download(url: str, target: Path, *, overwrite: bool = False) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return "cached"
    urllib.request.urlretrieve(url, target)
    return "downloaded"


def download_wearable_ssvep(
    root: str | Path,
    *,
    groups="all",
    overwrite: bool = False,
    extract: bool = True,
) -> pd.DataFrame:
    """Download selected THU wearable SSVEP archives."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for filename in ("Readme.pdf", "stimulation_information.pdf"):
        status = _download(f"{WEARABLE_BASE_URL}/{filename}", root / filename, overwrite=overwrite)
        rows.append({"kind": "metadata", "file": filename, "status": status})

    for group in _archive_groups(groups):
        filename = _archive_name(group)
        archive = root / filename
        status = _download(f"{WEARABLE_BASE_URL}/{filename}", archive, overwrite=overwrite)
        rows.append({"kind": "archive", "file": filename, "status": status})

        if not extract:
            continue

        expected = [root / f"S{subject:03d}.mat" for subject in range(group[0], group[1] + 1)]
        if overwrite or any(not path.exists() for path in expected):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(root)
            extract_status = "extracted"
        else:
            extract_status = "cached"
        rows.append({"kind": "extract", "file": filename, "status": extract_status})

    return pd.DataFrame(rows)


def normalize_wearable_subject(subject: str | int) -> str:
    """Return subject ids as ``S001`` ... ``S102`` strings."""

    if isinstance(subject, str) and subject.upper().startswith("S"):
        return f"S{int(subject[1:]):03d}"
    return f"S{int(subject):03d}"


def resolve_wearable_subjects(root: str | Path, subjects="available") -> list[str]:
    """Resolve subject selection to concrete wearable SSVEP subject ids."""

    root = Path(root)
    if isinstance(subjects, str):
        key = subjects.lower()
        if key == "all":
            return [f"S{index:03d}" for index in range(1, 103)]
        if key == "available":
            found = sorted(root.glob("S*.mat"), key=lambda path: int(path.stem[1:]))
            return [path.stem for path in found]
        return [normalize_wearable_subject(subjects)]
    if isinstance(subjects, int):
        return [normalize_wearable_subject(subjects)]
    return [normalize_wearable_subject(subject) for subject in subjects]


def load_wearable_subject(root: str | Path, subject: str | int) -> np.ndarray:
    """Load one wearable SSVEP subject matrix."""

    root = Path(root)
    subject = normalize_wearable_subject(subject)
    path = root / f"{subject}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run download_wearable_ssvep first.")
    return np.asarray(loadmat(path, squeeze_me=False, struct_as_record=False)["data"])


def wearable_frequency_table() -> pd.DataFrame:
    """Return the 12 target frequencies and phases."""

    return pd.DataFrame(
        {
            "condition_index": np.arange(1, len(WEARABLE_FREQUENCIES) + 1),
            "condition": [wearable_condition_label(freq) for freq in WEARABLE_FREQUENCIES],
            "stimulus_frequency_hz": WEARABLE_FREQUENCIES,
            "stimulus_phase_pi": WEARABLE_PHASES_PI,
        }
    )


def wearable_harmonic_bands(
    frequencies,
    *,
    max_freq: float = 80.0,
    half_width: float = 0.35,
) -> list[tuple[float, float]]:
    """Return merged bands around selected stimulus frequencies and harmonics."""

    bands = []
    for frequency in np.atleast_1d(np.asarray(frequencies, dtype=float)):
        harmonic = 1
        while harmonic * frequency <= max_freq:
            center = harmonic * frequency
            bands.append((center - half_width, center + half_width))
            harmonic += 1
    bands = sorted(bands)

    merged = []
    for low, high in bands:
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def wearable_bycycle_bands(frequencies, *, half_width: float = 0.75) -> dict[str, tuple[float, float]]:
    """Return condition-specific bycycle bands around each fundamental."""

    return {
        wearable_condition_label(freq): (float(freq) - half_width, float(freq) + half_width)
        for freq in np.atleast_1d(np.asarray(frequencies, dtype=float))
    }


def _condition_index(frequency: float) -> int:
    matches = np.flatnonzero(np.isclose(WEARABLE_FREQUENCIES, frequency, atol=1e-8))
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely find wearable SSVEP frequency {frequency:g} Hz.")
    return int(matches[0])


def load_wearable_condition_matrices(
    root: str | Path,
    subjects="available",
    *,
    frequencies=(11.25, 12.25),
    channels=("O2",),
    electrode_modes=WEARABLE_ELECTRODE_MODES,
    window_start_seconds: float = 0.5,
    window_seconds: float = 2.0,
    max_blocks: int | None = None,
    standardize: bool = False,
    dtype=np.float32,
) -> dict[str, SignalMatrix]:
    """Load wearable SSVEP windows into one matrix per stimulus frequency.

    Rows are subject x requested channel x electrode mode x block. No filtering
    is applied; the source files are already 250 Hz epoched data.
    """

    root = Path(root)
    subjects = resolve_wearable_subjects(root, subjects)
    frequencies = tuple(float(frequency) for frequency in frequencies)
    requested_channels = [channel.upper() for channel in channels]
    requested_modes = [mode.lower() for mode in electrode_modes]

    channel_lookup = {channel.upper(): index for index, channel in enumerate(WEARABLE_CHANNELS)}
    mode_lookup = {mode: index for index, mode in enumerate(WEARABLE_ELECTRODE_MODES)}
    channel_indices = [channel_lookup[channel] for channel in requested_channels if channel in channel_lookup]
    channel_names = [WEARABLE_CHANNELS[index] for index in channel_indices]
    mode_indices = [mode_lookup[mode] for mode in requested_modes if mode in mode_lookup]
    mode_names = [WEARABLE_ELECTRODE_MODES[index] for index in mode_indices]
    if not channel_indices:
        raise RuntimeError(f"No requested channels found: {channels}.")
    if not mode_indices:
        raise RuntimeError(f"No requested electrode modes found: {electrode_modes}.")

    start = int(round(window_start_seconds * WEARABLE_FS))
    stop = start + int(round(window_seconds * WEARABLE_FS))

    blocks = {wearable_condition_label(freq): [] for freq in frequencies}
    row_tables = {wearable_condition_label(freq): [] for freq in frequencies}
    freq_table = wearable_frequency_table().set_index("condition_index")

    for subject in subjects:
        data = load_wearable_subject(root, subject)
        if data.shape[:5] != (8, 710, 2, 10, 12):
            raise RuntimeError(f"Unexpected data shape for {subject}: {data.shape}.")
        if stop > data.shape[1]:
            continue
        n_blocks = data.shape[3] if max_blocks is None else min(int(max_blocks), data.shape[3])

        for frequency in frequencies:
            target_index = _condition_index(frequency)
            label = wearable_condition_label(frequency)
            X_rows = []
            rows = []
            for channel_index, channel in zip(channel_indices, channel_names):
                for mode_index, mode in zip(mode_indices, mode_names):
                    segment = data[channel_index, start:stop, mode_index, :n_blocks, target_index]
                    X_rows.append(segment.T)
                    for block_index in range(n_blocks):
                        rows.append(
                            {
                                "subject": subject,
                                "subject_index": int(subject[1:]),
                                "channel": channel,
                                "channel_index": channel_index,
                                "electrode_mode": mode,
                                "electrode_mode_index": mode_index,
                                "block_index": block_index,
                                "condition": label,
                                "condition_index": target_index + 1,
                                "stimulus_frequency_hz": frequency,
                                "stimulus_phase_pi": float(freq_table.loc[target_index + 1, "stimulus_phase_pi"]),
                                "window_start_s": start / WEARABLE_FS,
                                "window_stop_s": stop / WEARABLE_FS,
                                "window_seconds": (stop - start) / WEARABLE_FS,
                                "source_file": f"{subject}.mat",
                            }
                        )
            if X_rows:
                blocks[label].append(np.vstack(X_rows))
                row_tables[label].append(pd.DataFrame(rows))

    matrices = {}
    for label in blocks:
        if not blocks[label]:
            continue
        X = np.vstack(blocks[label]).astype(dtype, copy=False)
        if standardize:
            X = standardize_rows(X)
        rows = pd.concat(row_tables[label], ignore_index=True)
        matrices[label] = SignalMatrix(X, rows, WEARABLE_FS, label)
    if not matrices:
        raise RuntimeError(f"No wearable SSVEP matrices could be loaded from {root}.")
    return matrices


def summarize_wearable_signal_matrices(matrices: dict[str, SignalMatrix]) -> pd.DataFrame:
    """Return compact matrix and row provenance summaries."""

    rows = []
    for condition, matrix in matrices.items():
        meta = matrix.rows
        rows.append(
            {
                "condition": condition,
                "rows": matrix.X.shape[0],
                "columns": matrix.X.shape[1],
                "fs": matrix.fs,
                "subjects": meta["subject"].nunique() if "subject" in meta else np.nan,
                "channels": meta["channel"].nunique() if "channel" in meta else np.nan,
                "electrode_modes": meta["electrode_mode"].nunique() if "electrode_mode" in meta else np.nan,
                "blocks": meta["block_index"].nunique() if "block_index" in meta else np.nan,
            }
        )
    return pd.DataFrame(rows)
