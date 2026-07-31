"""BETA SSVEP dataset helpers."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .data import SignalMatrix, standardize_rows


BETA_WOF_BASE_URL = "https://bci.med.tsinghua.edu.cn/upload/liubingchuan_BETA_wof"
BETA_ARCHIVE_GROUPS = (
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
)
BETA_VISUAL_CHANNELS = ("PZ", "PO3", "PO5", "PO4", "PO6", "POZ", "O1", "OZ", "O2")
BETA_DEFAULT_ANALYSIS_CHANNEL = "OZ"


def normalize_beta_subject(subject: str | int) -> str:
    """Return BETA subject ids as ``S1`` ... ``S70`` strings."""

    if isinstance(subject, str) and subject.upper().startswith("S"):
        return f"S{int(subject[1:])}"
    return f"S{int(subject)}"


def beta_condition_label(frequency: float) -> str:
    """Return stable labels such as ``10p6Hz`` for frequency conditions."""

    return f"{frequency:g}Hz".replace(".", "p")


def _archive_name(group: tuple[int, int]) -> str:
    return f"S{group[0]}-S{group[1]}.tar.gz"


def _archive_groups(groups) -> tuple[tuple[int, int], ...]:
    if isinstance(groups, str):
        if groups.lower() != "all":
            raise ValueError("groups must be 'all' or an iterable of (start, stop) tuples.")
        return BETA_ARCHIVE_GROUPS
    return tuple((int(start), int(stop)) for start, stop in groups)


def _download(url: str, target: Path, *, overwrite: bool = False) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return "cached"
    urllib.request.urlretrieve(url, target)
    return "downloaded"


def download_beta_wof(
    root: str | Path,
    *,
    groups="all",
    overwrite: bool = False,
    extract: bool = True,
) -> pd.DataFrame:
    """Download selected BETA-without-software-filtering archives.

    The ``wof`` release is still downsampled to 250 Hz by the dataset authors,
    but it omits the original release's 3-100 Hz software band-pass filtering.
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for filename in ("note.pdf", "description.pdf"):
        status = _download(f"{BETA_WOF_BASE_URL}/{filename}", root / filename, overwrite=overwrite)
        rows.append({"kind": "metadata", "file": filename, "status": status})

    for group in _archive_groups(groups):
        filename = _archive_name(group)
        archive = root / filename
        status = _download(f"{BETA_WOF_BASE_URL}/{filename}", archive, overwrite=overwrite)
        rows.append({"kind": "archive", "file": filename, "status": status})

        if not extract:
            continue

        expected = [root / f"S{subject}.mat" for subject in range(group[0], group[1] + 1)]
        if overwrite or any(not path.exists() for path in expected):
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(root)
            extract_status = "extracted"
        else:
            extract_status = "cached"
        rows.append({"kind": "extract", "file": filename, "status": extract_status})

    return pd.DataFrame(rows)


def resolve_beta_subjects(root: str | Path, subjects="available") -> list[str]:
    """Resolve subject selection to concrete BETA subject ids."""

    root = Path(root)
    if isinstance(subjects, str):
        key = subjects.lower()
        if key == "all":
            return [f"S{index}" for index in range(1, 71)]
        if key == "available":
            found = sorted(root.glob("S*.mat"), key=lambda path: int(path.stem[1:]))
            return [path.stem for path in found]
        return [normalize_beta_subject(subjects)]
    if isinstance(subjects, int):
        return [normalize_beta_subject(subjects)]
    return [normalize_beta_subject(subject) for subject in subjects]


def load_beta_subject(root: str | Path, subject: str | int):
    """Load one BETA ``S*.mat`` file as the MATLAB ``data`` struct."""

    root = Path(root)
    subject = normalize_beta_subject(subject)
    path = root / f"{subject}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run download_beta_wof first.")
    return loadmat(path, squeeze_me=True, struct_as_record=False)["data"]


def beta_frequency_table(root: str | Path, subject: str | int | None = None) -> pd.DataFrame:
    """Return the 40 BETA target frequencies and phases."""

    root = Path(root)
    if subject is None:
        subjects = resolve_beta_subjects(root, "available")
        if not subjects:
            raise FileNotFoundError(f"No S*.mat files found in {root}.")
        subject = subjects[0]

    data = load_beta_subject(root, subject)
    freqs = np.asarray(data.suppl_info.freqs, dtype=float)
    phases = np.asarray(data.suppl_info.phases, dtype=float)
    return pd.DataFrame(
        {
            "condition_index": np.arange(1, len(freqs) + 1),
            "condition": [beta_condition_label(freq) for freq in freqs],
            "stimulus_frequency_hz": freqs,
            "stimulus_phase_rad": phases,
        }
    )


def beta_channel_table(root: str | Path, subject: str | int | None = None) -> pd.DataFrame:
    """Return BETA channel metadata."""

    root = Path(root)
    if subject is None:
        subjects = resolve_beta_subjects(root, "available")
        if not subjects:
            raise FileNotFoundError(f"No S*.mat files found in {root}.")
        subject = subjects[0]

    data = load_beta_subject(root, subject)
    chan = np.asarray(data.suppl_info.chan)
    return pd.DataFrame(
        {
            "channel_index": chan[:, 0].astype(int) - 1,
            "theta_deg": chan[:, 1].astype(float),
            "radius": chan[:, 2].astype(float),
            "channel": [str(name).upper() for name in chan[:, 3]],
        }
    )


def beta_harmonic_bands(
    frequencies,
    *,
    max_freq: float = 90.0,
    half_width: float = 0.35,
) -> list[tuple[float, float]]:
    """Return bands around each selected stimulus frequency and harmonic."""

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


def beta_bycycle_bands(frequencies, *, half_width: float = 1.0) -> dict[str, tuple[float, float]]:
    """Return condition-specific bycycle bands around each fundamental."""

    out = {}
    for frequency in np.atleast_1d(np.asarray(frequencies, dtype=float)):
        out[beta_condition_label(frequency)] = (frequency - half_width, frequency + half_width)
    return out


def _condition_index(freqs: np.ndarray, frequency: float) -> int:
    matches = np.flatnonzero(np.isclose(freqs, frequency, atol=1e-8))
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely find BETA stimulus frequency {frequency:g} Hz.")
    return int(matches[0])


def load_beta_condition_matrices(
    root: str | Path,
    subjects="available",
    *,
    frequencies=(10.6, 15.8),
    channels=(BETA_DEFAULT_ANALYSIS_CHANNEL,),
    window_start_seconds: float = 0.5,
    window_seconds: float = 2.0,
    max_blocks: int | None = None,
    standardize: bool = False,
    dtype=np.float32,
) -> dict[str, SignalMatrix]:
    """Load BETA SSVEP windows into one matrix per stimulus frequency.

    Rows are subject x requested channel x block. No filtering is applied.
    Use one channel per call when the downstream model requires spectrally similar rows.
    """

    root = Path(root)
    subjects = resolve_beta_subjects(root, subjects)
    frequencies = tuple(float(frequency) for frequency in frequencies)
    requested_channels = [channel.upper() for channel in channels]
    blocks = {beta_condition_label(freq): [] for freq in frequencies}
    row_tables = {beta_condition_label(freq): [] for freq in frequencies}
    fs_seen = set()

    for subject in subjects:
        data = load_beta_subject(root, subject)
        eeg = np.asarray(data.EEG)
        suppl = data.suppl_info
        fs = float(suppl.srate)
        fs_seen.add(fs)
        if len(fs_seen) > 1:
            raise RuntimeError(f"Mixed sampling rates are not supported: {sorted(fs_seen)}")

        channel_table = beta_channel_table(root, subject)
        channel_lookup = dict(zip(channel_table["channel"], channel_table["channel_index"]))
        channel_indices = [channel_lookup[channel] for channel in requested_channels if channel in channel_lookup]
        channel_names = [channel for channel in requested_channels if channel in channel_lookup]
        if not channel_indices:
            raise RuntimeError(f"No requested visual channels found for {subject}.")

        freqs = np.asarray(suppl.freqs, dtype=float)
        phases = np.asarray(suppl.phases, dtype=float)
        start = int(round(window_start_seconds * fs))
        stop = start + int(round(window_seconds * fs))
        if stop > eeg.shape[1]:
            continue

        n_blocks = eeg.shape[2] if max_blocks is None else min(int(max_blocks), eeg.shape[2])
        for frequency in frequencies:
            condition_index = _condition_index(freqs, frequency)
            label = beta_condition_label(frequency)
            X_rows = []
            rows = []
            for block_index in range(n_blocks):
                segment = np.take(eeg, channel_indices, axis=0)[:, start:stop, block_index, condition_index]
                X_rows.append(segment)
                for local_index, channel in enumerate(channel_names):
                    rows.append(
                        {
                            "subject": subject,
                            "subject_index": int(subject[1:]),
                            "channel": channel,
                            "channel_index": channel_indices[local_index],
                            "block_index": block_index,
                            "condition": label,
                            "condition_index": condition_index + 1,
                            "stimulus_frequency_hz": frequency,
                            "stimulus_phase_rad": phases[condition_index],
                            "window_start_s": start / fs,
                            "window_stop_s": stop / fs,
                            "window_seconds": (stop - start) / fs,
                            "age": getattr(suppl, "age", np.nan),
                            "gender": getattr(suppl, "gender", ""),
                            "source_file": f"{subject}.mat",
                        }
                    )
            if X_rows:
                blocks[label].append(np.vstack(X_rows))
                row_tables[label].append(pd.DataFrame(rows))

    if not fs_seen:
        raise RuntimeError("No usable BETA subject files were loaded.")

    matrices = {}
    fs = fs_seen.pop()
    for label in blocks:
        if not blocks[label]:
            continue
        X = np.vstack(blocks[label]).astype(dtype, copy=False)
        if standardize:
            X = standardize_rows(X)
        rows = pd.concat(row_tables[label], ignore_index=True)
        matrices[label] = SignalMatrix(X, rows, fs, label)
    return matrices


def summarize_beta_signal_matrices(matrices: dict[str, SignalMatrix]) -> pd.DataFrame:
    """Return compact matrix and row provenance summaries."""

    rows = []
    for condition, matrix in matrices.items():
        meta = matrix.rows
        rows.append(
            {
                "condition": condition,
                "stimulus_frequency_hz": meta["stimulus_frequency_hz"].iloc[0],
                "rows": matrix.X.shape[0],
                "columns": matrix.X.shape[1],
                "seconds": matrix.X.shape[1] / matrix.fs,
                "fs": matrix.fs,
                "subjects": meta["subject"].nunique(),
                "channels": meta["channel"].nunique(),
                "blocks": meta[["subject", "block_index"]].drop_duplicates().shape[0],
            }
        )
    return pd.DataFrame(rows)
