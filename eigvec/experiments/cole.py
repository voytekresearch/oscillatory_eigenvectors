"""Cole et al. 2017 Parkinson M1 ECoG beta dataset loader."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .data import SignalMatrix, standardize_rows, window_traces


COLE_2017_DATA_URL = "https://github.com/voytekresearch/Cole_2017/raw/master/data.mat"
COLE_2017_REPO_URL = "https://github.com/voytekresearch/Cole_2017"


def ensure_cole_2017_data(cache_dir: str | Path) -> Path:
    """Return local ``data.mat``, downloading it from the public repo if needed."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "data.mat"
    if not data_path.exists():
        urlretrieve(COLE_2017_DATA_URL, data_path)
    return data_path


def load_cole_2017_m1_beta(
    cache_dir: str | Path,
    conditions=("B", "D"),
    labels=None,
    fs: float = 1000.0,
    window_seconds: float = 2.0,
    step_seconds: float | None = None,
    standardize: bool = True,
) -> tuple[dict[str, SignalMatrix], pd.DataFrame]:
    """Load Cole 2017 M1 ECoG beta into condition-specific signal matrices.

    The public ``data.mat`` contains 23 M1 ECoG traces for condition ``B`` and
    condition ``D``. The original paper studies nonsinusoidal beta waveform
    shape in Parkinson disease. Here, rows are subject/trace by time-window.
    """

    data_path = ensure_cole_2017_data(cache_dir)
    data = loadmat(data_path, squeeze_me=True, struct_as_record=False)
    labels = labels or {
        "B": "pre_dbs_untreated",
        "D": "on_dbs",
    }

    matrices: dict[str, SignalMatrix] = {}
    summary = []
    for condition in conditions:
        if condition not in data:
            available = [key for key in data if not key.startswith("__")]
            raise ValueError(f"Condition {condition!r} not found. Available: {available}")

        traces = [np.asarray(trace, dtype=np.float32).squeeze() for trace in np.ravel(data[condition])]
        trace_lengths = {trace.shape[0] for trace in traces}
        if len(trace_lengths) != 1:
            raise RuntimeError(f"Condition {condition!r} has unequal trace lengths: {trace_lengths}")

        X, rows = window_traces(np.vstack(traces), fs, window_seconds, step_seconds)
        if standardize:
            X = standardize_rows(X)

        label = labels.get(condition, condition)
        rows.insert(0, "condition", label)
        rows.insert(1, "cole_condition", condition)
        rows.rename(columns={"trace_index": "subject_index"}, inplace=True)
        rows["source_file"] = str(data_path)

        matrices[label] = SignalMatrix(X, rows, fs=fs, condition=label)
        summary.append(
            {
                "condition": label,
                "cole_condition": condition,
                "shape": X.shape,
                "subjects_or_traces": len(traces),
                "windows_per_trace": rows["window_index"].nunique(),
                "fs": fs,
                "window_seconds": window_seconds,
                "source_file": str(data_path),
                "source_url": COLE_2017_REPO_URL,
            }
        )

    return matrices, pd.DataFrame(summary)
