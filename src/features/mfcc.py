"""Manual audio feature extraction for classical ML baselines."""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.data.audio import crop_or_pad, extract_segment, load_audio


def summarize_time_axis(values: np.ndarray) -> np.ndarray:
    """Summarize a time-feature matrix with simple statistics.

    Args:
        values: Feature matrix with features on axis 0 and time on axis 1.

    Returns:
        Flattened mean, standard deviation, minimum, and maximum statistics.
    """
    return np.concatenate(
        [
            values.mean(axis=1),
            values.std(axis=1),
            values.min(axis=1),
            values.max(axis=1),
        ]
    )


def extract_manual_features(waveform: np.ndarray, sample_rate: int, mfcc_count: int) -> np.ndarray:
    """Extract MFCC and spectral statistics from one waveform.

    Args:
        waveform: Mono waveform array.
        sample_rate: Audio sample rate.
        mfcc_count: Number of MFCC coefficients.

    Returns:
        One-dimensional feature vector.
    """
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=mfcc_count)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)
    rms = librosa.feature.rms(y=waveform)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    return np.concatenate(
        [
            summarize_time_axis(mfcc),
            summarize_time_axis(delta),
            summarize_time_axis(delta_delta),
            summarize_time_axis(np.vstack([centroid, bandwidth, rolloff, rms, zcr])),
        ]
    )


def build_feature_table(
    index_csv: str | Path,
    sample_rate: int,
    duration: float,
    mfcc_count: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build manual features for all rows in an index CSV.

    Args:
        index_csv: Metadata CSV path.
        sample_rate: Target sample rate.
        duration: Fixed segment duration.
        mfcc_count: Number of MFCC coefficients.

    Returns:
        Feature matrix, label vector, and metadata frame.
    """
    frame = pd.read_csv(index_csv)
    features = []
    labels = []
    for row in frame.itertuples(index=False):
        waveform = load_audio(row.file, sample_rate)
        segment = extract_segment(waveform, sample_rate, float(row.start), float(row.end))
        segment = crop_or_pad(segment, sample_rate, duration).numpy()
        features.append(extract_manual_features(segment, sample_rate, mfcc_count))
        labels.append(int(row.label))
    return np.vstack(features), np.asarray(labels), frame
