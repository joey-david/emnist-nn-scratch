#!/usr/bin/env python3
"""Utility to download a YouTube clip, extract audio features, and persist artifacts."""
from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import soundfile as sf
import yt_dlp

import librosa
import librosa.display
import matplotlib.pyplot as plt

try:  # pragma: no cover - optional availability depends on librosa version
    from librosa.feature import rhythm as _librosa_rhythm
except Exception:  # pragma: no cover
    _librosa_rhythm = None


@dataclass
class ProcessedClip:
    audio_path: Path
    image_path: Path
    features: Dict[str, object]


@dataclass
class DownloadedAudio:
    path: Path
    duration: float | None


FEATURE_HEADER: List[str] = [
    "filename",
    "length",
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_mean",
    "spectral_centroid_var",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
    "perceptr_mean",
    "perceptr_var",
    "tempo",
    "mfcc1_mean",
    "mfcc1_var",
    "mfcc2_mean",
    "mfcc2_var",
    "mfcc3_mean",
    "mfcc3_var",
    "mfcc4_mean",
    "mfcc4_var",
    "mfcc5_mean",
    "mfcc5_var",
    "mfcc6_mean",
    "mfcc6_var",
    "mfcc7_mean",
    "mfcc7_var",
    "mfcc8_mean",
    "mfcc8_var",
    "mfcc9_mean",
    "mfcc9_var",
    "mfcc10_mean",
    "mfcc10_var",
    "mfcc11_mean",
    "mfcc11_var",
    "mfcc12_mean",
    "mfcc12_var",
    "mfcc13_mean",
    "mfcc13_var",
    "mfcc14_mean",
    "mfcc14_var",
    "mfcc15_mean",
    "mfcc15_var",
    "mfcc16_mean",
    "mfcc16_var",
    "mfcc17_mean",
    "mfcc17_var",
    "mfcc18_mean",
    "mfcc18_var",
    "mfcc19_mean",
    "mfcc19_var",
    "mfcc20_mean",
    "mfcc20_var",
    "label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-y",
        "--youtube",
        required=True,
        help="YouTube URL to download",
    )
    parser.add_argument(
        "-f",
        "--features-csv",
        type=Path,
        help="Optional CSV file to append the computed features to",
    )
    parser.add_argument(
        "-l",
        "--label",
        default="",
        help="Genre/label to store alongside the features",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path.cwd() / "dataset" / "custom_wav",
        help="Directory where the WAV clip should be stored",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path.cwd() / "dataset" / "custom_spectrograms",
        help="Directory where the spectrogram image should be written",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration of the clip to extract in seconds",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="Starting offset (seconds) within the source audio; defaults to a random window",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate to use when loading and exporting the audio",
    )
    return parser.parse_args()


def download_audio(youtube_url: str, target_dir: Path) -> DownloadedAudio:
    target_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    duration = info.get("duration") if isinstance(info, dict) else None
    try:
        duration_value = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_value = None
    if not downloaded.exists():
        raise FileNotFoundError(f"failed to download audio for {youtube_url}")
    return DownloadedAudio(path=downloaded, duration=duration_value)


def load_clip(path: Path, sr: int, duration: float, offset: float) -> np.ndarray:
    warnings.filterwarnings("ignore", category=UserWarning)
    y, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
    if y.size == 0:
        raise RuntimeError("Loaded clip is empty; check the offset/duration")
    return y


def export_wav(samples: np.ndarray, sr: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, samples, sr)


def build_feature_row(
    filename: str, samples: np.ndarray, sr: int, label: str
) -> Dict[str, object]:
    def mean_var(feature: np.ndarray) -> np.ndarray:
        flat = np.ravel(feature)
        return np.array([float(np.mean(flat)), float(np.var(flat))])

    row: Dict[str, float] = {
        "filename": filename,
        "length": int(samples.size),
    }

    chroma = librosa.feature.chroma_stft(y=samples, sr=sr)
    row["chroma_stft_mean"], row["chroma_stft_var"] = mean_var(chroma)

    rms = librosa.feature.rms(y=samples)
    row["rms_mean"], row["rms_var"] = mean_var(rms)

    spec_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
    row["spectral_centroid_mean"], row["spectral_centroid_var"] = mean_var(spec_centroid)

    spec_bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)
    row["spectral_bandwidth_mean"], row["spectral_bandwidth_var"] = mean_var(spec_bandwidth)

    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)
    row["rolloff_mean"], row["rolloff_var"] = mean_var(rolloff)

    zcr = librosa.feature.zero_crossing_rate(samples)
    row["zero_crossing_rate_mean"], row["zero_crossing_rate_var"] = mean_var(zcr)

    harmony = librosa.effects.harmonic(samples)
    row["harmony_mean"] = float(np.mean(harmony))
    row["harmony_var"] = float(np.var(harmony))

    percussive = librosa.effects.percussive(samples)
    row["perceptr_mean"] = float(np.mean(percussive))
    row["perceptr_var"] = float(np.var(percussive))

    if _librosa_rhythm is not None:
        tempo = _librosa_rhythm.tempo(y=samples, sr=sr)
    else:
        tempo = librosa.beat.tempo(y=samples, sr=sr)
    row["tempo"] = float(tempo[0] if tempo.size else 0.0)

    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=20)
    for i in range(20):
        mean_val, var_val = mean_var(mfcc[i])
        row[f"mfcc{i + 1}_mean"] = mean_val
        row[f"mfcc{i + 1}_var"] = var_val

    row["label"] = label
    return row


def create_spectrogram(samples: np.ndarray, sr: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.switch_backend("Agg")
    fig = plt.figure(figsize=(6, 4), dpi=72)
    ax = fig.add_subplot(111)
    S = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, ax=ax, y_axis="mel", x_axis="time")
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(destination, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def append_to_csv(csv_path: Path, header: Iterable[str], row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(header)
        writer.writerow([row[key] for key in header])


def process_clip(args: argparse.Namespace) -> ProcessedClip:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        downloaded = download_audio(args.youtube, temp_dir)
        source_path = downloaded.path

        offset = args.offset
        if offset is None:
            track_duration = downloaded.duration
            if track_duration is None or not np.isfinite(track_duration):
                try:
                    file_info = sf.info(str(source_path))
                except (sf.SoundFileRuntimeError, RuntimeError):
                    file_info = None
                if file_info and np.isfinite(file_info.duration):
                    track_duration = float(file_info.duration)
                else:
                    track_duration = 0.0

            if track_duration <= args.duration:
                offset = 0.0
            else:
                max_offset = track_duration - args.duration
                offset = random.uniform(0.0, max_offset)

        samples = load_clip(
            source_path,
            sr=args.sample_rate,
            duration=args.duration,
            offset=float(offset),
        )

    base_stem = source_path.stem
    wav_filename = f"{base_stem}_clip.wav"
    audio_dir = args.wav_dir / args.label if args.label else args.wav_dir
    image_dir = args.image_dir / args.label if args.label else args.image_dir

    audio_path = audio_dir / wav_filename
    export_wav(samples, args.sample_rate, audio_path)

    features = build_feature_row(audio_path.name, samples, args.sample_rate, args.label)

    image_path = image_dir / audio_path.with_suffix(".png").name
    create_spectrogram(samples, args.sample_rate, image_path)

    return ProcessedClip(audio_path=audio_path, image_path=image_path, features=features)


def main() -> None:
    args = parse_args()
    clip = process_clip(args)

    if args.features_csv:
        append_to_csv(args.features_csv, FEATURE_HEADER, clip.features)

    ordered_values = [clip.features[key] for key in FEATURE_HEADER]
    print(",".join(str(value) for value in ordered_values))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface errors to CLI
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
