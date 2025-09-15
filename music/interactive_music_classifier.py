#!/usr/bin/env python3
"""Interactive CLI for classifying audio clips using the notebook neural net."""
from __future__ import annotations

import sys
import time
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency
    sd = None

from youtube_feature_extractor import (
    FEATURE_HEADER,
    DownloadedAudio,
    append_to_csv,
    build_feature_row,
    create_spectrogram,
    download_audio,
    export_wav,
    load_clip,
)


SAMPLE_RATE = 22050
MAX_CLIP_DURATION = 30.0
DATASET_PATH = Path("dataset") / "features_30_sec.csv"
AUDIO_OUTPUT_DIR = Path("dataset") / "custom_wav"
IMAGE_OUTPUT_DIR = Path("dataset") / "custom_spectrograms"

FEATURE_VECTOR_COLUMNS: List[str] = [
    column
    for column in FEATURE_HEADER
    if column not in {"filename", "length", "label"}
]


class NeuralNet:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_1_size: int,
        hidden_layer_2_size: int,
        learning_rate: float,
    ) -> None:
        self.output_size = output_size
        in_dim = input_size
        h1 = hidden_layer_1_size
        h2 = hidden_layer_2_size

        rng = np.random.default_rng()
        self.W0 = rng.standard_normal((h1, in_dim)) * np.sqrt(2 / in_dim)
        self.W1 = rng.standard_normal((h2, h1)) * np.sqrt(2 / h1)
        self.W2 = rng.standard_normal((output_size, h2)) * np.sqrt(2 / h2)
        self.b0 = np.zeros(h1)
        self.b1 = np.zeros(h2)
        self.b2 = np.zeros(output_size)

        self.L1 = np.zeros(h1)
        self.L2 = np.zeros(h2)
        self.lr = learning_rate

    @staticmethod
    def softmax(array: np.ndarray) -> np.ndarray:
        exps = np.exp(array - np.max(array, axis=-1, keepdims=True))
        return exps / exps.sum(axis=-1, keepdims=True)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return (x > 0) * x

    def feed_forward(self, input_vec: np.ndarray) -> np.ndarray:
        self.L1 = self.relu(self.W0 @ input_vec + self.b0)
        self.L2 = self.relu(self.W1 @ self.L1 + self.b1)
        output = self.softmax(self.W2 @ self.L2 + self.b2)
        return output

    def backward_pass(
        self, input_vec: np.ndarray, pred: np.ndarray, target: np.ndarray
    ) -> float:
        epsilon = 1e-12
        ce_loss = -np.log(pred[np.argmax(target)] + epsilon)

        grad = pred - target
        dW2 = np.outer(grad, self.L2)
        db2 = grad

        dL2 = self.W2.T @ grad * (self.L2 > 0)
        dW1 = np.outer(dL2, self.L1)
        db1 = dL2

        dL1 = self.W1.T @ dL2 * (self.L1 > 0)
        dW0 = np.outer(dL1, input_vec)
        db0 = dL1

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W0 -= self.lr * dW0
        self.b0 -= self.lr * db0

        return ce_loss

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int) -> List[float]:
        losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for x_vec, label_id in zip(x_train, y_train):
                y_ohe = np.zeros(self.output_size)
                y_ohe[label_id] = 1.0
                pred = self.feed_forward(x_vec)
                epoch_loss += self.backward_pass(x_vec, pred, y_ohe)
            losses.append(epoch_loss / max(1, len(x_train)))
        return losses


@dataclass
class ModelState:
    network: NeuralNet
    mu: np.ndarray
    sigma: np.ndarray
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]


def ensure_output_dirs() -> None:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    missing = set(FEATURE_VECTOR_COLUMNS + ["label"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def compute_normalization(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = matrix.mean(axis=0)
    sigma = matrix.std(axis=0) + 1e-8
    return mu, sigma


def train_model(dataset: pd.DataFrame, epochs: int = 50) -> ModelState:
    labels = dataset["label"].astype(str)
    label_to_id: Dict[str, int] = {
        label: idx for idx, label in enumerate(sorted(labels.unique()))
    }
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    feature_matrix = dataset[FEATURE_VECTOR_COLUMNS].to_numpy(dtype=float)
    label_ids = labels.map(label_to_id).to_numpy()

    mu, sigma = compute_normalization(feature_matrix)
    normalized = (feature_matrix - mu) / sigma

    network = NeuralNet(
        input_size=normalized.shape[1],
        output_size=len(label_to_id),
        hidden_layer_1_size=32,
        hidden_layer_2_size=16,
        learning_rate=1e-3,
    )
    network.train(normalized, label_ids, epochs=epochs)
    return ModelState(network=network, mu=mu, sigma=sigma, label_to_id=label_to_id, id_to_label=id_to_label)


def normalize_features(model_state: ModelState, feature_vector: np.ndarray) -> np.ndarray:
    return (feature_vector - model_state.mu) / model_state.sigma


def predict_label(model_state: ModelState, feature_vector: np.ndarray) -> Tuple[str, np.ndarray]:
    normalized = normalize_features(model_state, feature_vector)
    probs = model_state.network.feed_forward(normalized)
    prediction = model_state.id_to_label[int(np.argmax(probs))]
    return prediction, probs


def clip_duration_from_metadata(downloaded: DownloadedAudio) -> Optional[float]:
    duration = downloaded.duration
    if duration is None or not np.isfinite(duration):
        return None
    return float(duration)


def extract_youtube_samples(youtube_url: str, target_duration: float) -> Tuple[str, np.ndarray]:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        downloaded = download_audio(youtube_url, temp_dir)
        track_duration = clip_duration_from_metadata(downloaded)

        clip_duration = target_duration
        if track_duration and track_duration > 0:
            clip_duration = min(target_duration, track_duration)

        offset = 0.0
        if track_duration and track_duration > clip_duration:
            max_offset = track_duration - clip_duration
            offset = random.uniform(0.0, max_offset)

        samples = load_clip(
            downloaded.path,
            sr=SAMPLE_RATE,
            duration=clip_duration,
            offset=float(offset),
        )

    base_name = f"yt_{downloaded.path.stem}_{int(time.time())}"
    return base_name, samples


def record_microphone(max_duration: float) -> Tuple[str, np.ndarray]:
    if sd is None:
        raise RuntimeError(
            "sounddevice is not available; recording cannot be performed."
        )

    try:
        duration_text = input(
            f"Enter recording duration in seconds (max {max_duration}, default {max_duration}): "
        ).strip()
        duration = float(duration_text) if duration_text else max_duration
    except ValueError:
        duration = max_duration

    duration = max(0.1, min(max_duration, duration))
    print("Recording... Press Ctrl+C to cancel.")
    try:
        recording = sd.rec(
            int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32"
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        raise RuntimeError("Recording cancelled.")

    samples = np.squeeze(recording)
    elapsed = samples.size / SAMPLE_RATE
    if elapsed < duration:
        duration = elapsed
    base_name = f"mic_{int(time.time())}"
    return base_name, samples


def save_artifacts(
    base_name: str, samples: np.ndarray, label: str = ""
) -> Tuple[Path, Path, Dict[str, object]]:
    ensure_output_dirs()
    audio_path = AUDIO_OUTPUT_DIR / f"{base_name}.wav"
    image_path = IMAGE_OUTPUT_DIR / f"{base_name}.png"

    export_wav(samples, SAMPLE_RATE, audio_path)
    features = build_feature_row(audio_path.name, samples, SAMPLE_RATE, label)
    create_spectrogram(samples, SAMPLE_RATE, image_path)
    return audio_path, image_path, features


def feature_vector_from_row(feature_row: Dict[str, object]) -> np.ndarray:
    return np.array([float(feature_row[column]) for column in FEATURE_VECTOR_COLUMNS])


def append_features(dataset_path: Path, feature_row: Dict[str, object]) -> None:
    append_to_csv(dataset_path, FEATURE_HEADER, feature_row)


def format_probabilities(id_to_label: Dict[int, str], probs: np.ndarray) -> str:
    pairs = [f"{id_to_label[idx]}: {prob:.2%}" for idx, prob in enumerate(probs)]
    return ", ".join(pairs)


def prompt_true_label(predicted: str) -> str:
    while True:
        answer = input(
            "Was the prediction correct? (y/n): "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return predicted
        if answer in {"n", "no"}:
            label = input("Enter the correct label: ").strip()
            if label:
                return label
        else:
            print("Please answer with 'y' or 'n'.")


def prompt_retrain() -> bool:
    answer = input("Retrain the model with the updated dataset? (y/n): ").strip().lower()
    return answer in {"y", "yes"}


def run_session() -> None:
    try:
        dataset = load_dataset(DATASET_PATH)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        sys.exit(1)

    model_state = train_model(dataset)
    print("Model ready. Available labels:", ", ".join(sorted(model_state.label_to_id)))

    while True:
        choice = input(
            "Select an option - [y] YouTube, [r] Record, [q] Quit: "
        ).strip().lower()

        if choice in {"q", "quit"}:
            print("Goodbye!")
            return

        try:
            if choice in {"y", "youtube"}:
                youtube_url = input("Enter YouTube URL: ").strip()
                if not youtube_url:
                    print("No URL provided.")
                    continue
                base_name, samples = extract_youtube_samples(
                    youtube_url, target_duration=MAX_CLIP_DURATION
                )
            elif choice in {"r", "record"}:
                base_name, samples = record_microphone(MAX_CLIP_DURATION)
            else:
                print("Unrecognized option. Try again.")
                continue
        except Exception as exc:
            print(f"Failed to acquire audio: {exc}")
            continue

        audio_path, image_path, feature_row = save_artifacts(base_name, samples, label="")
        feature_vector = feature_vector_from_row(feature_row)

        prediction, probabilities = predict_label(model_state, feature_vector)
        print(f"Prediction: {prediction}")
        print("Confidence:", format_probabilities(model_state.id_to_label, probabilities))

        true_label = prompt_true_label(prediction)
        feature_row["label"] = true_label
        append_features(DATASET_PATH, feature_row)
        print("Saved artifacts:")
        print(f"  Audio: {audio_path}")
        print(f"  Spectrogram: {image_path}")
        print(f"  Features appended to: {DATASET_PATH}")

        if prompt_retrain():
            try:
                dataset = load_dataset(DATASET_PATH)
                model_state = train_model(dataset)
                print(
                    "Retraining complete. Updated labels:",
                    ", ".join(sorted(model_state.label_to_id)),
                )
            except Exception as exc:
                print(f"Retraining failed: {exc}")


def main() -> None:
    try:
        run_session()
    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")


if __name__ == "__main__":
    main()
