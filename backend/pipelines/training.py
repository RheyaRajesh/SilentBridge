"""
Training pipeline for the SignLanguageLSTM model.

Handles:
    1. Data collection — storing labeled keypoint sequences to disk
    2. Dataset building — loading collected data into PyTorch tensors
    3. Training loop — training the LSTM model with progress tracking
    4. Model saving — persisting trained weights and vocabulary
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from backend.models.sign_lstm import SignLanguageLSTM, create_model, save_model

# Constants
SEQ_LENGTH = 30
INPUT_DIM = 162

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
COLLECTED_DIR = os.path.join(DATA_DIR, "collected")
MODELS_DIR = os.path.join(DATA_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_lstm.pt")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")

# Ensure directories exist
os.makedirs(COLLECTED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class SignGestureDataset(Dataset):
    """PyTorch dataset for collected sign gesture sequences."""

    def __init__(self, sequences: list, labels: list):
        """
        Args:
            sequences: List of numpy arrays, each (SEQ_LENGTH, INPUT_DIM)
            labels: List of integer class indices
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def collect_sample(label: str, keypoint_sequence: list) -> dict:
    """
    Store a single labeled keypoint sequence to disk.

    Args:
        label: Text label for the gesture (e.g., "hello", "thank_you").
        keypoint_sequence: List of `SEQ_LENGTH` frames, each a list of `INPUT_DIM` floats.

    Returns:
        Status dict with sample count information.
    """
    seq_array = np.array(keypoint_sequence, dtype=np.float32)

    if seq_array.shape != (SEQ_LENGTH, INPUT_DIM):
        raise ValueError(
            f"Invalid sequence shape: {seq_array.shape}. "
            f"Expected ({SEQ_LENGTH}, {INPUT_DIM})."
        )

    # Create label directory
    label_dir = os.path.join(COLLECTED_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Save as .npy file with timestamp
    timestamp = int(time.time() * 1000)
    filepath = os.path.join(label_dir, f"{timestamp}.npy")
    np.save(filepath, seq_array)

    # Count total samples for this label
    sample_count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])

    return {
        "status": "collected",
        "label": label,
        "sample_count": sample_count,
        "filepath": filepath,
    }


def get_collection_stats() -> dict:
    """Get statistics about collected training data."""
    stats = {}
    if not os.path.exists(COLLECTED_DIR):
        return stats

    for label_name in os.listdir(COLLECTED_DIR):
        label_dir = os.path.join(COLLECTED_DIR, label_name)
        if os.path.isdir(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
            stats[label_name] = count

    return stats


def _load_collected_data() -> tuple:
    """
    Load all collected data from disk.

    Returns:
        (sequences, labels, vocab)
        sequences: list of numpy arrays (SEQ_LENGTH, INPUT_DIM)
        labels: list of integer class indices
        vocab: dict {label_str: index_int}
    """
    sequences = []
    labels = []
    vocab = {}

    if not os.path.exists(COLLECTED_DIR):
        return sequences, labels, vocab

    label_dirs = sorted(
        [d for d in os.listdir(COLLECTED_DIR)
         if os.path.isdir(os.path.join(COLLECTED_DIR, d))]
    )

    for idx, label_name in enumerate(label_dirs):
        vocab[label_name] = idx
        label_dir = os.path.join(COLLECTED_DIR, label_name)
        npy_files = [f for f in os.listdir(label_dir) if f.endswith(".npy")]

        for npy_file in npy_files:
            filepath = os.path.join(label_dir, npy_file)
            try:
                seq = np.load(filepath)
                if seq.shape == (SEQ_LENGTH, INPUT_DIM):
                    sequences.append(seq)
                    labels.append(idx)
                else:
                    print(f"[Training] Skipping {filepath}: shape {seq.shape}")
            except Exception as e:
                print(f"[Training] Error loading {filepath}: {e}")

    return sequences, labels, vocab


def train_model(
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    progress_callback=None,
) -> dict:
    """
    Train the SignLanguageLSTM model on collected data.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for Adam optimizer.
        progress_callback: Optional callable(epoch, loss, accuracy) for progress updates.

    Returns:
        Training results dict with final metrics.
    """
    # Load data
    sequences, labels, vocab = _load_collected_data()

    if len(sequences) == 0:
        return {
            "status": "error",
            "message": "No training data collected. Use the data collection mode first.",
        }

    num_classes = len(vocab)
    if num_classes < 2:
        return {
            "status": "error",
            "message": f"Need at least 2 gesture classes. Currently have {num_classes}.",
        }

    total_samples = len(sequences)
    print(f"[Training] Starting training: {total_samples} samples, {num_classes} classes")
    print(f"[Training] Vocabulary: {vocab}")

    # Create dataset and dataloader
    dataset = SignGestureDataset(sequences, labels)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Create model
    device = "cpu"
    model = create_model(num_classes=num_classes, device=device)
    model.train()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    history = {"loss": [], "accuracy": []}
    best_accuracy = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients in LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        scheduler.step()

        avg_loss = epoch_loss / total
        accuracy = correct / total
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            save_model(model, MODEL_PATH, vocab)
            # Also save vocab as JSON for easy inspection
            with open(VOCAB_PATH, "w") as f:
                json.dump(vocab, f, indent=2)

        if progress_callback:
            progress_callback(epoch + 1, avg_loss, accuracy)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"[Training] Epoch {epoch + 1}/{epochs} — "
                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

    # Final save (in case best wasn't the last)
    if best_accuracy == 0:
        save_model(model, MODEL_PATH, vocab)
        with open(VOCAB_PATH, "w") as f:
            json.dump(vocab, f, indent=2)

    results = {
        "status": "complete",
        "epochs": epochs,
        "final_loss": history["loss"][-1],
        "final_accuracy": history["accuracy"][-1],
        "best_accuracy": best_accuracy,
        "num_classes": num_classes,
        "total_samples": total_samples,
        "vocabulary": vocab,
        "model_path": MODEL_PATH,
    }

    print(f"[Training] Complete — Best accuracy: {best_accuracy:.4f}")
    return results


def delete_collected_data(label: str = None) -> dict:
    """
    Delete collected training data.

    Args:
        label: If specified, delete data for this label only. If None, delete all.

    Returns:
        Status dict.
    """
    import shutil

    if label:
        label_dir = os.path.join(COLLECTED_DIR, label)
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
            return {"status": "deleted", "label": label}
        return {"status": "not_found", "label": label}
    else:
        if os.path.exists(COLLECTED_DIR):
            shutil.rmtree(COLLECTED_DIR)
            os.makedirs(COLLECTED_DIR, exist_ok=True)
        return {"status": "all_deleted"}
