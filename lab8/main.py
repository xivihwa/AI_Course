
#!/usr/bin/env python3

import os
import sys
import re
from collections import Counter
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import warnings

warnings.filterwarnings("ignore")

from jiwer import wer, cer
import textdistance
from spellchecker import SpellChecker


BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/kaggle/input/lab8-data/LJSpeech-1.1"
NUM_WORKERS = 0
SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 160
N_MFCC = 13


def download_ljspeech_if_missing(data_path: str = DATA_DIR) -> str:
    if os.path.exists(data_path):
        return data_path

    print("Dataset folder not found. Attempting to download LJSpeech (best-effort)...")
    if os.system("which wget > /dev/null") == 0:
        os.system("wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -q")
        os.system("tar -xjf LJSpeech-1.1.tar.bz2")
        if os.path.exists("LJSpeech-1.1"):
            return "LJSpeech-1.1"
    raise FileNotFoundError(
        f"LJSpeech dataset not found at '{data_path}'. "
        "Please download LJSpeech and set DATA_DIR accordingly."
    )


def load_ljspeech_metadata(data_path: str) -> pd.DataFrame:
    metadata_path = os.path.join(data_path, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    rows = []
    with open(metadata_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                fname, _, text = parts[:3]
                audio_path = os.path.join(data_path, "wavs", f"{fname}.wav")
                if os.path.exists(audio_path):
                    rows.append({"audio_path": audio_path, "text": text})
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} audio files from LJSpeech metadata.")
    return df


class AudioProcessor:
    def __init__(self, sr: int = SR, n_mels: int = N_MELS, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load(self, path: str) -> np.ndarray:
        audio, sr = librosa.load(path, sr=self.sr)
        return audio

    def extract_log_mel(self, audio: np.ndarray) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel.T

    def process(self, path: str) -> np.ndarray:
        audio = self.load(path)
        return self.extract_log_mel(audio)


class TextTransform:
    def __init__(self):
        self.chars = [" ", "'"] + list("abcdefghijklmnopqrstuvwxyz")
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.blank_idx = len(self.chars)

    def text_to_indices(self, text: str) -> List[int]:
        text = text.lower()
        indices = [self.char_to_idx.get(ch, self.char_to_idx[" "]) for ch in text if ch in self.char_to_idx]
        return indices

    def indices_to_text(self, indices: List[int]) -> str:
        return "".join(self.idx_to_char.get(i, "") for i in indices)

    def vocab_size(self) -> int:
        return len(self.chars) + 1


class LJSpeechDataset(Dataset):
    def __init__(self, df: pd.DataFrame, audio_proc: AudioProcessor, text_transform: TextTransform):
        self.df = df.reset_index(drop=True)
        self.audio_proc = audio_proc
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        features = self.audio_proc.process(row["audio_path"])
        text_indices = self.text_transform.text_to_indices(row["text"])
        return torch.FloatTensor(features), torch.LongTensor(text_indices), features.shape[0], len(text_indices)


def collate_fn(batch):
    feats, texts, feat_lens, text_lens = zip(*batch)
    feats_padded = pad_sequence(feats, batch_first=True)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    return feats_padded, texts_padded, torch.LongTensor(feat_lens), torch.LongTensor(text_lens)


class DeepSpeech2Like(nn.Module):
    def __init__(self, n_mels: int, n_classes: int, n_rnn_layers: int = 5, rnn_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Dropout(dropout),
        )

        self._conv_out_size = self._estimate_conv_output(n_mels)

        self.rnn_layers = nn.ModuleList()
        for i in range(n_rnn_layers):
            input_size = self._conv_out_size if i == 0 else rnn_hidden * 2
            self.rnn_layers.append(nn.GRU(input_size, rnn_hidden, num_layers=1, bidirectional=True, batch_first=True))

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(rnn_hidden * 2) for _ in range(n_rnn_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, n_classes)

    def _estimate_conv_output(self, n_mels: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, 100, n_mels)
            x = self.conv(x)
            b, c, t, f = x.size()
            return c * f

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, channels, time, features = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, time, channels * features)

        output_lengths = (lengths.float() / 4.0).ceil().long()
        for i, rnn in enumerate(self.rnn_layers):
            x_packed = pack_padded_sequence(x, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
            x_packed, _ = rnn(x_packed)
            x, _ = pad_packed_sequence(x_packed, batch_first=True)

            x = x.transpose(1, 2)
            x = self.batch_norms[i](x)
            x = x.transpose(1, 2)
            x = self.dropout(x)

        x = self.fc(x)
        x = nn.functional.log_softmax(x, dim=-1)
        x = x.transpose(0, 1)
        return x, output_lengths


def train_network(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    text_transform: TextTransform,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    device: str = DEVICE,
):
    model = model.to(device)
    criterion = nn.CTCLoss(blank=text_transform.blank_idx, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (feats, texts, feat_lens, text_lens) in enumerate(train_loader, 1):
            feats = feats.to(device)
            texts = texts.to(device)
            feat_lens = feat_lens.to(device)
            text_lens = text_lens.to(device)

            optimizer.zero_grad()
            outputs, out_lens = model(feats, feat_lens)

            loss = criterion(outputs, texts, out_lens, text_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, texts, feat_lens, text_lens in val_loader:
                feats = feats.to(device)
                texts = texts.to(device)
                feat_lens = feat_lens.to(device)
                text_lens = text_lens.to(device)
                outputs, out_lens = model(feats, feat_lens)
                loss = criterion(outputs, texts, out_lens, text_lens)
                val_loss += loss.item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train Loss: {avg_train:.4f}")
        print(f"  Val   Loss: {avg_val:.4f}")

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "deepspeech2_best.pth")
            print("Saved best model to deepspeech2_best.pth")

        print("-" * 60)

    return model


class GreedyDecoder:
    def __init__(self, text_transform: TextTransform):
        self.text_transform = text_transform
        self.blank = text_transform.blank_idx

    def decode(self, log_probs: torch.Tensor) -> List[str]:
        argmax = torch.argmax(log_probs, dim=-1)
        batch_size = argmax.size(1)
        decoded = []
        for b in range(batch_size):
            indices = argmax[:, b].cpu().numpy().tolist()
            prev = None
            chars = []
            for idx in indices:
                if idx != prev and idx != self.blank:
                    if idx < len(self.text_transform.idx_to_char):
                        chars.append(self.text_transform.idx_to_char[idx])
                prev = idx
            decoded.append("".join(chars))
        return decoded


def test_network(model: nn.Module, test_loader: DataLoader, text_transform: TextTransform, device: str = DEVICE):
    model = model.to(device)
    model.eval()
    decoder = GreedyDecoder(text_transform)
    predictions = []
    targets = []

    with torch.no_grad():
        for feats, texts, feat_lens, text_lens in test_loader:
            feats = feats.to(device)
            outputs, out_lens = model(feats, feat_lens)
            batch_preds = decoder.decode(outputs)
            for i in range(texts.size(0)):
                target_idx = texts[i, : text_lens[i]].cpu().numpy().tolist()
                targets.append(text_transform.indices_to_text(target_idx))
            predictions.extend(batch_preds)

    wer_score = wer(targets, predictions)
    cer_score = cer(targets, predictions)

    print("\nTest results:")
    print(f"  WER: {wer_score * 100:.2f}%")
    print(f"  CER: {cer_score * 100:.2f}%")

    print("\nExample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"\nTarget    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")

    return predictions, targets, wer_score


class TextCorrector:

    def __init__(self):
        self.spell = SpellChecker()
        self.word_freq = Counter()

    def fit_corpus(self, texts: List[str]):
        for t in texts:
            words = t.lower().split()
            self.word_freq.update(words)

    def spell_check(self, text: str) -> str:
        words = text.split()
        out = []
        for w in words:
            if w in self.spell:
                out.append(w)
            else:
                c = self.spell.correction(w)
                out.append(c if c else w)
        return " ".join(out)

    def edit_distance_correct(self, text: str, vocabulary: Set[str], max_distance: int = 2) -> str:
        words = text.split()
        out = []
        for w in words:
            if w in vocabulary:
                out.append(w)
            else:
                best = min(vocabulary, key=lambda v: textdistance.levenshtein(w, v))
                if textdistance.levenshtein(w, best) <= max_distance:
                    out.append(best)
                else:
                    out.append(w)
        return " ".join(out)

    def frequency_correct(self, text: str) -> str:
        words = text.split()
        out = []
        for w in words:
            if w in self.word_freq:
                out.append(w)
            else:
                candidates = [v for v in self.word_freq.keys() if textdistance.levenshtein(w, v) <= 2]
                if candidates:
                    best = max(candidates, key=lambda v: self.word_freq[v])
                    out.append(best)
                else:
                    out.append(w)
        return " ".join(out)

    @staticmethod
    def regex_clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        return text


def compare_corrections(preds: List[str], targets: List[str], corrector: TextCorrector, vocabulary: Set[str]):
    print("\n" + "=" * 80)
    print("COMPARING ERROR-CORRECTION METHODS")
    print("=" * 80)
    methods = {
        "No correction": preds,
        "SpellChecker": [corrector.spell_check(p) for p in preds],
        "EditDistance": [corrector.edit_distance_correct(p, vocabulary) for p in preds],
        "Frequency": [corrector.frequency_correct(p) for p in preds],
        "Regex": [TextCorrector.regex_clean(p) for p in preds],
    }

    results = {}
    for name, corrected in methods.items():
        w = wer(targets, corrected)
        c = cer(targets, corrected)
        results[name] = {"WER": w * 100, "CER": c * 100}
        print(f"\n{name}:")
        print(f"  WER: {w * 100:.2f}%")
        print(f"  CER: {c * 100:.2f}%")

    best = min(results.items(), key=lambda kv: kv[1]["WER"])
    print("\n" + "=" * 80)
    print(f"Best method: {best[0]} (WER: {best[1]['WER']:.2f}%)")
    print("=" * 80)
    return results


def main():
    print("=" * 80)
    print("DeepSpeech2-like training for LJSpeech")
    print("=" * 80)
    print(f"Using device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    print("-" * 80)

    dataset_path = download_ljspeech_if_missing(DATA_DIR)
    df = load_ljspeech_metadata(dataset_path)

    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        print("scikit-learn not available. Please install scikit-learn for train/test splitting.")
        sys.exit(1)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    audio_proc = AudioProcessor(sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    text_transform = TextTransform()
    print(f"Vocabulary size (characters + blank): {text_transform.vocab_size()}")

    train_dataset = LJSpeechDataset(train_df, audio_proc, text_transform)
    val_dataset = LJSpeechDataset(val_df, audio_proc, text_transform)
    test_dataset = LJSpeechDataset(test_df, audio_proc, text_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = DeepSpeech2Like(n_mels=N_MELS, n_classes=text_transform.vocab_size(), n_rnn_layers=5, rnn_hidden=512, dropout=0.1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    model = train_network(model, train_loader, val_loader, text_transform, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE)

    if os.path.exists("deepspeech2_best.pth"):
        model.load_state_dict(torch.load("deepspeech2_best.pth", map_location=DEVICE))
    else:
        print("Warning: saved model not found (deepspeech2_best.pth). Using current model state for testing.")

    preds, targets, test_wer = test_network(model, test_loader, text_transform, device=DEVICE)

    corrector = TextCorrector()
    print("Training text corrector on train corpus...")
    corrector.fit_corpus(train_df["text"].tolist())
    vocab = set(" ".join(train_df["text"].tolist()).lower().split())

    results = compare_corrections(preds, targets, corrector, vocab)

    print("\nAll done.")
    return model, results


if __name__ == "__main__":
    main()
