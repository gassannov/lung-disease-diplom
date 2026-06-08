"""Log-mel spectrogram feature extraction."""

import torch
import torchaudio


class LogMelExtractor:
    """Extract normalized log-mel spectrograms from fixed-length waveforms.

    Example:
        extractor = LogMelExtractor(16000, 128, 25, 10)
        spectrogram = extractor(waveform)
    """

    def __init__(self, sample_rate: int, mel_bins: int, window_ms: int, hop_ms: int) -> None:
        """Initialize the log-mel extractor.

        Args:
            sample_rate: Audio sample rate.
            mel_bins: Number of mel bins.
            window_ms: Window size in milliseconds.
            hop_ms: Hop size in milliseconds.

        Returns:
            None.
        """
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(sample_rate * window_ms / 1000),
            win_length=int(sample_rate * window_ms / 1000),
            hop_length=int(sample_rate * hop_ms / 1000),
            n_mels=mel_bins,
            power=2.0,
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a normalized log-mel spectrogram.

        Args:
            waveform: Mono waveform tensor.

        Returns:
            Tensor with shape [1, mel_bins, frames].
        """
        spectrogram = self.transform(waveform)
        spectrogram = torch.log(spectrogram.clamp_min(1e-6))
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std().clamp_min(1e-6)
        return spectrogram.unsqueeze(0)

