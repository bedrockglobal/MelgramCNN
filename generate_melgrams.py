import numpy as np
from librosa.core import stft
from librosa.feature import melspectrogram
from librosa import power_to_db


def mel_spectrogram(signal, size, hop_length, window, n_mels):
    """
    Computes the Short Time Fourier Transform (STFT) Spectrogram and converts to mel-frequency.
    :param signal: An array of size N to compute the STFT on.
    :param size: FFT window size (scalar)
    :param hop_length: Number (scalar) of frames between STFT columns.
    :param window: (string) A window specification (usually hanning)
    :param n_mels: (integer) number of mel-filter banks to apply to spectrogram. This parameter also determines the
                             dimension of the y dimension of the mel-spectrogram.
    :return: A numpy array of shape (n_mels, len(signal))
    """
    stft_out = stft(signal.astype(float), n_fft=size, hop_length=hop_length, window=window)
    s = np.abs(stft_out)  # spectrogram
    s_mel = melspectrogram(S=s**2, n_mels=n_mels)  # mel-spectrogram
    s_mel = power_to_db(s_mel, ref=np.max)  # normalization
    return s_mel