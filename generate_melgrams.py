import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Mel-Spectrograms')
    parser.add_argument('trainset', type=bool, help='True if train set, else False')
    parser.add_argument('basedir', type=str, help='Input directory ')
    args = parser.parse_args()

    train_flag = args.trainset
    base_dir = args.indir # base directory containing test/train sets

    M = 4096  # window size
    RATE = 4e6  # sample rate
    HOP_LENGTH = 950  # number of points to shift for each window
    sample_size = 150000  # signal sample size we're working on
    n_mels = 64  # number of mel bands to generate
    num_samples = 1000  # number of times to sample from an experiment

    print('Generating MelSpectrograms...')
    if not train_flag:
        test_names = os.listdir(os.path.join(base_dir, 'testset'))
        for file in tqdm(test_names):
            df = pd.read_csv(base_dir + file)
            S_mel = mel_spectrogram(df['acoustic_data'].values, M, HOP_LENGTH, 'hanning', n_mels)

            file_path = os.path.join('./testset/', file[:-4] + '.jpg')
            plt.imsave(file_path, S_mel)
        sys.exit()

    # quake indexes
    quakes = np.array([5656574, 50085878, 104677356,
                       138772453, 187641820, 218652630,
                       245829585, 307838917, 338276287,
                       375377848, 419368880, 461811623,
                       495800225, 528777115, 585568144, 621985673])

    quake_length = list(np.diff(quakes))
    files = ['file_{}.csv'.format(i) for i in range(1, 16)]  # files to read in

    for i, file in enumerate(files):
        df = pd.read_csv('./split/' + file, header=None, names=['acoustic_data', 'time_to_failure'])
        random_idxs = [np.random.randint(0, quake_length[i] - sample_size) for j in range(num_samples)]
        for random_idx in random_idxs:
            start = random_idx
            end = start + sample_size
            signal = df['acoustic_data'].iloc[start:end].values
            label = df['time_to_failure'].iloc[end]
            S_mel = mel_spectrogram(signal, size=M, hop_lenght=HOP_LENGTH, window='hanning')
            fname = './trainset/{}_{}_.jpg'.format(file[:-4], label)
            plt.imsave(fname, S_mel)
        del df

    print('Done.')