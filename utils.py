#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 23:15:44 2018

@author: Rustem
"""
import hyper_params as hparams
import librosa
import librosa.filters
import numpy as np


def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    librosa.output.write_wav(path, wav, hparams.sample_rate)


def save_feature(f, path):
    np.save(path, f, allow_pickle=False)


def trim_wav(wav):
    trimmed_wav, _ = librosa.effects.trim(wav, top_db=hparams.trim_treshold)
    return trimmed_wav


def remove_all_silence(wav):
    splitted_wav = np.array([])
    event_chunks = librosa.effects.split(wav, top_db=hparams.trim_treshold)
    for start, stop in event_chunks:
        splitted_wav = np.append(splitted_wav, wav[start:stop])
    return splitted_wav


def pad_chunk(chunk, wav):
    if len(wav) < hparams.window_size:
        wav = np.pad(wav, (0, hparams.window_size - len(wav)), 'wrap')
    chunk = np.append(chunk, wav[:hparams.window_size-len(chunk)])     
    return chunk

def windows(wav, window_size):
    start = 0
    while start < len(wav):
        yield int(start), int(start + window_size)
        start += (window_size / 2)
        
        
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode  
        

def melspectrogram(wav):
    d = _stft(wav)
    s = _amp_to_db(_linear_to_mel(np.abs(d))) - hparams.ref_level_db
    return _normalize(s)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


def _linear_to_mel(spectrogram):
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _normalize(s):
    return np.clip((s - hparams.min_level_db) / -hparams.min_level_db, 0, 1)