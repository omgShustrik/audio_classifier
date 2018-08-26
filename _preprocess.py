#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:14:46 2018

@author: Rustem
"""
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
from scipy.io import wavfile
import utils
import re
import matplotlib.pyplot as plt
import itertools
import matplotlib
import hyper_params as hparams
import numpy as np
import os
import os.path
import librosa

import argparse


def preprocess_data(args):
    dataset_name = os.path.basename(args.dataset_dir)
    output_dir = os.path.join(os.getcwd(), args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    metadata = preprocess_from_path(args.dataset_dir, args.metadata_filename, output_dir, args.num_workers, tqdm=tqdm)
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding="utf-8") as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')    
    return


def preprocess_from_path(dataset_dir, metadata_filename, output_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    with open(os.path.join(dataset_dir, metadata_filename), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            wav_name = parts[0]
            target_class = parts[-1]
            wav_path = os.path.join(dataset_dir, 'audio', '%s' % wav_name) #test_audio
            futures.append(executor.submit(partial(_process_utterance, output_dir, wav_path, wav_name, target_class)))
    results = [future.result() for future in tqdm(futures)]
    return [r for r in results if r is not None]


def _process_utterance(output_dir, wav_path, wav_name, target_class):
    if os.path.isfile(wav_path):
        wav = utils.load_wav(wav_path)
        trimmed_wav = utils.trim_wav(wav)[0:5*22050]
        mel_spectrogram = utils.melspectrogram(trimmed_wav).astype(np.float32)
        mel_filename = "mel-%s.npy" % wav_name.split('.')[0]
        utils.save_feature(mel_spectrogram.T, os.path.join(output_dir, mel_filename))
        utils.save_wav(trimmed_wav, os.path.join(output_dir, wav_name))
        return (wav_name, mel_filename, target_class)    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)  # folder with dataset (absolute path)
    parser.add_argument('--metadata_filename', default='meta.txt')  # dataset's metadata filename
    parser.add_argument('--output_folder', default='training')  # name of folder with preprocessed data
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    preprocess_data(args)
    return


if __name__ == "__main__":
    main()