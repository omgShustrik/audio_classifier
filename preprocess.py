#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 23:14:46 2018

@author: Rustem
"""
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import utils
import hyper_params as hparams
import numpy as np
import os
import os.path

import argparse


def preprocess_data(args):
    """
    The entry point to processing and augmentation.
    Augmentation to the largest number of samples of the same class.
    """
    output_dir = os.path.join(os.getcwd(), args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    metadata = preprocess_from_path(args.dataset_dir,
                                    args.metadata_filename,
                                    output_dir,
                                    args.num_workers,
                                    tqdm=tqdm)
    all_labels = []
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding="utf-8") as f:
        for m in metadata:
            all_labels.append(m[-1])
            f.write('|'.join([str(x) for x in m]) + '\n')
    number_of_exaples = {}

    for i in range(hparams.num_classes):
        number_of_exaples[i] = all_labels.count(i)

    for l, c in number_of_exaples.items():
        delta = max(number_of_exaples.values()) - c
        if delta != 0:
            metadata = augmantation_from_path(args.dataset_dir,
                                              args.metadata_filename,
                                              output_dir, l, delta,
                                              args.num_workers,
                                              tqdm=tqdm)
            with open(os.path.join(output_dir, 'train.txt'), 'a', encoding="utf-8") as f:
                for m in metadata:
                    all_labels.append(m[-1])
                    f.write('|'.join([str(x) for x in m]) + '\n')
    return


def get_label_number(label):
    labels_map = {'background': 0,
                  'bags': 1,
                  'door': 2,
                  'keyboard': 3,
                  'knocking_door': 4,
                  'ring': 5,
                  'speech': 6,
                  'tool': 7}
    return labels_map.get(label)


def preprocess_from_path(dataset_dir, metadata_filename, output_dir, num_workers=1, tqdm=lambda x: x):
    """
    Preprocessing wav step by step
    Load -> Remove silences -> Divide to chunks -> Extract features
    Return: list of metadata samples
    """
    print("Start preprocess_from_path...")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    with open(os.path.join(dataset_dir, metadata_filename), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            wav_name = parts[0]
            target_class = get_label_number(parts[-1])
            wav_path = os.path.join(
                dataset_dir, 'audio', '%s' % wav_name)  # test_audio
            if os.path.isfile(wav_path):
                wav = utils.remove_all_silence(utils.load_wav(wav_path))
                index = 0
                for (start, end) in utils.windows(wav, hparams.window_size):
                    chunk = wav[start:end]
                    if(len(chunk) != hparams.window_size):
                        chunk = utils.pad_chunk(chunk, wav)
                    futures.append(executor.submit(partial(_process_utterance,
                                                           output_dir,
                                                           chunk,
                                                           wav_name,
                                                           target_class,
                                                           index)))
                    index += 1
    results = [future.result() for future in tqdm(futures)]
    return [r for r in results if r is not None]


def augmantation_from_path(dataset_dir, metadata_filename, output_dir, current_class, augmantation_amount, num_workers=1, tqdm=lambda x: x):
    """
    Preprocessing wav step by step
    Load -> Remove silences -> Random start -> Divide to chunks -> Extract features
    Return: list of metadata samples
    """
    print("Start augmantation_from_path...")
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    aug_index = 0
    while_loop = True
    while while_loop:
        with open(os.path.join(dataset_dir, metadata_filename), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                wav_name = parts[0]
                target_class = get_label_number(parts[-1])
                if target_class == current_class:
                    wav_path = os.path.join(
                        dataset_dir, 'audio', '%s' % wav_name)  # audio
                    if os.path.isfile(wav_path):
                        wav = utils.remove_all_silence(
                            utils.load_wav(wav_path))
                        wav = wav[np.random.randint(1, 22050):]
                        index = 0
                        for (start, end) in utils.windows(wav, hparams.window_size):
                            if aug_index >= augmantation_amount:
                                while_loop = False
                                break
                            chunk = wav[start:end]
                            if(len(chunk) != hparams.window_size):
                                chunk = utils.pad_chunk(chunk, wav)
                            futures.append(executor.submit(partial(_process_utterance,
                                                                   output_dir,
                                                                   chunk,
                                                                   "aug-%s-%s" % (aug_index, wav_name),
                                                                   target_class,
                                                                   index)))
                            index += 1
                            aug_index += 1
    results = [future.result() for future in tqdm(futures)]
    return [r for r in results if r is not None]


def _process_utterance(output_dir, chunk, wav_name, target_class, index):
    mel_spectrogram = utils.melspectrogram(chunk).astype(np.float32)
    mel_filename = "mel-%s-%s.npy" % (wav_name.split('.')[0], index)
    trimmed_wav_name = "%s-%s.wav" % (wav_name.split('.')[0], index)
    utils.save_feature(mel_spectrogram.T, os.path.join(output_dir, mel_filename))
    #utils.save_wav(chunk, os.path.join(output_dir, trimmed_wav_name))
    return (trimmed_wav_name, mel_filename, target_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--metadata_filename', default='meta.txt')
    parser.add_argument('--output_folder', default='training')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    preprocess_data(args)
    return


if __name__ == "__main__":
    main()