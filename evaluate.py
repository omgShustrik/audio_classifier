#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:25:56 2018

@author: Rustem
"""
import hyper_params as hparams
import numpy as np
import utils
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import argparse
import os
import os.path


def get_files(test_data_dir):
    """
    Get all files from path
    """
    list_of_files = os.listdir(test_data_dir)
    return sorted(list_of_files)


def preprocess_one(wav):
    """
    Devide wav to chunks
    """
    chunked_features = []
    for (start, end) in utils.windows(wav, hparams.window_size):
        chunk = wav[start:end]
        if(len(chunk) != hparams.window_size):
            chunk = utils.pad_chunk(chunk, wav)
        mel_spectrogram = utils.melspectrogram(chunk).astype(np.float32)
        chunked_features.append(mel_spectrogram.T)
    return np.array(chunked_features)


def get_text_label(label):
    labels_map = {0: 'background',
                  1: 'bags',
                  2: 'door',
                  3: 'keyboard',
                  4: 'knocking_door',
                  5: 'ring',
                  6: 'speech',
                  7: 'tool'}
    return labels_map.get(label)


def RNN(x, weight, bias):
    cell = rnn_cell.LSTMCell(
        hparams.n_hidden, state_is_tuple=True, name="init_cell")
    cell = rnn_cell.MultiRNNCell([cell])
    output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2], name="output")
    last = tf.gather(output, int(output.get_shape()[0]) - 1, name="last")
    return tf.nn.softmax(tf.matmul(last, weight) + bias, name="softmax")


def evaluate(args):
    x = tf.placeholder("float", [None, hparams.n_steps, hparams.n_input], name="x")

    bias = tf.Variable(tf.random_normal([hparams.n_classes]), name="bias")
    weight = tf.Variable(tf.truncated_normal( [hparams.n_hidden, 
                                               hparams.n_classes], 
                                               stddev=0.1), name="weights")
    prediction = RNN(x, weight, bias)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, args.path_to_model)

    list_of_files = get_files(args.test_data_dir)

    for file in list_of_files:
        wav = utils.remove_all_silence(utils.load_wav(
                            os.path.join(args.test_data_dir, '%s' % file)))
        features = preprocess_one(wav)
        pred = sess.run(prediction, feed_dict={x: features})
        # Prediction for one example
        join_pred = np.around(np.sum(pred, axis=0)/pred.shape[0], decimals=3)
        # Write to file
        with open("result.txt", "a") as f:
            f.write(file + '\t' + '{0:.3f}'.format(np.max(join_pred)) +
                    '\t' + get_text_label(np.argmax(join_pred)) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', required=True)
    parser.add_argument('--path_to_model', required=True)
    args = parser.parse_args()
    evaluate(args)
    return


if __name__ == "__main__":
    main()