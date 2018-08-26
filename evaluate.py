#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:25:56 2018

@author: Rustem
"""
from scipy import signal
import hyper_params as hparams
import librosa
import librosa.filters
import numpy as np
import utils
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import argparse
import os
import os.path

def get_label_from_name(file):
    labels_map  = {'background': 0, 
                   'bags': 1, 
                   'door': 2, 
                   'keyboard': 3, 
                   'knocking': 4, 
                   'ring': 5, 
                   'speech': 6, 
                   'tool': 7}
    if 'knocking' in file:
        return labels_map.get('knocking')
    if 'background' in file:
        return labels_map.get('background')
    if 'bags' in file:
        return labels_map.get('bags')
    if 'door' in file:
        return labels_map.get('door')
    if 'keyboard' in file:
        return labels_map.get('keyboard')
    if 'ring' in file:
        return labels_map.get('ring')
    if 'speech' in file:
        return labels_map.get('speech')    
    if 'tool' in file:
        return labels_map.get('tool')    
    
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

def extract_data_from_dir(args):
    test_data_dir = args.test_data_dir
    path_to_model = args.path_to_model
    wavs = []
    labels = []
    list_of_files = os.listdir(test_data_dir) 
    for file in list_of_files:
        wav = utils.trim_wav(utils.load_wav(os.path.join(test_data_dir,'%s' % file)))
        wavs.append(wav)
        labels.append(get_label_from_name(file))
    return preprocess_data(wavs, labels)

def preprocess_data(wavs, labels):
    print('Start preprocess_data')
    window_size = hparams.window * (hparams.number_of_chunks - 1)
    chunked_features = []
    cunked_labels = []
    for e in range(len(labels)):
        wav = wavs[e]
        label = labels[e]
        if len(wav) < window_size:
            'too short example'
        for (start, end) in windows(wav, window_size):
            if(len(wav[start:end]) == window_size):
                chunk = wav[start:end]
                mel_spectrogram = utils.melspectrogram(chunk).astype(np.float32)
                chunked_features.append(mel_spectrogram.T)
                cunked_labels.append(label)
    return np.array(chunked_features), np.array(cunked_labels,dtype = np.int)            
#    with open(os.path.join('./training/', 'train.txt'), encoding="utf-8") as f:
#        for line in f:
#            parts = line.strip().split('|')
#            feature_name = parts[1] #mel spectogram
#            label = parts[-1]
#            feature = np.load(os.path.join('./training/', feature_name))
#            features.append(feature)
#            labels.append(labels_map.get(label))
#    return np.array(features), np.array(labels,dtype = np.int)
#
#
def RNN(x, weight, bias):
    cell = rnn_cell.LSTMCell(hparams.n_hidden, state_is_tuple = True, name="init_cell")
    cell = rnn_cell.MultiRNNCell([cell])
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2], name="output")
    last = tf.gather(output, int(output.get_shape()[0]) - 1, name="last")
    return tf.nn.softmax(tf.matmul(last, weight) + bias, name="softmax")

def evaluate(args):
    features, labels = extract_data_from_dir(args)
    labels = one_hot_encode(labels)
    print(len(features))
    print(len(labels))    
    print('Start evaluate')  
    session = tf.Session()    
    saver = tf.train.import_meta_graph('./model.ckpt.meta')
    saver.restore(session, "./model.ckpt")
    
    
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    

    bias = graph.get_tensor_by_name("bias:0")
    weight = graph.get_tensor_by_name("weights:0")
    prediction = RNN(x, weight, bias)
    
    #confusion_matrix = tf.as_string(tf.confusion_matrix(tf.argmax(prediction,1), tf.argmax(y,1), num_classes=8, name="confusion_matrix"))
    
    #init = tf.global_variables_initializer() 
    #session.run(init)
    
    print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: features, y: labels}) , 3)) 
    #print('Test confusion matrix: ',session.run(confusion_matrix, feed_dict={x: features, y: labels}))
    #print('Test Predictions: ',session.run(prediction, feed_dict={x: features}))    
    #print('Predictions: ',session.run(prediction, feed_dict={x: features}))  
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', required=True)  # dataset's metadata filename
    parser.add_argument('--path_to_model', required=True)  # folder with dataset (absolute path)
    args = parser.parse_args()
    evaluate(args)
    return


if __name__ == "__main__":
    main()