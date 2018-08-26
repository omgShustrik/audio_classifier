#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 20:39:32 2018

@author: Rustem
"""
from scipy import signal
import hyper_params as hparams
import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import argparse
import os
import os.path



def extract_data(args):
    train_data_dir = args.train_data_dir
    train_filename = args.train_filename
    labels_map  = {'background': 0, 
                   'bags': 1, 
                   'door': 2, 
                   'keyboard': 3, 
                   'knocking_door': 4, 
                   'ring': 5, 
                   'speech': 6, 
                   'tool': 7}
    features = []
    labels = []
    with open(os.path.join(train_data_dir, train_filename), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('|')
            feature_name = parts[1] #mel spectogram
            label = parts[-1]
            feature = np.load(os.path.join(train_data_dir, feature_name))
            features.append(feature)
            labels.append(labels_map.get(label))
    return np.array(features), np.array(labels,dtype = np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode    
    

def RNN(x, weight, bias):
    cell = rnn_cell.LSTMCell(hparams.n_hidden, state_is_tuple = True)
    cell = rnn_cell.MultiRNNCell([cell])
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)


def train(args):
    tf.reset_default_graph()
    print("Load data...")
    features, labels = extract_data(args)
    np.random.seed([111])
    shuffled = np.random.permutation(np.arange(len(labels)))
    features_shuffled = features[shuffled]
    labels_shuffled = labels[shuffled]   
    labels_shuffled = one_hot_encode(labels_shuffled)
    
    # Split train/test set
    cutoff = int(len(labels_shuffled)*0.9)
    _features, ts_features = features_shuffled[:cutoff], features_shuffled[cutoff:]
    _labels, ts_labels = labels_shuffled[:cutoff], labels_shuffled[cutoff:]
        
    x = tf.placeholder("float", [None, hparams.n_steps, hparams.n_input])
    y = tf.placeholder("float", [None, hparams.n_classes])
    
    #weight = tf.Variable(tf.random_normal([hparams.n_hidden, hparams.n_classes]))
    bias = tf.Variable(tf.random_normal([hparams.n_classes]))    
    weight = tf.Variable(tf.truncated_normal([hparams.n_hidden, hparams.n_classes], stddev=0.1))
    
    prediction = RNN(x, weight, bias)
    
    # Define loss and optimizer
    loss_f = -tf.reduce_sum(y * tf.log(prediction))
    optimizer = tf.train.AdamOptimizer(learning_rate = hparams.learning_rate).minimize(loss_f)
    
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()  
    
    
    with tf.Session() as session:
        session.run(init)
        
        for itr in range(hparams.training_iters):  
            # Shuffle training data
            shuffled = np.random.permutation(np.arange(len(_labels)))
            _features = _features[shuffled]
            _labels = _labels[shuffled]
            tr_features = _features
            tr_labels = _labels
            if itr % hparams.display_step == 0:
                cutoff = int(len(_labels)*0.9)
                tr_features, val_features = _features[:cutoff], _features[cutoff:]
                tr_labels, val_labels = _labels[:cutoff], _labels[cutoff:]
            
            
            offset = (itr * hparams.batch_size) % (tr_labels.shape[0] - hparams.batch_size)
            batch_x = tr_features[offset:(offset + hparams.batch_size), :, :]
            batch_y = tr_labels[offset:(offset + hparams.batch_size), :]
            _, c = session.run([optimizer, loss_f],feed_dict={x: batch_x, y : batch_y})
            # Calculate batch accuracy
            acc = session.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = session.run(loss_f, feed_dict={x: batch_x, y: batch_y})
            print( "Iter " + str(itr) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                
            if itr % hparams.display_step == 0:
                print('Validation accuracy: ',round(session.run(accuracy, feed_dict={x: val_features, y: val_labels}) , 3))
        
        print('Test accuracy: ',round(session.run(accuracy, feed_dict={x: ts_features, y: ts_labels}) , 3))    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', required=True)  # folder with dataset (absolute path)
    parser.add_argument('--train_filename', default='train.txt')  # dataset's metadata filename
    args = parser.parse_args()
    train(args)
    return


if __name__ == "__main__":
    main()