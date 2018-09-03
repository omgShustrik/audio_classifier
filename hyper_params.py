#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 23:34:52 2018

@author: Rustem
"""
# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20050
frame_length_ms = 50.
frame_shift_ms = 12.5
trim_treshold = 20 #db
window_size = 51200 
window_size_s = 2.5 
min_level_db = -110#150
ref_level_db = 20#70

#Model
n_classes = 8
learning_rate = 0.001
training_iters = 10
batch_size = 64
display_step = 50

# Network Parameters
n_input = 80 
n_steps = 205
n_hidden = 300
