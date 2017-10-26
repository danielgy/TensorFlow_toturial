#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/10 16:34
# @Author  : Zoe
# @Site    : 
# @File    : inference.py.py
# @Software: PyCharm Community Edition

import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10
HIDDEN1=300
def inference(input_data,keep_prob):
    with tf.variable_scope("hidden-layer"):
        weight1=tf.get_variable("weight",[INPUT_NODE,HIDDEN1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases1=tf.get_variable("bias",[HIDDEN1],
                                     initializer=tf.constant_initializer(0.1))
        hidden1=tf.nn.relu(tf.matmul(input_data,weight1)+biases1)

        hidden1_drop=tf.nn.dropout(hidden1,keep_prob)

    with tf.variable_scope("output-layer"):
        weight2 = tf.get_variable("weight", [ HIDDEN1,OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases2 = tf.get_variable("bias", [OUTPUT_NODE],
                                  initializer=tf.constant_initializer(0.1))
        logist=tf.nn.softmax(tf.matmul(hidden1_drop,weight2)+biases2)
    return logist


