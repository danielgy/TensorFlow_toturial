#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/7/10 16:50
# @Author  : Zoe
# @Site    : 
# @File    : eval.py.py
# @Software: PyCharm Community Edition

import time
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import train
import inference
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

EVAL_INTERVAL_SECS=10
NUM_CHANNELS=1

def evaluate(mnist):

    x = tf.placeholder(tf.float32, [None,train.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32, [None, train.OUTPUT_NODE], name='y-input')
    validate_x,validate_y=mnist.test.images, mnist.test.labels


    y=inference.inference(x,keep_prob=1)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    saver=tf.train.Saver()

    # while True:
    with tf.Session() as sess:
        ckpt=tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
            global_step=ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
            accuracy_score=sess.run(accuracy,feed_dict={x:validate_x,y_:validate_y})
            print ("After %s training step, validation accuracy=%g" %(global_step,accuracy_score))
        else:
            print ("No checkpoint file found")
            return time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    data=read_data_sets('MNIST_data', one_hot=True)
    evaluate(data)

if __name__=="__main__":
    tf.app.run()



##TODO: RESULT:
# After 10000 training step, validation accuracy=0.9799
