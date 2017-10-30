import time
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import train
import inference
import numpy as np


EVAL_INTERVAL_SECS=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None,
                                        train.IMAGE_SIZE1,
                                        train.IMAGE_SIZE2,
                                        inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, train.OUTPUT_NODE], name='y-input')
        validate_x,validate_y=mnist.test.images, mnist.test.labels

        reshaped_validate_x = np.reshape(validate_x, [-1,
                                      train.IMAGE_SIZE1,
                                      train.IMAGE_SIZE2,
                                      inference.NUM_CHANNELS])

        y=inference.inference(x,train=False,regularizer=None,OUTPUT_NODE=train.OUTPUT_NODE)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        variable_averages=tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)

        # while True:
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step=ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                accuracy_score=sess.run(accuracy,feed_dict={x:reshaped_validate_x,y_:validate_y})
                print ("After %s training step, validation accuracy=%g" %(global_step,accuracy_score))
            else:
                print ("No checkpoint file found")
                return time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    data=read_data_sets('MNIST_data', one_hot=True)
    evaluate(data)

if __name__=="__main__":
    tf.app.run()


##TODO:  Result
# After 30000 training step, validation accuracy=0.9951

