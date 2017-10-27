import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


NUM_CHANNELS=1
# NUM_LABELS=10

CONV1_DEEP=32
CONV1_SIZE=5

CONV2_DEEP=64
CONV2_SIZE=5

FC_SIZE=1024

# 给定CNN输入和所有参数，计算前向传播
def inference(input_tensor,train,regularizer,OUTPUT_NODE):
    with tf.variable_scope('mnist_layer1-conv1'):
        conv1_weights=tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.1))

        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        # print("relu1.shape",relu1.get_shape())

    with tf.variable_scope('mnist_layer2-pool1'):
        pool1=tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # print("pool1.shape", pool1.get_shape())
    with tf.variable_scope('mnist_layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.1))

        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('mnist_layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    pool_shape=pool2.get_shape().as_list()
    # print (pool_shape)
    node=pool_shape[1]*pool_shape[2]*pool_shape[3]

    reshaped=tf.reshape(pool2,[-1,node])

    with tf.variable_scope('mnist_layer5-fc1'):
        fc1_weights=tf.get_variable("weights",[node,FC_SIZE],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases=tf.get_variable("bias",[FC_SIZE],
                                   initializer=tf.constant_initializer(0.1))

        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)

        if train:
            fc1=tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('mnist_layer6-fc2'):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))

        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        logit = tf.nn.softmax(tf.matmul(fc1, fc2_weights) + fc2_biases)
        # print (logit.get_shape())
    return logit
