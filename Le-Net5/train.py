import inference
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import os



BATCH_SIZE=300
LEARNING_RATE_BASE=0.01
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRANING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

model_dir = "saver"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

MODEL_SAVE_PATH=model_dir
MODEL_NAME="model.ckpt"


INPUT_NODE=784
OUTPUT_NODE=10
IMAGE_SIZE1=28
IMAGE_SIZE2=28



def train(mnist):
    x=tf.placeholder (tf.float32,[BATCH_SIZE,
                                  IMAGE_SIZE1,
                                  IMAGE_SIZE2,
                                  inference.NUM_CHANNELS],
                      name='x-input')

    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=inference.inference(x,train=True,regularizer=regularizer,OUTPUT_NODE=OUTPUT_NODE)

    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))


    train_step = tf.train.AdamOptimizer(0.0003).minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRANING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs=np.reshape(xs,[BATCH_SIZE,
                                       IMAGE_SIZE1,
                                       IMAGE_SIZE2,
                                       inference.NUM_CHANNELS])
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})

            if i%100==0:
                train_accuracy = accuracy.eval(feed_dict={x: reshaped_xs, y_: ys})
                print ("After %d training steps, loss %g, training accuracy %g"%(step,loss_value,train_accuracy))

        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main(argv=None):
    mnist=read_data_sets('MNIST_data', one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




##TODO:    training process
# After 1 training steps, loss 3.60566, training accuracy 0.09
# After 101 training steps, loss 2.78025, training accuracy 0.74
# After 201 training steps, loss 2.46405, training accuracy 0.953333
# After 301 training steps, loss 2.33461, training accuracy 0.95
# After 401 training steps, loss 2.26163, training accuracy 0.963333
# After 501 training steps, loss 2.17952, training accuracy 0.976667
# After 601 training steps, loss 2.12224, training accuracy 0.973333
# After 701 training steps, loss 2.05661, training accuracy 0.99
# After 801 training steps, loss 2.01017, training accuracy 0.983333
# After 901 training steps, loss 1.97245, training accuracy 0.973333
# After 1001 training steps, loss 1.92855, training accuracy 0.986667
# After 1101 training steps, loss 1.89036, training accuracy 0.986667
# After 1201 training steps, loss 1.85852, training accuracy 0.98
# After 1301 training steps, loss 1.83677, training accuracy 0.983333
# After 1401 training steps, loss 1.78828, training accuracy 0.996667
# After 1501 training steps, loss 1.76635, training accuracy 0.99
# After 1601 training steps, loss 1.75408, training accuracy 0.983333
# After 1701 training steps, loss 1.7237, training accuracy 0.983333
# After 1801 training steps, loss 1.71268, training accuracy 0.983333
# After 1901 training steps, loss 1.6809, training accuracy 0.993333
# After 2001 training steps, loss 1.6678, training accuracy 0.993333
# After 2101 training steps, loss 1.65615, training accuracy 0.99
# After 2201 training steps, loss 1.64559, training accuracy 0.983333
# After 2301 training steps, loss 1.62525, training accuracy 0.993333
# After 2401 training steps, loss 1.61524, training accuracy 0.993333
# After 2501 training steps, loss 1.60948, training accuracy 0.99
# After 2601 training steps, loss 1.59449, training accuracy 0.993333
# After 2701 training steps, loss 1.57716, training accuracy 1
# After 2801 training steps, loss 1.5783, training accuracy 0.99
# After 2901 training steps, loss 1.56698, training accuracy 0.996667
# After 3001 training steps, loss 1.55917, training accuracy 0.993333
# After 3101 training steps, loss 1.55566, training accuracy 0.986667
# After 3201 training steps, loss 1.55038, training accuracy 0.99
# After 3301 training steps, loss 1.54554, training accuracy 0.993333
# After 3401 training steps, loss 1.54369, training accuracy 0.993333
# After 3501 training steps, loss 1.52945, training accuracy 0.993333
# After 3601 training steps, loss 1.52329, training accuracy 0.996667
# After 3701 training steps, loss 1.52524, training accuracy 0.993333
# After 3801 training steps, loss 1.51918, training accuracy 0.99
# After 3901 training steps, loss 1.517, training accuracy 0.996667
# After 4001 training steps, loss 1.51308, training accuracy 0.99
# After 4101 training steps, loss 1.50852, training accuracy 0.996667
# After 4201 training steps, loss 1.51742, training accuracy 0.99
# After 4301 training steps, loss 1.50124, training accuracy 1
# After 4401 training steps, loss 1.49999, training accuracy 0.996667
# After 4501 training steps, loss 1.50132, training accuracy 0.996667
# After 4601 training steps, loss 1.50567, training accuracy 0.983333
# After 4701 training steps, loss 1.49591, training accuracy 1
# After 4801 training steps, loss 1.49399, training accuracy 0.993333
# After 4901 training steps, loss 1.49332, training accuracy 0.993333
# After 5001 training steps, loss 1.49674, training accuracy 0.996667
# After 5101 training steps, loss 1.5038, training accuracy 0.99
# After 5201 training steps, loss 1.48796, training accuracy 0.996667
# After 5301 training steps, loss 1.48764, training accuracy 1
# After 5401 training steps, loss 1.48758, training accuracy 0.99
# After 5501 training steps, loss 1.49114, training accuracy 0.986667
# After 5601 training steps, loss 1.48241, training accuracy 1
# After 5701 training steps, loss 1.49237, training accuracy 0.993333
# After 5801 training steps, loss 1.48772, training accuracy 0.993333
# After 5901 training steps, loss 1.48896, training accuracy 0.996667
# After 6001 training steps, loss 1.48869, training accuracy 0.993333
# After 6101 training steps, loss 1.47925, training accuracy 1
# After 6201 training steps, loss 1.48168, training accuracy 0.996667
# After 6301 training steps, loss 1.48358, training accuracy 0.996667
# After 6401 training steps, loss 1.48069, training accuracy 0.996667
# After 6501 training steps, loss 1.47814, training accuracy 1
# After 6601 training steps, loss 1.47954, training accuracy 1
# After 6701 training steps, loss 1.47957, training accuracy 0.996667
# After 6801 training steps, loss 1.47983, training accuracy 0.993333
# After 6901 training steps, loss 1.47863, training accuracy 1
# After 7001 training steps, loss 1.47822, training accuracy 0.996667
# After 7101 training steps, loss 1.48232, training accuracy 0.993333
# After 7201 training steps, loss 1.48026, training accuracy 0.996667
# After 7301 training steps, loss 1.47721, training accuracy 1
# After 7401 training steps, loss 1.47636, training accuracy 0.996667
# After 7501 training steps, loss 1.47919, training accuracy 0.996667
# After 7601 training steps, loss 1.4775, training accuracy 0.996667
# After 7701 training steps, loss 1.4763, training accuracy 1
# After 7801 training steps, loss 1.47703, training accuracy 0.996667
# After 7901 training steps, loss 1.47317, training accuracy 1
# After 8001 training steps, loss 1.47587, training accuracy 1
# After 8101 training steps, loss 1.47625, training accuracy 0.996667
# After 8201 training steps, loss 1.47302, training accuracy 1
# After 8301 training steps, loss 1.47733, training accuracy 0.996667
# After 8401 training steps, loss 1.47897, training accuracy 1
# After 8501 training steps, loss 1.47979, training accuracy 0.99
# After 8601 training steps, loss 1.47419, training accuracy 0.996667
# After 8701 training steps, loss 1.47871, training accuracy 1
# After 8801 training steps, loss 1.47876, training accuracy 0.996667
# After 8901 training steps, loss 1.48149, training accuracy 0.993333
# After 9001 training steps, loss 1.47711, training accuracy 0.996667
# After 9101 training steps, loss 1.47686, training accuracy 1
# After 9201 training steps, loss 1.47579, training accuracy 1
# After 9301 training steps, loss 1.47332, training accuracy 1
# After 9401 training steps, loss 1.47118, training accuracy 1
# After 9501 training steps, loss 1.47195, training accuracy 0.996667
# After 9601 training steps, loss 1.4742, training accuracy 1
# After 9701 training steps, loss 1.47274, training accuracy 0.996667
# After 9801 training steps, loss 1.47385, training accuracy 0.996667
# After 9901 training steps, loss 1.47878, training accuracy 0.993333
# After 10001 training steps, loss 1.47127, training accuracy 1
# After 10101 training steps, loss 1.47397, training accuracy 0.996667
# After 10201 training steps, loss 1.47794, training accuracy 0.993333
# After 10301 training steps, loss 1.47101, training accuracy 1
# After 10401 training steps, loss 1.47061, training accuracy 0.996667
# After 10501 training steps, loss 1.47162, training accuracy 1
# After 10601 training steps, loss 1.47015, training accuracy 1
# After 10701 training steps, loss 1.47138, training accuracy 1
# After 10801 training steps, loss 1.47057, training accuracy 1
# After 10901 training steps, loss 1.47404, training accuracy 0.996667
# After 11001 training steps, loss 1.47752, training accuracy 0.996667
# After 11101 training steps, loss 1.47624, training accuracy 0.996667
# After 11201 training steps, loss 1.47017, training accuracy 1
# After 11301 training steps, loss 1.4736, training accuracy 1
# After 11401 training steps, loss 1.47667, training accuracy 0.996667
# After 11501 training steps, loss 1.47044, training accuracy 1
# After 11601 training steps, loss 1.47484, training accuracy 0.996667
# After 11701 training steps, loss 1.47805, training accuracy 1
# After 11801 training steps, loss 1.47689, training accuracy 0.993333
# After 11901 training steps, loss 1.47314, training accuracy 1
# After 12001 training steps, loss 1.4693, training accuracy 1
# After 12101 training steps, loss 1.47913, training accuracy 0.99
# After 12201 training steps, loss 1.47255, training accuracy 1
# After 12301 training steps, loss 1.47288, training accuracy 0.996667
# After 12401 training steps, loss 1.47214, training accuracy 1
# After 12501 training steps, loss 1.4768, training accuracy 0.993333
# After 12601 training steps, loss 1.47147, training accuracy 1
# After 12701 training steps, loss 1.47822, training accuracy 0.996667
# After 12801 training steps, loss 1.46921, training accuracy 1
# After 12901 training steps, loss 1.47005, training accuracy 1
# After 13001 training steps, loss 1.47453, training accuracy 1
# After 13101 training steps, loss 1.47965, training accuracy 0.99
# After 13201 training steps, loss 1.46925, training accuracy 1
# After 13301 training steps, loss 1.46997, training accuracy 1
# After 13401 training steps, loss 1.46909, training accuracy 1
# After 13501 training steps, loss 1.47594, training accuracy 1
# After 13601 training steps, loss 1.46933, training accuracy 1
# After 13701 training steps, loss 1.47097, training accuracy 1
# After 13801 training steps, loss 1.4697, training accuracy 1
# After 13901 training steps, loss 1.47753, training accuracy 0.996667
# After 14001 training steps, loss 1.4729, training accuracy 0.996667
# After 14101 training steps, loss 1.46865, training accuracy 1
# After 14201 training steps, loss 1.47352, training accuracy 0.996667
# After 14301 training steps, loss 1.47214, training accuracy 0.996667
# After 14401 training steps, loss 1.4772, training accuracy 0.993333
# After 14501 training steps, loss 1.46899, training accuracy 1
# After 14601 training steps, loss 1.47046, training accuracy 1
# After 14701 training steps, loss 1.47785, training accuracy 0.996667
# After 14801 training steps, loss 1.4684, training accuracy 1
# After 14901 training steps, loss 1.46841, training accuracy 1
# After 15001 training steps, loss 1.47173, training accuracy 0.996667
# After 15101 training steps, loss 1.46866, training accuracy 1
# After 15201 training steps, loss 1.47469, training accuracy 0.996667
# After 15301 training steps, loss 1.47029, training accuracy 1
# After 15401 training steps, loss 1.46979, training accuracy 1
# After 15501 training steps, loss 1.47218, training accuracy 0.996667
# After 15601 training steps, loss 1.46844, training accuracy 1
# After 15701 training steps, loss 1.46942, training accuracy 1
# After 15801 training steps, loss 1.47301, training accuracy 1
# After 15901 training steps, loss 1.46982, training accuracy 1
# After 16001 training steps, loss 1.47156, training accuracy 0.996667
# After 16101 training steps, loss 1.4693, training accuracy 1
# After 16201 training steps, loss 1.46941, training accuracy 1
# After 16301 training steps, loss 1.47526, training accuracy 0.99
# After 16401 training steps, loss 1.46783, training accuracy 1
# After 16501 training steps, loss 1.46886, training accuracy 1
# After 16601 training steps, loss 1.46782, training accuracy 1
# After 16701 training steps, loss 1.47182, training accuracy 0.996667
# After 16801 training steps, loss 1.47246, training accuracy 1
# After 16901 training steps, loss 1.4712, training accuracy 1
# After 17001 training steps, loss 1.4686, training accuracy 1
# After 17101 training steps, loss 1.47172, training accuracy 0.996667
# After 17201 training steps, loss 1.47092, training accuracy 0.996667
# After 17301 training steps, loss 1.46789, training accuracy 1
# After 17401 training steps, loss 1.46774, training accuracy 1
# After 17501 training steps, loss 1.47071, training accuracy 0.996667
# After 17601 training steps, loss 1.46818, training accuracy 1
# After 17701 training steps, loss 1.46817, training accuracy 0.996667
# After 17801 training steps, loss 1.47454, training accuracy 0.996667
# After 17901 training steps, loss 1.46885, training accuracy 1
# After 18001 training steps, loss 1.4709, training accuracy 0.996667
# After 18101 training steps, loss 1.4707, training accuracy 0.996667
# After 18201 training steps, loss 1.46827, training accuracy 1
# After 18301 training steps, loss 1.47082, training accuracy 1
# After 18401 training steps, loss 1.47072, training accuracy 0.996667
# After 18501 training steps, loss 1.46859, training accuracy 1
# After 18601 training steps, loss 1.47115, training accuracy 1
# After 18701 training steps, loss 1.46805, training accuracy 0.996667
# After 18801 training steps, loss 1.46991, training accuracy 1
# After 18901 training steps, loss 1.47429, training accuracy 0.993333
# After 19001 training steps, loss 1.47153, training accuracy 0.996667
# After 19101 training steps, loss 1.46775, training accuracy 1
# After 19201 training steps, loss 1.46759, training accuracy 1
# After 19301 training steps, loss 1.4703, training accuracy 0.996667
# After 19401 training steps, loss 1.47158, training accuracy 1
# After 19501 training steps, loss 1.47061, training accuracy 0.996667
# After 19601 training steps, loss 1.46723, training accuracy 1
# After 19701 training steps, loss 1.46782, training accuracy 1
# After 19801 training steps, loss 1.46768, training accuracy 1
# After 19901 training steps, loss 1.47274, training accuracy 0.996667
# After 20001 training steps, loss 1.46773, training accuracy 1
# After 20101 training steps, loss 1.47034, training accuracy 0.993333
# After 20201 training steps, loss 1.47301, training accuracy 0.996667
# After 20301 training steps, loss 1.46994, training accuracy 1
# After 20401 training steps, loss 1.4673, training accuracy 1
# After 20501 training steps, loss 1.46817, training accuracy 1
# After 20601 training steps, loss 1.47142, training accuracy 0.993333
# After 20701 training steps, loss 1.46833, training accuracy 1
# After 20801 training steps, loss 1.46718, training accuracy 1
# After 20901 training steps, loss 1.46892, training accuracy 0.996667
# After 21001 training steps, loss 1.46709, training accuracy 1
# After 21101 training steps, loss 1.46838, training accuracy 1
# After 21201 training steps, loss 1.46699, training accuracy 1
# After 21301 training steps, loss 1.467, training accuracy 0.996667
# After 21401 training steps, loss 1.46714, training accuracy 0.996667
# After 21501 training steps, loss 1.46824, training accuracy 0.996667
# After 21601 training steps, loss 1.47386, training accuracy 0.993333
# After 21701 training steps, loss 1.4681, training accuracy 1
# After 21801 training steps, loss 1.46922, training accuracy 1
# After 21901 training steps, loss 1.46678, training accuracy 1
# After 22001 training steps, loss 1.46935, training accuracy 1
# After 22101 training steps, loss 1.46892, training accuracy 0.996667
# After 22201 training steps, loss 1.47053, training accuracy 0.996667
# After 22301 training steps, loss 1.46792, training accuracy 1
# After 22401 training steps, loss 1.46712, training accuracy 1
# After 22501 training steps, loss 1.46973, training accuracy 1
# After 22601 training steps, loss 1.46699, training accuracy 0.996667
# After 22701 training steps, loss 1.46723, training accuracy 0.996667
# After 22801 training steps, loss 1.46674, training accuracy 1
# After 22901 training steps, loss 1.46743, training accuracy 0.993333
# After 23001 training steps, loss 1.46738, training accuracy 0.996667
# After 23101 training steps, loss 1.46697, training accuracy 1
# After 23201 training steps, loss 1.46726, training accuracy 0.996667
# After 23301 training steps, loss 1.46688, training accuracy 1
# After 23401 training steps, loss 1.4668, training accuracy 1
# After 23501 training steps, loss 1.46743, training accuracy 1
# After 23601 training steps, loss 1.4667, training accuracy 1
# After 23701 training steps, loss 1.467, training accuracy 1
# After 23801 training steps, loss 1.47095, training accuracy 0.996667
# After 23901 training steps, loss 1.46657, training accuracy 0.996667
# After 24001 training steps, loss 1.46709, training accuracy 1
# After 24101 training steps, loss 1.46818, training accuracy 1
# After 24201 training steps, loss 1.47116, training accuracy 0.996667
# After 24301 training steps, loss 1.47061, training accuracy 0.996667
# After 24401 training steps, loss 1.46689, training accuracy 1
# After 24501 training steps, loss 1.46832, training accuracy 1
# After 24601 training steps, loss 1.46856, training accuracy 1
# After 24701 training steps, loss 1.46775, training accuracy 1
# After 24801 training steps, loss 1.46832, training accuracy 1
# After 24901 training steps, loss 1.46823, training accuracy 1
# After 25001 training steps, loss 1.46635, training accuracy 1
# After 25101 training steps, loss 1.46642, training accuracy 1
# After 25201 training steps, loss 1.4678, training accuracy 1
# After 25301 training steps, loss 1.4665, training accuracy 1
# After 25401 training steps, loss 1.46659, training accuracy 1
# After 25501 training steps, loss 1.46864, training accuracy 1
# After 25601 training steps, loss 1.46969, training accuracy 0.993333
# After 25701 training steps, loss 1.46633, training accuracy 1
# After 25801 training steps, loss 1.46918, training accuracy 1
# After 25901 training steps, loss 1.47077, training accuracy 0.996667
# After 26001 training steps, loss 1.46681, training accuracy 1
# After 26101 training steps, loss 1.46675, training accuracy 0.996667
# After 26201 training steps, loss 1.46822, training accuracy 1
# After 26301 training steps, loss 1.46627, training accuracy 0.996667
# After 26401 training steps, loss 1.46668, training accuracy 1
# After 26501 training steps, loss 1.46672, training accuracy 1
# After 26601 training steps, loss 1.46645, training accuracy 1
# After 26701 training steps, loss 1.46915, training accuracy 1
# After 26801 training steps, loss 1.46729, training accuracy 1
# After 26901 training steps, loss 1.46658, training accuracy 1
# After 27001 training steps, loss 1.47034, training accuracy 1
# After 27101 training steps, loss 1.46651, training accuracy 1
# After 27201 training steps, loss 1.46609, training accuracy 1
# After 27301 training steps, loss 1.47013, training accuracy 1
# After 27401 training steps, loss 1.47034, training accuracy 0.996667
# After 27501 training steps, loss 1.4695, training accuracy 0.996667
# After 27601 training steps, loss 1.46698, training accuracy 1
# After 27701 training steps, loss 1.46961, training accuracy 0.996667
# After 27801 training steps, loss 1.46688, training accuracy 0.996667
# After 27901 training steps, loss 1.46639, training accuracy 1
# After 28001 training steps, loss 1.469, training accuracy 1
# After 28101 training steps, loss 1.46584, training accuracy 1
# After 28201 training steps, loss 1.46585, training accuracy 1
# After 28301 training steps, loss 1.46597, training accuracy 1
# After 28401 training steps, loss 1.46635, training accuracy 0.996667
# After 28501 training steps, loss 1.46634, training accuracy 1
# After 28601 training steps, loss 1.46855, training accuracy 1
# After 28701 training steps, loss 1.46588, training accuracy 1
# After 28801 training steps, loss 1.46617, training accuracy 1
# After 28901 training steps, loss 1.46621, training accuracy 1
# After 29001 training steps, loss 1.46582, training accuracy 1
# After 29101 training steps, loss 1.46588, training accuracy 1
# After 29201 training steps, loss 1.46632, training accuracy 1
# After 29301 training steps, loss 1.46576, training accuracy 1
# After 29401 training steps, loss 1.4666, training accuracy 1
# After 29501 training steps, loss 1.46616, training accuracy 1
# After 29601 training steps, loss 1.46736, training accuracy 1
# After 29701 training steps, loss 1.46678, training accuracy 1
# After 29801 training steps, loss 1.46817, training accuracy 1
# After 29901 training steps, loss 1.46609, training accuracy 1
