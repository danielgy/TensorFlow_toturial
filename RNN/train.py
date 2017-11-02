# hyperparameters
batch_size = 128
n_inputs = 28 # 28 cl
n_steps = 28 # 28 rows -> time stamps
n_hidden_unins = 128 # hidden units
n_classes = 10
epochs = 100

if __name__ == '__main__':
    st = time.time()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    xs = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="inputs")
    ys = tf.placeholder(tf.float32, [None, n_classes], name="outputs")
    model = model.RNN_Model(xs, ys)
    # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="predict/inlayer|predict/rnn")
    # reuse_vars_dict = dict([(var.name, var.name) for var in train_vars])
    # print ("train vars:",reuse_vars_dict)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        batch = mnist.train.num_examples / batch_size
        for epoch in range(epochs):
            for i in range(int(batch)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x = batch_x.reshape([batch_size, n_inputs, n_steps])
                sess.run(model.optimizer, feed_dict={xs: batch_x, ys: batch_y})
            accu, loss = sess.run(model.accuracy, feed_dict={xs: batch_x, ys: batch_y})
            print('epoch:', epoch + 1, 'loss:', loss, 'train accuracy:', accu)
        end = time.time()
        print('*' * 30)
        print('training finish.\ncost time:', int(end - st), 'seconds\ntest accuracy:',
              sess.run(model.accuracy[0], feed_dict=
              {xs: mnist.test.images.reshape([-1, n_steps, n_inputs]), ys: mnist.test.labels}))



##TODO: train process

# epoch: 1 loss: 0.426177 train accuracy: 0.875
# epoch: 2 loss: 0.190436 train accuracy: 0.945312
# epoch: 3 loss: 0.103907 train accuracy: 0.96875
# epoch: 4 loss: 0.115983 train accuracy: 0.960938
# epoch: 5 loss: 0.136543 train accuracy: 0.945312
# epoch: 6 loss: 0.0696603 train accuracy: 0.976562
# epoch: 7 loss: 0.0654153 train accuracy: 0.976562
# epoch: 8 loss: 0.0908167 train accuracy: 0.984375
# epoch: 9 loss: 0.0344795 train accuracy: 0.984375
# epoch: 10 loss: 0.0360453 train accuracy: 0.992188
# epoch: 11 loss: 0.0593322 train accuracy: 0.984375
# epoch: 12 loss: 0.0089405 train accuracy: 1.0
# epoch: 13 loss: 0.0198046 train accuracy: 0.992188
# epoch: 14 loss: 0.00538229 train accuracy: 1.0
# epoch: 15 loss: 0.00554219 train accuracy: 1.0
# epoch: 16 loss: 0.0351925 train accuracy: 0.976562
# epoch: 17 loss: 0.00768662 train accuracy: 1.0
# epoch: 18 loss: 0.00207847 train accuracy: 1.0
# epoch: 19 loss: 0.0159064 train accuracy: 0.992188
# epoch: 20 loss: 0.0294105 train accuracy: 0.992188
# epoch: 21 loss: 0.0149525 train accuracy: 0.992188
# epoch: 22 loss: 0.0120278 train accuracy: 1.0
# epoch: 23 loss: 0.011114 train accuracy: 1.0
# epoch: 24 loss: 0.0149713 train accuracy: 0.992188
# epoch: 25 loss: 0.00760254 train accuracy: 1.0
# epoch: 26 loss: 0.0167206 train accuracy: 0.984375
# epoch: 27 loss: 0.0050269 train accuracy: 1.0
# epoch: 28 loss: 0.00504275 train accuracy: 1.0
# epoch: 29 loss: 0.00231944 train accuracy: 1.0
# epoch: 30 loss: 0.00395261 train accuracy: 1.0
# epoch: 31 loss: 0.00339805 train accuracy: 1.0
# epoch: 32 loss: 0.00444342 train accuracy: 1.0
# epoch: 33 loss: 0.000920703 train accuracy: 1.0
# epoch: 34 loss: 0.00128484 train accuracy: 1.0
# epoch: 35 loss: 0.00669292 train accuracy: 0.992188
# epoch: 36 loss: 0.00163957 train accuracy: 1.0
# epoch: 37 loss: 0.00221118 train accuracy: 1.0
# epoch: 38 loss: 0.00446441 train accuracy: 1.0
# epoch: 39 loss: 0.00168331 train accuracy: 1.0
# epoch: 40 loss: 0.0017838 train accuracy: 1.0
# epoch: 41 loss: 0.00197805 train accuracy: 1.0
# epoch: 42 loss: 0.00218299 train accuracy: 1.0
# epoch: 43 loss: 0.002345 train accuracy: 1.0
# epoch: 44 loss: 0.00164111 train accuracy: 1.0
# epoch: 45 loss: 0.0014332 train accuracy: 1.0
# epoch: 46 loss: 0.000559565 train accuracy: 1.0
# epoch: 47 loss: 0.00138017 train accuracy: 1.0
# epoch: 48 loss: 0.000359039 train accuracy: 1.0
# epoch: 49 loss: 0.000993544 train accuracy: 1.0
# epoch: 50 loss: 0.00826959 train accuracy: 1.0
# epoch: 51 loss: 0.00165513 train accuracy: 1.0
# epoch: 52 loss: 0.00383531 train accuracy: 1.0
# epoch: 53 loss: 0.00345915 train accuracy: 1.0
# epoch: 54 loss: 0.000583641 train accuracy: 1.0
# epoch: 55 loss: 0.000409852 train accuracy: 1.0
# epoch: 56 loss: 0.00016325 train accuracy: 1.0
# epoch: 57 loss: 0.000360035 train accuracy: 1.0
# epoch: 58 loss: 0.000293305 train accuracy: 1.0
# epoch: 59 loss: 0.000234783 train accuracy: 1.0
# epoch: 60 loss: 0.00016899 train accuracy: 1.0
# epoch: 61 loss: 0.00256475 train accuracy: 1.0
# epoch: 62 loss: 0.00152844 train accuracy: 1.0
# epoch: 63 loss: 0.00138145 train accuracy: 1.0
# epoch: 64 loss: 6.75324e-05 train accuracy: 1.0
# epoch: 65 loss: 8.42986e-05 train accuracy: 1.0
# epoch: 66 loss: 0.000546167 train accuracy: 1.0
# epoch: 67 loss: 0.00755186 train accuracy: 0.992188
# epoch: 68 loss: 0.00231283 train accuracy: 1.0
# epoch: 69 loss: 7.91846e-05 train accuracy: 1.0
# epoch: 70 loss: 0.000340066 train accuracy: 1.0
# epoch: 71 loss: 0.000262903 train accuracy: 1.0
# epoch: 72 loss: 0.000127092 train accuracy: 1.0
# epoch: 73 loss: 0.000117066 train accuracy: 1.0
# epoch: 74 loss: 9.88594e-05 train accuracy: 1.0
# epoch: 75 loss: 0.000213902 train accuracy: 1.0
# epoch: 76 loss: 9.63204e-05 train accuracy: 1.0
# epoch: 77 loss: 0.000164345 train accuracy: 1.0
# epoch: 78 loss: 2.75961e-05 train accuracy: 1.0
# epoch: 79 loss: 0.000897106 train accuracy: 1.0
# epoch: 80 loss: 0.000381499 train accuracy: 1.0
# epoch: 81 loss: 0.000229418 train accuracy: 1.0
# epoch: 82 loss: 0.000145419 train accuracy: 1.0
# epoch: 83 loss: 0.000103624 train accuracy: 1.0
# epoch: 84 loss: 0.00012806 train accuracy: 1.0
# epoch: 85 loss: 0.00033566 train accuracy: 1.0
# epoch: 86 loss: 4.98282e-05 train accuracy: 1.0
# epoch: 87 loss: 2.14409e-05 train accuracy: 1.0
# epoch: 88 loss: 0.000163397 train accuracy: 1.0
# epoch: 89 loss: 7.77569e-05 train accuracy: 1.0
# epoch: 90 loss: 7.44149e-05 train accuracy: 1.0
# epoch: 91 loss: 4.0345e-05 train accuracy: 1.0
# epoch: 92 loss: 0.00138165 train accuracy: 1.0
# epoch: 93 loss: 6.04947e-05 train accuracy: 1.0
# epoch: 94 loss: 0.000188576 train accuracy: 1.0
# epoch: 95 loss: 0.000309948 train accuracy: 1.0
# epoch: 96 loss: 0.000148765 train accuracy: 1.0
# epoch: 97 loss: 0.000934982 train accuracy: 1.0
# epoch: 98 loss: 0.000359906 train accuracy: 1.0
# epoch: 99 loss: 0.00132019 train accuracy: 1.0
# epoch: 100 loss: 0.000352469 train accuracy: 1.0
# ******************************
# training finish.
# cost time: 643 seconds
# test accuracy: 0.9869
