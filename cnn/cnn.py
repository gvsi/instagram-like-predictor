# Convolutional Neural Network implementation for Instagram image analysis
import tensorflow as tf
import numpy as np

class CNN:
    # runs feed foward network to predict number of likes
    def predict(self, X, Xa):
        return self.s.run(self.Y, feed_dict={self.X_: X, self.Xa_: Xa, self.pkeep_: 1})

    # evaluates the mean loss of the model
    def evaluate(self, X, Xa, Y):
        return self.s.run(self._get_loss(Y), feed_dict={self.X_: X, self.Xa_: Xa, self.pkeep_: 1})
    
    # trains then model
    def train(self, X_b, Xa_b, Y_b, iterations=1, pkeep=0.9): 
        for b in range(X_b.shape[0]):
            for i in range(iterations):
                self.s.run(self.train_step, 
                feed_dict={self.X_: X_b[b], self.Xa_: Xa_b[b], self.Y_: Y_b[b], self.pkeep_: pkeep})
    
    # save model parameters for restore
    def save(self, path):
        tf.train.Saver().save(self.s, path)

    def get_last_layer(self, X, Xa):
        return self.s.run(self.last_layer, feed_dict={self.X_: X, self.Xa_: Xa, self.pkeep_: 1})
        

    def __init__(self, imSize, \
        C=[[3, 5, 2], [6, 4, 2], [20, 4, 2], [40, 3, 2]], \
        F=[500, 200, 40, 8], target=0, path=None):
        target_gpu = '/gpu:' + str(target)
        with tf.device(target_gpu):
            self.X_, self.Xa_, self.Y_, self.pkeep_, self.Y, self.last_layer, self.train_step, self.s = \
            self._make_model(imSize, C, F)
        if path != None:
            tf.train.Saver().restore(self.s, path)

    def _make_model(self, imSize, C, F):
        # Model hyper parameters:
        C_IN = 3 # 3 image input channels
        F_OUT = 1 # single predictor, number of likes
        # Convolutional Layers [# out channels, filter size, stride]
        C.insert(0, [C_IN, -1, -1])
        # Fully connected layer sizes
        F.append(F_OUT)

        # Reserve data placeholders 
        X_ = tf.placeholder(tf.float32, [None, imSize, imSize, C_IN]) # image data
        Xa_ = tf.placeholder(tf.float32, [None, 3]) # auxiliary features 
        Y_ = tf.placeholder(tf.float32, [None, F_OUT]) # number of likes
        pkeep_ = tf.placeholder(tf.float32) # dropout percentage

        # Map feed foward network
        Y, last_layer = self._map_output(X_, Xa_, pkeep_, C, F)

        # Define training method
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        train_step = optimizer.minimize(tf.norm(Y - Y_))

        # Initialize runtime enviornment
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        s.run(tf.global_variables_initializer())

        return X_, Xa_, Y_, pkeep_, Y, last_layer, train_step, s

    def _map_output(self, X_, Xa_, pkeep_, C, F):
        Y = X_
        # Construct convolutional layers 
        for i in range(1, len(C)):
            Y = self._conv_layer(Y, C[i - 1][0], C[i][0], C[i][1], C[i][2])
        # Flatten layers to 1-D
        Y, F1 = self._flatten(Y)
        # Add number of followers control parameter here, once convolution is complete
        
        # Uncomment the following two lines to add in the user metadata features
        #Y = tf.concat([Y, Xa_], 1)
        #F.insert(0, F1 + 3)
        F.insert(0, F1)
        # Construct fully connected layers 
        last_layer = 0
        for i in range(1, len(F)):
            Y = self._full_layer(Y, F[i - 1], F[i], pkeep_)
            if i == (len(F) - 2):
                last_layer = Y
        return Y, last_layer
    
    def _conv_layer(self, input, channels_in, channels_out, filter_size, stride):
        W = tf.Variable(tf.truncated_normal([filter_size, filter_size, channels_in, channels_out], stddev=0.01))
        B = tf.Variable(tf.ones([channels_out]))
        out = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME') + B)
        return out

    def _flatten(self, input):
        input_shape = input.get_shape().as_list()
        n = 1 
        for i in range(1, len(input_shape)):
            n *= input_shape[i]
        out = tf.reshape(input, shape=[-1, n])
        return out, n

    def _full_layer(self, input, channels_in, channels_out, pkeep_):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.01))
        B = tf.Variable(tf.ones([channels_out]))
        out = tf.nn.relu(tf.matmul(input, W) + B)
        out = tf.nn.dropout(out, pkeep_)
        return out
        
    def _get_loss(self, Y):
        return tf.norm(Y - self.Y) / np.sqrt(Y.shape[0])
