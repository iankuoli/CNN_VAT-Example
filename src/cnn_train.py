from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

import src.cnn_model as cnn
import src.utils as utils
import src.vat.vat as vat
from src.losses.face_losses2 import arcface_loss, focal_loss_with_softmax

# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 1024 # number of neurons for 1st fully-connected layer.

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Build neural network model.
conv_net = cnn.ConvNet(num_classes)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)


# Optimization process.
def run_optimization(x, y, loss_type='cat', use_vat=False):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:

        # Forward pass.
        embed, pred = conv_net(x, is_training=True)

        # Compute inference loss.
        if loss_type == 'arcface':
            w_init_method = tf.initializers.GlorotNormal()
            arcface_logit = arcface_loss(embedding=embed, labels=y, w_init=w_init_method, out_num=num_classes)
            inference_loss = tf.reduce_mean(focal_loss_with_softmax(logits=arcface_logit, labels=y))
        else:
            inference_loss = utils.cross_entropy_loss(pred, y)

        loss = inference_loss

        if use_vat:
            l_logit = conv_net(x, is_training=True)
            vat_loss = vat.virtual_adversarial_loss(x, l_logit, conv_net, is_training=True)
            loss += vat_loss

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    #run_optimization(batch_x, batch_y, loss_type='arcface', use_vat=True)
    run_optimization(batch_x, batch_y, use_vat=True)

    if step % display_step == 0:
        embed, pred = conv_net(batch_x)
        loss = utils.cross_entropy_loss(pred, batch_y)
        acc = utils.accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
