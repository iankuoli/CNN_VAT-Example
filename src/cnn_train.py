from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

import src.cnn_model as cnn
import src.utils as utils
import src.vat.vat2 as vat
from src.losses.face_losses2 import arcface_loss, focal_loss_with_softmax

# MNIST dataset parameters. Total classes (0-9 digits).
num_classes = 10

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

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
        embeds, preds = conv_net(x, is_training=True)

        # Compute inference loss.
        if loss_type == 'arcface':
            arcface_logits = arcface_loss(embedding=embeds, labels=y, out_num=num_classes,
                                          w_init=tf.initializers.GlorotNormal())
            embeds_loss = tf.reduce_mean(focal_loss_with_softmax(logits=arcface_logits, labels=y))
            inference_loss = utils.cross_entropy_loss(preds, y)
            loss = embeds_loss + inference_loss
        else:
            loss = utils.cross_entropy_loss(preds, y)

        if use_vat:
            vat_loss = vat.virtual_adversarial_loss(x, embeds, conv_net, is_training=True)
            loss += vat_loss

    # Variables to update, i.e. trainable variables.
    trainable_variables = conv_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# Run training for the given number of steps.
for epoch in range(3):
    print('Start of epoch %d' % (epoch,))

    for step, (batch_x, batch_y) in enumerate(train_data):
        use_loss = 'arcface'
        use_vat = False

        run_optimization(batch_x, batch_y, loss_type=use_loss, use_vat=use_vat)

        if step % display_step == 0:

            embed, pred = conv_net(batch_x)

            acc = utils.accuracy(pred, batch_y)

            if use_loss == 'arcface':
                arcface_logit = arcface_loss(embedding=embed, labels=batch_y, out_num=num_classes,
                                             w_init=tf.initializers.GlorotNormal())
                embed_loss = tf.reduce_mean(focal_loss_with_softmax(logits=arcface_logit, labels=batch_y))
                infer_loss = utils.cross_entropy_loss(pred, batch_y)
                print("step: %i, embed_loss: %f, infer_loss: %f, accuracy: %f" % (step, embed_loss, infer_loss, acc))
            else:
                loss = utils.cross_entropy_loss(pred, batch_y)
                print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

'''
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y, use_vat=True)c

    if step % display_step == 0:
        embed = conv_net(batch_x)
        loss = utils.cross_entropy_loss(embed, batch_y)
        acc = utils.accuracy(embed, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
'''
