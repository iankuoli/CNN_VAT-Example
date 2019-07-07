from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
import numpy as np

import src.prob_cnn_model as cnn
import src.utils as utils
import src.vat.vat2 as vat
from src.losses.face_losses2 import arcface_loss, focal_loss_with_softmax

# MNIST dataset parameters. Total classes (0-9 digits).
num_classes = 10

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 256
display_step = 10

# Hyper-parameter for loss
use_loss = 'cat'
use_vat = False
xi_vat = 1e-4
m_arcface = 0.5

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert x to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Convert y to int32.
y_train, y_test = np.array(y_train, np.int32), np.array(y_test, np.int32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Build neural network model.
conv_net = cnn.ConvNet(num_classes, use_loss=use_loss)

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.Adam(learning_rate)


# Optimization process.
def run_optimization(x, y, step, loss_type='cat', use_vat=False):

    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:

        # Forward pass.
        embeds, preds = conv_net(x, is_training=True)

        # Compute inference loss.
        if loss_type == 'arcface':
            arcface_logits = arcface_loss(embedding=embeds, labels=y, out_num=num_classes,
                                          weights=conv_net.out.weights[0], m=m_arcface)
            embeds_loss = tf.reduce_mean(focal_loss_with_softmax(logits=arcface_logits, labels=y))
            inference_loss = utils.cross_entropy_loss(preds, y)
            loss = embeds_loss + inference_loss
        else:
            # Compute the -ELBO as the loss, averaged over the batch size.
            tmp = tf.one_hot(y, num_classes)
            tmp2 = preds.log_prob(y)
            neg_log_likelihood = -tf.reduce_mean(input_tensor=tmp2)
            kl = sum(conv_net.losses) / batch_size
            elbo_loss = neg_log_likelihood + kl
            loss = elbo_loss

        if use_vat:
            vat_loss = vat.virtual_adversarial_loss(x, embeds, conv_net,
                                                    xi=xi_vat*vat.vat_decay(step), is_training=False)
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

        run_optimization(batch_x, batch_y, step, loss_type=use_loss, use_vat=use_vat)

        if step % display_step == 0:

            embed, pred = conv_net(x_test)

            predictions = tf.argmax(input=pred, axis=1)
            acc = utils.accuracy(pred, y_test)

            if use_loss == 'arcface':
                arcface_logit = arcface_loss(embedding=embed, labels=y_test, out_num=num_classes,
                                             weights=conv_net.out.weights[0], m=m_arcface)
                embed_loss = tf.reduce_mean(focal_loss_with_softmax(logits=arcface_logit, labels=y_test))
                infer_loss = utils.cross_entropy_loss(pred, y_test)
                print("step: %i, embed_loss: %f, infer_loss: %f, accuracy: %f" % (step, embed_loss, infer_loss, acc))
            else:
                # Compute the -ELBO as the loss, averaged over the batch size.
                loss = sum(conv_net.losses)
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
