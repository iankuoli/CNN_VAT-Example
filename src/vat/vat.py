import tensorflow as tf
import numpy
import sys, os


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('epsilon', 8.0, "norm length for (virtual) adversarial training ")
tf.app.flags.DEFINE_integer('num_power_iterations', 1, "the number of power iterations")
tf.app.flags.DEFINE_float('xi', 1e-6, "small constant for finite difference")


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * tf.nn.log_softmax(p_logit), 1))
    return qlogq - qlogp


def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d


def generate_virtual_adversarial_perturbation(x, logit, forward, is_training=True):
    d = tf.random_normal(shape=tf.shape(x))

    for _ in range(FLAGS.num_power_iterations):
        d = FLAGS.xi * get_normalized_vector(d)
        logit_p = logit
        logit_m = forward(x + d, update_batch_stats=False, is_training=is_training)
        dist = kl_divergence_with_logit(logit_p, logit_m)
        grad = tf.gradients(dist, [d], aggregation_method=2)[0]
        d = tf.stop_gradient(grad)

    return FLAGS.epsilon * get_normalized_vector(d)


def virtual_adversarial_loss(x, logit, forward, is_training=True, name="vat_loss"):
    r_vadv = generate_virtual_adversarial_perturbation(x, logit, forward, is_training=is_training)
    logit = tf.stop_gradient(logit)
    logit_p = logit
    logit_m = forward(x + r_vadv, update_batch_stats=False, is_training=is_training)
    loss = kl_divergence_with_logit(logit_p, logit_m)
    return tf.identity(loss, name=name)


def generate_adversarial_perturbation(x, loss):
    grad = tf.gradients(loss, [x], aggregation_method=2)[0]
    grad = tf.stop_gradient(grad)
    return FLAGS.epsilon * get_normalized_vector(grad)


def adversarial_loss(x, y, loss, forward, is_training=True, name="at_loss"):
    r_adv = generate_adversarial_perturbation(x, loss)
    logit = forward(x + r_adv, is_training=is_training, update_batch_stats=False)
    loss = L.ce_loss(logit, y)
    return loss
