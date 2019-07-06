import tensorflow as tf
import math


def arcface_loss(embedding, labels, out_num, m=0.5, s=64., w_init=None):
    '''
    :param embedding: the input embedding vectors
    :param labels: the input labels, the shape should be e.g., (batch_size, 1)
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :param s: the scalar value, default is 64
    :param w_init: the method for weight initialization
    :return: the final calculated output, this output is send into the tf.nn.softmax directly
    '''

    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)

    # Normalization of inputs and weights
    weights = tf.Variable(w_init(shape=(embedding.get_shape().as_list()[-1], out_num)), name='embedding_weights',
                          dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)

    # cos(theta+m)
    cos_t = tf.matmul(embedding_unit, weights_unit, name='cos_t')
    cos_t2 = tf.square(cos_t, name='cos_2')
    sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
    sin_t = tf.sqrt(sin_t2, name='sin_t')
    cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

    # this condition controls the theta+m should in range [0, pi]
    #      0 <= theta+m <= pi
    #     -m <= theta <= pi-m
    cond_v = cos_t - threshold
    cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

    keep_val = s*(cos_t - mm)
    cos_mt_temp = tf.where(cond, cos_mt, keep_val)

    mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')

    s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

    output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

    return output


def cosineface_losses(embedding, labels, out_num, m=0.4, s=30., w_init=None):
    '''
    :param embedding: the input embedding vectors
    :param labels: the input labels, the shape should be e.g., (batch_size, 1)
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :param s: the scalar value, default is 64
    :param w_init: the method for weight initialization
    :return: the final calculated output, this output is send into the tf.nn.softmax directly
    '''

    # Normalization of inputs and weights
    weights = tf.Variable(w_init(shape=(embedding.get_shape().as_list()[-1], out_num)), name='embedding_weights',
                          dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)

    # cos_theta - m
    cos_t = tf.matmul(embedding_unit, weights_unit, name='cos_t')
    cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

    mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
    inv_mask = tf.subtract(1., mask, name='inverse_mask')

    output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')

    cosineface_loss = tf.keras.Model(inputs=embedding, outputs=output)

    return cosineface_loss


def combine_loss_val(embedding, labels, out_num, margin_a, margin_m, margin_b, s, w_init=None):
    '''
    :param embedding: the input embedding vectors
    :param labels: the input labels, the shape should be e.g., (batch_size, 1)
    :param out_num: output class num
    :param margin_a: the margin value w.r.t. a
    :param margin_m: the margin value w.r.t. m
    :param margin_b: the margin value w.r.t. b
    :param s: the scalar value
    :param w_init: the method for weight initialization
    :return: the final calculated output, this output is send into the tf.nn.softmax directly
    '''

    weights = tf.Variable(w_init(shape=(embedding.get_shape().as_list()[-1], out_num)), name='embedding_weights',
                          dtype=tf.float32)
    weights_unit = tf.nn.l2_normalize(weights, axis=0)
    embedding_unit = tf.nn.l2_normalize(embedding, axis=1)
    cos_t = tf.matmul(embedding_unit, weights_unit)
    ordinal = tf.constant(list(range(0, embedding.get_shape().as_list()[0])), tf.int64)
    ordinal_y = tf.stack([ordinal, labels], axis=1)
    zy = cos_t * s
    sel_cos_t = tf.gather_nd(zy, ordinal_y)
    if margin_a != 1.0 or margin_m != 0.0 or margin_b != 0.0:
        if margin_a == 1.0 and margin_m == 0.0:
            s_m = s * margin_b
            new_zy = sel_cos_t - s_m
        else:
            cos_value = sel_cos_t / s
            t = tf.acos(cos_value)
            if margin_a != 1.0:
                t *= margin_a
            if margin_m > 0.0:
                t += margin_m
            body = tf.cos(t)
            if margin_b > 0.0:
                body -= margin_b
            new_zy = body * s
    updated_logits = tf.add(zy, tf.scatter_nd(ordinal_y, tf.subtract(new_zy, sel_cos_t), zy.get_shape()))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=updated_logits))
    predict_cls = tf.argmax(updated_logits, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    predict_cls_s = tf.argmax(zy, 1)
    accuracy_s = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predict_cls_s, tf.int64), tf.cast(labels, tf.int64)), 'float'))
    return zy, loss, accuracy, accuracy_s, predict_cls_s


def focal_loss_with_softmax(labels, logits, gamma=2):
    """
    labels: shape([batch_size], type = int32)
    logits: shape([batch_size,num_classes], type = float32)
    gamma: hyper-parameter
    return L: mean loss
    """
    y_pred = tf.nn.softmax(logits, axis=-1)
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    loss = -labels * ((1 - y_pred) ** gamma) * tf.math.log(y_pred)
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    return loss
