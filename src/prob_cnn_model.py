import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
import tensorflow_probability as tfp


# Create TF Model.
class ConvNet(Model):

    # Set layers.
    def __init__(self, num_classes, use_loss='cat', s=64):
        super(ConvNet, self).__init__()

        self.use_loss = use_loss
        self.cos_scale = s

        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = tfp.layers.Convolution2DFlipout(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = tfp.layers.Convolution2DFlipout(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = tfp.layers.DenseFlipout(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout1 = layers.Dropout(rate=0.5)

        # Fully connected layer.
        self.fc2 = tfp.layers.DenseFlipout(64, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout2 = layers.Dropout(rate=0.5)

        # Output layer, class prediction.
        self.out = tfp.layers.DenseFlipout(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):

        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=is_training)
        x = self.fc2(x)
        embed = self.dropout2(x, training=is_training)
        out = self.out(embed)

        if self.use_loss == 'arcface':
            embed_unit = tf.nn.l2_normalize(embed, axis=1)
            weights_unit = tf.nn.l2_normalize(self.out.weights[0], axis=1)
            cos_t = tf.matmul(embed_unit, weights_unit, name='cos_t')
            out = cos_t * self.cos_scale

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            out = tf.nn.softmax(out)
        else:
            out = tfp.distributions.Categorical(logits=out)

        return embed, out
