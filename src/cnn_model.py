import tensorflow as tf
from tensorflow.keras import Model, layers


# Create TF Model.
class ConvNet(Model):

    # Set layers.
    def __init__(self, num_classes, s=64):
        super(ConvNet, self).__init__()

        # Convolution Layer with 32 filters and a kernel size of 5.
        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool1 = layers.MaxPool2D(2, strides=2)

        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        self.maxpool2 = layers.MaxPool2D(2, strides=2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout1 = layers.Dropout(rate=0.5)

        # Fully connected layer.
        self.fc2 = layers.Dense(32)
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout2 = layers.Dropout(rate=0.5)

        self.cos_weight = tf.nn.l2_normalize(tf.Variable(tf.initializers.GlorotNormal()(shape=(32, num_classes)),
                                                         name='embedding_weights', dtype=tf.float32),
                                             axis=0)
        self.cos_scale = s

        # Output layer, class prediction.
        self.out = layers.Dense(num_classes)

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

        embed_unit = tf.nn.l2_normalize(embed, axis=1)
        cos_t = tf.matmul(embed_unit, self.cos_weight, name='cos_t')

        prediction = self.out(cos_t * self.cos_scale)

        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            prediction = tf.nn.softmax(prediction)

        return embed, prediction
