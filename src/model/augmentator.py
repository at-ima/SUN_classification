import tensorflow as tf
from tensorflow.keras import layers


class Augmentator(tf.keras.Model):
    def __init__(self, model, aug_model):
        super(Augmentator, self).__init__()
        self.aug_model = aug_model
        self.model = model
    def call(self, x, training=False):
        if training:
            x = self.aug_model(x)
        x = self.model(x, training=training)
        return x