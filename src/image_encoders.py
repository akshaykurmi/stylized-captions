import tensorflow as tf


class InceptionResNetEncoder(tf.keras.Model):
    IMAGE_FEATURE_SHAPE = (9, 9, 1536)

    def __init__(self):
        super().__init__()
        self.resnet = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights="imagenet")
        for layer in self.resnet.layers:
            layer.trainable = False

    def call(self, inp):
        x = tf.keras.applications.inception_resnet_v2.preprocess_input(inp)
        x = self.resnet(x)
        return x

    @property
    def output_shape(self):
        return self.resnet.layers[-1].output_shape
