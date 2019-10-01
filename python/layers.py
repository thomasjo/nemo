from tensorflow import keras


class Dropout(keras.layers.Dropout):
    def __init__(self, rate, force=False, **kwargs):
        super().__init__(rate, **kwargs)
        self.force = force


    def call(self, inputs, training=None):
        if self.force:
            training = True
        return super().call(inputs, training)


    def get_config(self):
        config = { "force": self.force }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
