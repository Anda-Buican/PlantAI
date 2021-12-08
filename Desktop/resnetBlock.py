import tensorflow.keras as keras


class ResnBlock(keras.layers.Layer):
    def __init__(self, filters,strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers=[
            keras.layers.Conv2D(filters, 3, strides=strides, padding="same"),
            keras.layers.BatchNormalization(scale=False, trainable = False),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding="same"),
            keras.layers.BatchNormalization(scale=False, trainable = False)]
        self.skip_layers=[]
        if strides>1:
            self. skip_layers=[keras.layers.Conv2D(filters, 1, strides=strides, padding="same"),
            keras.layers.BatchNormalization(scale=False, trainable = False)]

    def call(self, inputs):
        z = inputs
        for layer in self.main_layers:
            z= layer(z)
        skip_z=inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)
        return self.activation( z + skip_z)




