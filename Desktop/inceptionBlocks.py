import tensorflow.keras as keras
import tensorflow.keras.layers
import tensorflow.keras.activations

class InceptionReduction(keras.layers.Layer):
    def __init__(self, filters, canaleAdd, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get("relu")
        self.path1 = [
            keras.layers.Conv2D(filters, 1, strides=1),
            self.activation,
            keras.layers.Conv2D(178+canaleAdd, 3, strides=1,padding="same"),
            self.activation,
            keras.layers.Conv2D(178+canaleAdd, 3, strides=2),
            self.activation
        ]
        self.path2 = [
            keras.layers.Conv2D(filters, 1, strides=1),
            self.activation,
            keras.layers.Conv2D(302+canaleAdd, 3, strides=2),
            self.activation,
        ]
        self.path3 = [
            keras.layers.MaxPooling2D(pool_size=(3, 3),strides=2),
        ]

    def call(self, inputs):
        p1 = inputs
        for layer in self.path1:
            p1= layer(p1)
        p2 = inputs
        for layer in self.path2:
            p2 = layer(p2)
        p3 = inputs
        for layer in self.path3:
            p3 = layer(p3)
        return keras.layers.Concatenate(axis=-1)([p1,p2,p3])

class InceptionBlock1(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get("relu")
        self.path1 = [
            keras.layers.Conv2D(64, 1, strides=1),
            self.activation,
            keras.layers.Conv2D(96, 3, strides=1,padding="same"),
            self.activation,
            keras.layers.Conv2D(96, 3, strides=1, padding="same"),
            self.activation
        ]
        self.path2 = [
            keras.layers.Conv2D(48, 1, strides=1),
            self.activation,
            keras.layers.Conv2D(64, 3, strides=1, padding="same"),
            self.activation,
        ]
        self.path3 = [
            keras.layers.MaxPooling2D(pool_size=(3, 3),strides=1,padding="same"),
            keras.layers.Conv2D(64, 1, strides=1),
            self.activation,
        ]
        self.path4 = [
            keras.layers.Conv2D(64, 1, strides=1),
            self.activation,
        ]

    def call(self, inputs):
        p1 = inputs
        for layer in self.path1:
            p1= layer(p1)
        p2 = inputs
        for layer in self.path2:
            p2 = layer(p2)
        p3 = inputs
        for layer in self.path3:
            p3 = layer(p3)
        p4 = inputs
        for layer in self.path4:
            p4 = layer(p4)
        return keras.layers.Concatenate(axis=-1)([p1,p2,p3,p4])

class InceptionBlock2(keras.layers.Layer):
    def __init__(self, filtre, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get("relu")
        self.path1 = [
            keras.layers.Conv2D(filtre, 1, strides=1),
            self.activation,
            keras.layers.ZeroPadding2D(padding=(0, 3)),
            keras.layers.Conv2D(filtre, (1,7), strides=1),
            self.activation,
            keras.layers.ZeroPadding2D(padding=(3, 0)),
            keras.layers.Conv2D(filtre, (7,1), strides=1),
            self.activation,
            keras.layers.ZeroPadding2D(padding=(0, 3)),
            keras.layers.Conv2D(filtre, (1, 7), strides=1),
            self.activation,
            keras.layers.ZeroPadding2D(padding=(3, 0)),
            keras.layers.Conv2D(192, (7, 1), strides=1),
            self.activation
        ]
        self.path2 = [
            keras.layers.Conv2D(filtre, 1, strides=1),
            self.activation,
            keras.layers.ZeroPadding2D(padding=(0, 3)),
            keras.layers.Conv2D(filtre, (1,7), strides=1),
            self.activation,
            keras.layers.ZeroPadding2D(padding=(3, 0)),
            keras.layers.Conv2D(192, (7,1), strides=1),
            self.activation,
        ]
        self.path3 = [
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding="same"),
            keras.layers.Conv2D(192, 1, strides=1),
            self.activation,
        ]
        self.path4 = [
            keras.layers.Conv2D(192, 1, strides=1),
            self.activation,
        ]

    def call(self, inputs):
        p1 = inputs
        for layer in self.path1:
            p1 = layer(p1)
        p2 = inputs
        for layer in self.path2:
            p2 = layer(p2)
        p3 = inputs
        for layer in self.path3:
            p3 = layer(p3)
        p4 = inputs
        for layer in self.path4:
            p4 = layer(p4)
        return keras.layers.Concatenate(axis=-1)([p1, p2, p3, p4])



class InceptionBlock3(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get("relu")
        self.path1 = [
            keras.layers.Conv2D(448, 1, strides=1),
            self.activation,
            keras.layers.Conv2D(384, (3,3), strides=1,padding="same"),
            self.activation
        ]
        self.path1_sus = [
            keras.layers.ZeroPadding2D(padding=(0, 1)),
            keras.layers.Conv2D(384, (1, 3), strides=1),
            self.activation,
        ]
        self.path1_jos = [
            keras.layers.ZeroPadding2D(padding=(1, 0)),
            keras.layers.Conv2D(384, (3, 1), strides=1),
            self.activation,
        ]
        self.path2 = [
            keras.layers.Conv2D(384, 1, strides=1),
            self.activation,
        ]
        self.path2_sus = [
            keras.layers.ZeroPadding2D(padding=(0, 1)),
            keras.layers.Conv2D(384, (1, 3), strides=1),
            self.activation,
        ]
        self.path2_jos = [
            keras.layers.ZeroPadding2D(padding=(1, 0)),
            keras.layers.Conv2D(384, (3, 1), strides=1),
            self.activation,
        ]
        self.path3 = [
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding="same"),
            keras.layers.Conv2D(192, 1, strides=1),
            self.activation,
        ]
        self.path4 = [
            keras.layers.Conv2D(320, 1, strides=1),
            self.activation,
        ]

    def call(self, inputs):
        # calea 1
        p1 = inputs
        for layer in self.path1:
            p1 = layer(p1)
        p1_sus = p1
        for layer in self.path1_sus:
            p1_sus = layer(p1_sus)
        p1_jos = p1
        for layer in self.path1_jos:
            p1_jos = layer(p1_jos)
        p1 = keras.layers.Concatenate(axis=-1)([p1_sus,p1_jos])
        # calea 2
        p2 = inputs
        for layer in self.path2:
            p2 = layer(p2)
        p2_sus = p2
        for layer in self.path2_sus:
            p2_sus = layer(p2_sus)
        p2_jos = p2
        for layer in self.path2_jos:
            p2_jos = layer(p2_jos)
        p2 = keras.layers.Concatenate(axis=-1)([p2_sus, p2_jos])
        # calea 3
        p3 = inputs
        for layer in self.path3:
            p3 = layer(p3)
        # calea 4
        p4 = inputs
        for layer in self.path4:
            p4 = layer(p4)
        return keras.layers.Concatenate(axis=-1)([p1, p2, p3, p4])
