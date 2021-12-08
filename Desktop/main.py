import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow_datasets as tfds
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# fix random seed for reproducibility
seed = 21
numpy.random.seed(seed)

# load data
(train_dataset,test_dataset), info = tfds.load('oxford_flowers102', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True)

# normalize inputs from 0-255 to 0.0-1.0
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train / 255.0
# X_test = X_test / 255.0
def normalImg(image, label):
    image = tf.image.resize(image,(128,128))
    return tf.cast(image, tf.float32)/255.0, label
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
train_dataset = train_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(info.splits["train"].num_examples)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(AUTOTUNE)

test_dataset = test_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)
test_dataset = test_dataset.batch(128)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
num_classes = 102

# Create the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128,128,3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(num_classes))
model.add(Activation('softmax'))

epochs = 20
# optimizer = 'Adam'
lr = 0.001

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

print(model.summary())

model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

# Final evaluation of the model

scores = model.evaluate(test_dataset, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))