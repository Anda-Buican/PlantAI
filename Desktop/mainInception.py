import random

import numpy
import keras
from utile import *
from inceptionBlocks import *
from keras.models import Sequential
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 4

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# load data
#(test_dataset,train_dataset), info = tfds.load('oxford_flowers102', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True)
# print(info)
# print(info.features['label'].names)
# exit(0)


AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_dataset = train_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)
# train_dataset = train_dataset.cache()
# train_dataset = train_dataset.shuffle(info.splits["train"].num_examples)
# train_dataset = train_dataset.batch(BATCH_SIZE)
# train_dataset = train_dataset.prefetch(AUTOTUNE)
#

# test_dataset = test_dataset.batch(16)
# test_dataset = test_dataset.prefetch(AUTOTUNE)
seed = random.randint(0,99999)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory('flowers/',
labels = 'inferred',
label_mode= "int",
color_mode='rgb',
batch_size=BATCH_SIZE,
image_size=(299,299),
shuffle=True,
seed=seed,
validation_split= 0.2,
subset="training"
)
train_dataset = train_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)
# train_dataset = train_dataset.map(augmentation, num_parallel_calls = AUTOTUNE)



test_dataset = tf.keras.preprocessing.image_dataset_from_directory('flowers/',
labels = 'inferred',
label_mode= "int",
color_mode='rgb',
batch_size=1,
image_size=(299,299),
shuffle=True,
seed=seed,
validation_split= 0.2,
subset="validation"
)
test_dataset = test_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)

num_classes = 5

# Create the model
model = Sequential()
# Augmentari
model.add(RandomFlip("horizontal"))
model.add(RandomRotation(0.1))
model.add(RandomZoom(height_factor=(-0.2,0.2)))
model.add(RandomContrast(0.1))
# Partea initiala de convolutii
model.add(Conv2D(32, input_shape=(299,299,3), kernel_size=(3,3), strides=2))
model.add(Conv2D(32, kernel_size=(3,3), strides=1))
model.add(Conv2D(64, kernel_size=(3,3), strides=1, padding="same"))
model.add(MaxPooling2D((3,3),strides=2))
model.add(Conv2D(80, kernel_size=(3,3), strides=1))
model.add(Conv2D(192, kernel_size=(3,3), strides=2))
model.add(Conv2D(288, kernel_size=(3,3), strides=1,padding="same"))
# Continuam cu 3 blocuri Inception (Tip1)
model.add(InceptionBlock1())
model.add(InceptionBlock1())
model.add(InceptionBlock1())
# Reducem dimensiunea
model.add(InceptionReduction(64,0))
# Continuam cu 5 blocuri Inception (Tip2)
model.add(InceptionBlock2(128))
model.add(InceptionBlock2(160))
model.add(InceptionBlock2(160))
model.add(InceptionBlock2(160))
model.add(InceptionBlock2(192))
# Reducem dimensiunea
model.add(InceptionReduction(192,16))
# Continuam cu 2 blocuri Inception (Tip3)
model.add(InceptionBlock3())
model.add(InceptionBlock3())
# Final retea
model.add(keras.layers.GlobalAveragePooling2D())
model.add(Flatten())
model.add(keras.layers.Dense(num_classes, activation='softmax'))
epochs = 35
lr = 0.0001

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

# pastram istoricul pentru grafice
history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

print(model.summary())

predict=[]
labels=[]
for x, y in test_dataset:
    labels.append(y.numpy()[0])
    predict.append(np.argmax(model.predict(x),axis=1)[0])
conf = tf.math.confusion_matrix(labels, predict).numpy()
confStr=np.array2string(conf, separator=',', threshold=500)
f = open("matriceConfuzieInceptionAugm.csv", "w")
for linie in conf:
    linie.tofile(f, sep=",")
    f.write("\n")
f.close()


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('InceptionV2 Augm | Acuratete')
plt.ylabel('acuratete')
plt.xlabel('epoca')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('InceptionV2 Augm | Loss')
plt.ylabel('loss')
plt.xlabel('epoca')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Final evaluation of the model
model.save('Retele/inceptionPersAugm')
scores = model.evaluate(test_dataset, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))


