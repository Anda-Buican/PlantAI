import random

from tensorflow.python.keras.layers import *

import time
from utile import *
from resnetBlock import ResnBlock
from resnetBlock2 import ResnBlock2
from inceptionBlocks import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
# import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

# Variabile de configurare
class Retele(Enum):
    RESNET50 = 1
    INCEPTIONV3 = 2
    PERSONALA = 3
    INCEPTIONPERS = 4

BATCH_SIZE = 4
DIM_IMG = 299
RETEA_RULARE = Retele.INCEPTIONPERS
RETEA_DENUMIRE = "inception_v2_c6"
SALVEAZA = True

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# load data
# (test_dataset,train_dataset), info = tfds.load('oxford_flowers102', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True)
# print(info)
# print(info.features['label'].names)
# exit(0)


AUTOTUNE = tf.data.experimental.AUTOTUNE
# train_dataset = train_dataset.map(normalImg2, num_parallel_calls = AUTOTUNE)
# train_dataset = train_dataset.cache()
# train_dataset = train_dataset.shuffle(info.splits["train"].num_examples)
# train_dataset = train_dataset.batch(BATCH_SIZE)
# train_dataset = train_dataset.prefetch(AUTOTUNE)
#
#
# test_dataset = test_dataset.batch(1)
# test_dataset = test_dataset.map(normalImg2, num_parallel_calls = AUTOTUNE)
# test_dataset = test_dataset.prefetch(AUTOTUNE)
seed = random.randint(0,99999)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory('flowers/',
labels = 'inferred',
label_mode= "int",
color_mode='rgb',
batch_size=BATCH_SIZE,
image_size=(DIM_IMG,DIM_IMG),
shuffle=True,
seed=64,
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
image_size=(DIM_IMG,DIM_IMG),
shuffle=True,
seed=64,
validation_split= 0.2,
subset="validation"
)
test_dataset = test_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)

num_classes = 6

# Create the model
model = Sequential()

# constructie retea in functie de tipul dorit
if RETEA_RULARE == Retele.PERSONALA:
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(height_factor=(-0.2,0.2)))
    model.add(RandomContrast(0.1))
    model.add(Conv2D(64, (7, 7),(2,2), input_shape=(224,224,3), padding='same', use_bias=False))
    model.add(Activation(keras.activations.relu))
    model.add(MaxPooling2D(pool_size=(3, 3),padding="same",strides=2))
    straturi = [64,64,64,128,128,128,128,256,256,256,256,256,256,512,512,512]
    for i in range(len(straturi)):
        if i!=0 and straturi[i]!=straturi[i-1]:
            model.add(ResnBlock2(straturi[i], 2))
        else:
            model.add(ResnBlock2(straturi[i], 1))

    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    epochs = 35
elif RETEA_RULARE == Retele.INCEPTIONPERS:
    # Augmentari
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(height_factor=(-0.2, 0.2)))
    model.add(RandomContrast(0.1))
    # Partea initiala de convolutii
    model.add(Conv2D(32, input_shape=(299, 299, 3), kernel_size=(3, 3), strides=2, activation="relu"))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=1,activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
    model.add(MaxPooling2D((3, 3), strides=2))
    model.add(Conv2D(80, kernel_size=(3, 3), strides=1, activation="relu"))
    model.add(Conv2D(192, kernel_size=(3, 3), strides=2, activation="relu"))
    model.add(Conv2D(288, kernel_size=(3, 3), strides=1, padding="same", activation="relu"))
    # Continuam cu 3 blocuri Inception (Tip1)
    model.add(InceptionBlock1())
    model.add(InceptionBlock1())
    model.add(InceptionBlock1())
    # Reducem dimensiunea
    model.add(InceptionReduction(64, 0))
    # Continuam cu 5 blocuri Inception (Tip2)
    model.add(InceptionBlock2(128))
    model.add(InceptionBlock2(160))
    model.add(InceptionBlock2(160))
    model.add(InceptionBlock2(160))
    model.add(InceptionBlock2(192))
    # Reducem dimensiunea
    model.add(InceptionReduction(192, 16))
    # Continuam cu 2 blocuri Inception (Tip3)
    model.add(InceptionBlock3())
    model.add(InceptionBlock3())
    # Final retea
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    epochs = 35
elif RETEA_RULARE == Retele.RESNET50:
    ReteaPreantrenata = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=([224, 224, 3]),
        pooling='avg'
    )
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(height_factor=(-0.2, 0.2)))
    model.add(RandomContrast(0.1))
    model.add(ReteaPreantrenata)
    model.add(Flatten())
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    epochs = 10
elif RETEA_RULARE == Retele.INCEPTIONV3:
    ReteaPreantrenata = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=([299,299,3]),
        pooling='avg'
    )
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.1))
    model.add(RandomZoom(height_factor=(-0.2, 0.2)))
    model.add(RandomContrast(0.1))
    model.add(ReteaPreantrenata)
    model.add(Flatten())
    # model.add(Dropout(0.3)) # 0.3
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    epochs = 10

lr = 0.00001

start = time.time()

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])

# pastram istoricul pentru grafice
istoric = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

done = time.time()
elapsed = done - start

# Afisare structura model
print(model.summary())

# Afisare timp trecut
print("Durata antrenare")
print(str(int(elapsed/60)) + ":"+ str(int(elapsed)%60))

predict=[]
labels=[]
for x, y in test_dataset:
    labels.append(y.numpy()[0])
    predict.append(np.argmax(model.predict(x),axis=1)[0])
conf = tf.math.confusion_matrix(labels, predict).numpy()
confStr=np.array2string(conf, separator=',', threshold=500)
f = open("MatriceConf/matriceConfuzie_"+RETEA_DENUMIRE+".txt", "w")
for linie in conf:
    linie.tofile(f, sep=",")
    f.write("\n")
f.close()


# Istoricul pentru acuratete
plt.plot(istoric.history['accuracy'])
plt.plot(istoric.history['val_accuracy'])
plt.title(RETEA_DENUMIRE + ' | Acuratete in functie de epoca')
plt.ylabel('acuratete')
plt.xlabel('epoca')
plt.legend(['antrenare', 'validare'], loc='upper left')
plt.show()
# Istoricul pentru loss
plt.plot(istoric.history['loss'])
plt.plot(istoric.history['val_loss'])
plt.title(RETEA_DENUMIRE + ' | Loss in functie de epoca')
plt.ylabel('loss')
plt.xlabel('epoca')
plt.legend(['antrenare', 'validare'], loc='upper left')
plt.show()

# Final evaluation of the model
if SALVEAZA:
    model.save('Retele/'+RETEA_DENUMIRE)
scores = model.evaluate(test_dataset, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))


