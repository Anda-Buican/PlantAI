import numpy
import keras
import numpy as np

from utile import *
from resnetBlock import ResnBlock
from resnetBlock2 import ResnBlock2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
# import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

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
# (test_dataset,train_dataset), info = tfds.load('oxford_flowers102', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True)
# print(info)
# print(info.features['label'].names)
# exit(0)



AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
train_dataset = train_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(info.splits["train"].num_examples)
# train_dataset = train_dataset.map(augmentation, num_parallel_calls = AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = test_dataset.map(normalImg, num_parallel_calls = AUTOTUNE)
test_dataset = test_dataset.batch(1)


# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
num_classes = 102

# Create the model RESNET fara preantrenare
model = Sequential()

model.add(Conv2D(64, (7, 7),(2,2), input_shape=(224,224,3), padding='same', use_bias=False))
#model.add(BatchNormalization())
model.add(Activation(keras.activations.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
straturi = [64,64,64,64,128,128,128,128,128,128,256,256,256,256,256,256,256,512,512,512,512,512]
for i in range(len(straturi)):
    if i!=0 and straturi[i]!=straturi[i-1]:
        model.add(ResnBlock2(straturi[i], 2))
    else:
        model.add(ResnBlock2(straturi[i], 1))

model.add(keras.layers.pooling.GlobalAveragePooling2D())
model.add(Flatten())
model.add(keras.layers.Dense(num_classes, activation='softmax'))



# Model secvential pentru a adauga stratul Dense/Dropout la final
model = Sequential()

# Resnet preantrenat modificat pentru a se plia pe parametrii
ReteaPreantrenata = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=([250,250,3]),
    pooling='avg'
)

# Inception preantrenat modificat pentru a se plia pe parametrii
# ReteaPreantrenata = tf.keras.applications.InceptionV3(
#     include_top=False,
#     weights="imagenet",
#     input_shape=([224,224,3]),
#     pooling='avg'
# )
model.add(ReteaPreantrenata)
model.add(Flatten())
# model.add(Dropout(0.3)) # 0.3
model.add(keras.layers.Dense(3, activation='softmax'))


epochs = 10
# optimizer = 'Adam'
lr = 0.0001

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

# print(model.summary())

history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
predict=[]
labels=[]
for x, y in test_dataset:
    labels.append(y.numpy()[0])
    predict.append(np.argmax(model.predict(x),axis=1)[0])
conf = tf.math.confusion_matrix(labels, predict).numpy()
confStr=np.array2string(conf, separator=',', threshold=500)
f = open("matriceConfuzie.txt", "w")
for linie in conf:
    linie.tofile(f, sep=",")
    f.write("\n")
f.close()




# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Final evaluation of the model
model.save('Retele/resNet50')
scores = model.evaluate(test_dataset, verbose=2)
print("Accuracy: %.2f%%" % (scores[1]*100))


