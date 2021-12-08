#Buican Laura Andreea
#Grupa 311
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
import tensorflow as tf
from resnetBlock2 import *
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from utile import *


# Parametrii pentru rularea stabila a Tensorflow
config = tf.compat.v1.ConfigProto(gpu_options =tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# load data
(test_dataset,train_dataset), info = tfds.load('oxford_flowers102', split=['train','test'], shuffle_files=True, as_supervised=True, with_info=True)

# dimensiunea batch-ului
BATCH_SIZE = 16

# # Citire date despre poze din fisiere
# test = np.genfromtxt('test.txt', dtype=None,delimiter= ',', encoding=None) # incarcam test
# train = np.genfromtxt('train.txt', dtype=None,delimiter= ',',names=('path', 'label'), encoding=None) # incarcam train
# validation = np.genfromtxt('validation.txt', dtype=None,delimiter= ',',names=('path', 'label'), encoding=None) # incarcam validarea
#
# # caile catre array-urile cu poze salvate de mine/care vor fi salvate
# caleArrayTrain = 'train.npy'
# caleArrayTest = 'test.npy'
# caleArrayValidation = 'validation.npy'
#
#
# if os.path.exists(caleArrayTrain):  # verificam daca exista fisierul de train creat
#     # citim direct din fisierul nostru
#     trainLista = np.load(caleArrayTrain, allow_pickle=True)
# else:
#     sursaimg=train['path']
#     trainLista=[]
#     # luam fiecare poza, o deschidem cu cv2 si o adaugam la lista
#     for cale in sursaimg:
#         imagineTrain = cv2.imread("train/"+cale)
#         trainLista.append(imagineTrain)
#     trainLista = np.array(trainLista)
#     # salvam array-ul numpy ca sa nu facem acest procedeu de fiecare data
#     trainLista.dump(caleArrayTrain)
#
# if os.path.exists(caleArrayTest):  # verificam daca exista fisierul de test creat
#     # citim direct din fisierul nostru
#     testLista = np.load(caleArrayTest, allow_pickle=True)
# else:
#     sursaimg=test
#     testLista=[]
#     # luam fiecare poza, o deschidem cu cv2 si o adaugam la lista
#     for cale in sursaimg:
#         imagineTest = cv2.imread("test/"+cale)
#         testLista.append(imagineTest)
#     testLista = np.array(testLista)
#     # salvam array-ul numpy ca sa nu facem acest procedeu de fiecare data
#     testLista.dump(caleArrayTest)
#
# if os.path.exists(caleArrayValidation):  # verificam daca exista fisierul de validare creat
#     # citim direct din fisierul nostru
#     validationLista = np.load(caleArrayValidation, allow_pickle=True)
# else:
#     sursaimg=validation['path']
#     validationLista=[]
#     # luam fiecare poza, o deschidem cu cv2 si o adaugam la lista
#     for cale in sursaimg:
#         imagineValidation = cv2.imread("validation/"+cale)
#         validationLista.append(imagineValidation)
#     validationLista = np.array(validationLista)
#     # salvam array-ul numpy ca sa nu facem acest procedeu de fiecare data
#     validationLista.dump(caleArrayValidation)
#


# prelucrari initiale pt SVM, nu le mai folosesc
# trainLista = trainLista.reshape(15000,-1)
# testLista = testLista.reshape(15000,-1)
# validationLista = validationLista.reshape(15000,-1)


# # convertim listele in tensori
# trainLista=tf.convert_to_tensor(trainLista)
# # pentru fit pe train + validare
# # trainLista=tf.convert_to_tensor(np.concatenate((trainLista,validationLista)))
# testLista=tf.convert_to_tensor(testLista)
# validationLista=tf.convert_to_tensor(validationLista)



# #normalizare si augmentare
# def normalImg(image, label):
#     image = tf.image.resize(image,(100,100))
#     # return tf.cast(image, tf.float32)/255.0-0.5, label
#     return tf.image.per_image_standardization(image), label
#
# def normalImgTest(image):
#     image = tf.image.resize(image,(100,100))
#     # return tf.cast(image, tf.float32)/255.0-0.5
#     return tf.image.per_image_standardization(image)
#
#
# def augmentation(image,label):
#     image=tf.image.rgb_to_grayscale(image)
#     return image, label
#
# def augmentationTest(image):
#     image=tf.image.rgb_to_grayscale(image)
#     return image


# Parti din primele incercari pe SVM
# svm_model = svm.LinearSVC(C=10, verbose=1, max_iter=1000)

# # svm_model.fit(trainLista, train['label'])  # train
# print( "Done fitting")
# print( "--- %s seconds ---" )#% (time.time() - start_time))

# predicted_val1_labels = svm_model.predict(normalized_validation1)  # predict
# predicted_val2_labels = svm_model.predict(validationLista)  # predict


# trainDataset = tf.data.Dataset.from_tensor_slices((trainLista,train['label']))
# Am incercat si rularea pe train+validare
# trainDataset = tf.data.Dataset.from_tensor_slices((trainLista,np.concatenate((train['label'],validation['label']))))
# trainDataset = trainDataset.map(augmentation)
trainDataset = trainDataset.map(normalImg)
train_dataset = trainDataset.shuffle(reshuffle_each_iteration=True, buffer_size=15000)
trainDataset = trainDataset.batch(BATCH_SIZE)

testDataset = tf.data.Dataset.from_tensor_slices(testLista)
# testDataset = testDataset.map(augmentationTest)
testDataset = testDataset.map(normalImg)
testDataset = testDataset.batch(BATCH_SIZE)

validationDataset = tf.data.Dataset.from_tensor_slices((validationLista,validation['label']))
# validationDataset = validationDataset.map(augmentation)
validationDataset = validationDataset.map(normalImg)
validationDataset = validationDataset.batch(BATCH_SIZE)


for image,label in trainDataset:
    # verificare dimensiuni
    print(image.shape)
    print(label)
    # cv2.imshow('poza', image[0].numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    break


# RESNET IMPLEMENTAT DE MINE

# model = Sequential()
# model.add(Conv2D(64, (7, 7),(2,2), input_shape=(224,224,3), padding='same', use_bias=False))
# model.add(BatchNormalization())
# model.add(Activation(keras.activations.relu))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # straturi = [64,64,128,128,128,256,256,256,256,256,512,512,512] # 65
# # straturi = [64,64,128,128,256,256,256,512,512]  # 64-66
# straturi = [64,64,64,128,128,128,128,256,256,256,256,256,256,512,512,512]  # 71
# for i in range(len(straturi)):
#     if i!=0 and straturi[i]!=straturi[i-1]:
#         model.add(ResnBlock2(straturi[i], 2))
#     else:
#         model.add(ResnBlock2(straturi[i], 1))
#
# model.add(keras.layers.pooling.GlobalAveragePooling2D())
# model.add(Flatten())
# # model.add(Dropout(0.3)) # 0.3
# model.add(keras.layers.Dense(3, activation='softmax'))

# numarul de epoci
epochs = 10
# Rata de invatare a optimizatorului Adam
lr = 0.00005

# Model secvential pentru a adauga stratul Dense/Dropout la final
model = Sequential()

# Resnet preantrenat modificat pentru a se plia pe parametrii
# ReteaPreantrenata = tf.keras.applications.ResNet50V2(
#     include_top=False,
#     weights="imagenet",
#     input_shape=([250,250,3]),
#     pooling='avg'
# )

# Inception preantrenat modificat pentru a se plia pe parametrii
ReteaPreantrenata = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=([100,100,3]),
    pooling='avg'
)
model.add(ReteaPreantrenata)
model.add(Flatten())
# model.add(Dropout(0.3)) # 0.3
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

# Antrenare model si pastrare istoric pentru a afisa grafice (acuratete/loss)
history = model.fit(trainDataset, epochs=epochs, validation_data=validationDataset)

# prezicere pe validare pentru a afisa matricea de confuzie
validPredict = model.predict(validationDataset, verbose=1)
validPredict = np.argmax(validPredict, axis=1)
# afisare matricea de confuzie
print(tf.math.confusion_matrix(validation['label'], validPredict))

model.save('res50')

# Grafice
# Grafice acuratete
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('acuratete model')
plt.ylabel('acuratete')
plt.xlabel('epoci')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Grafice loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss model')
plt.ylabel('loss')
plt.xlabel('epoci')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluare test
testPredict = model.predict(testDataset, verbose=1)
testPredict = np.argmax(testPredict, axis=1)
print(testPredict)

f = open("predictii.txt", "w")
f.write("id,label\n")
for i in range(len(testPredict)):
    f.write(test[i]+","+str(testPredict[i])+"\n")
f.close()



