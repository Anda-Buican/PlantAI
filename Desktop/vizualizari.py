import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from utile import *
import tensorflow_datasets as tfds
from sklearn import svm, datasets

if False:
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory('flowers/',
    labels = 'inferred',
    label_mode= "int",
    color_mode='rgb',
    batch_size=1,
    image_size=(1000,1000),
    shuffle=True,
    seed=111,
    validation_split= 0.2,
    subset="training"
    )
else:
    (test_dataset, train_dataset), info = tfds.load('oxford_flowers102', split=['train', 'test'], shuffle_files=True,
                                                    as_supervised=True, with_info=True)



if False:
    ####### Resize
    plt.figure(figsize=(30, 6))
    image, label = next(iter(train_dataset))
    dimensiuni = [500, 25, 50, 75, 100]
    for i in range(1, 6):
        resize_and_rescale = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(dimensiuni[i - 1], dimensiuni[i - 1]),
            layers.experimental.preprocessing.Rescaling(1. / 255)
        ])
        augmented_image = resize_and_rescale(image)
        # Prima poza
        ax = plt.subplot(1, 5, i)
        plt.title("Imagine "+str(dimensiuni[i-1])+"x"+str(dimensiuni[i-1]), fontsize=20)
        plt.imshow(augmented_image[0])
        plt.axis("off")

iter = iter(train_dataset)
####### Show
plt.figure(figsize=(20, 20))
for i in range(1, 7):
    image, label = next(iter)
    # Prima poza
    ax = plt.subplot(3, 2, i)
    plt.title(OXFORDCLASSES[np.array(label)], fontsize=40)
    plt.imshow(np.array(image).astype(int))
    plt.axis("off")
plt.show()

if False:
    ###### Augmentari
    plt.figure(figsize=(30, 30))
    image, label = next(iter(train_dataset))
    augmentari = tf.keras.Sequential([
            layers.experimental.preprocessing.Resizing(224, 224),
            layers.experimental.preprocessing.Rescaling(1. / 255),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(height_factor=(-0.2,0.2)),
            layers.experimental.preprocessing.RandomContrast(0.2)
        ])

    for i in range(16):

        augmented_image = augmentari(image)
        # Prima poza
        ax = plt.subplot(4, 4, i+1)
        # plt.title("Imagine augmentari", fontsize=20)
        plt.imshow(augmented_image[0])
        plt.axis("off")
    plt.show()


if False:
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]  # we only take the first two features. We could
                          # avoid this ugly slicing by using a two-dim dataset
    y = iris.target

    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    poly_svc = svm.SVC(kernel='poly', degree=4, C=C).fit(X, y)
    lin_svc = svm.LinearSVC(C=C,max_iter=2000).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC cu kernel liniar',
              'LinearSVC',
              'SVC cu kernel RBF',
              'SVC cu kernel polynomial(grad 4)']


    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.seismic, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.seismic)
        plt.xlabel('Lungimea petalei')
        plt.ylabel('Latimea petalei')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
