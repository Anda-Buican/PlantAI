import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from utile import *
import numpy as np

config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

poze_reprez = ["daisy/5794839_200acd910c_n.jpg","dandelion/13920113_f03e867ea7_m.jpg",
               "rose/145862135_ab710de93c_n.jpg","sunflower/40410814_fba3837226_n.jpg","tulip/142235237_da662d925c.jpg"]

def predictImg(topN, path):
    img= image.load_img(path)
    img_original = img
    model = keras.models.load_model('Retele/inceptionV3_6')
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img,axis=0)
    img,_ = normalImg(img, "", 299)
    val = model.predict(img)
    # extragem clasa maxima
    print("Clasa prezisa:")
    index = np.argsort(-val[0])[:topN]
    print(index)
    ax = plt.subplot(2, 3, 2)
    plt.imshow(img_original)
    plt.title("Imagine introdusa")
    plt.axis("off")

    x = 0
    for i in index:
        ax = plt.subplot(2, 3, 4+x)
        x += 1
        plt.imshow(image.load_img("flowers/"+poze_reprez[i]))
        plt.title("{poz}. {clasa} {acc:.2f} %".format(poz=x, clasa=MYCLASSES[i], acc=val[0][i]))
        plt.axis("off")
        print(MYCLASSES[i])
    plt.show()


predictImg(3,'flowers/rose/410421672_563550467c.jpg')