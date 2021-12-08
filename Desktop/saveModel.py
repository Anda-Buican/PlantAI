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

def saveModel():
    model = keras.models.load_model('Retele/model4')
    convertor = tf.lite.TFLiteConverter.from_saved_model(model)
    tflModel = convertor.convert()
    with open('liteModel(25.07.2021).tflite', 'wb') as f:
        f.write(tflModel)

saveModel()
