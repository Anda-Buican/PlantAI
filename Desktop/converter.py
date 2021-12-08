import tensorflow as tf

# Convertesc modelul
convertor = tf.lite.TFLiteConverter.from_saved_model('Retele/resnet50_pre_c6')
liteModel = convertor.convert()


# salvez modelul lite
with open('resnet50_Augm_c6.tflite', 'wb') as f:
  f.write(liteModel)

