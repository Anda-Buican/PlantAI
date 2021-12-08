import tensorflow as tf

OXFORDCLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
                 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
                 "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
                 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
                 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
                 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
                 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
                 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
                 'bolero deep blue',
                 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy',
                 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
                 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush',
                 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
                 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus',
                 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
                 'tree mallow',
                 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss', 'foxglove',
                 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
                 'trumpet creeper',
                 'blackberry lily']
MYCLASSES = ['Margareta','Papadie','Reject', 'Trandafir', 'Floarea-soarelui', 'Lalea']


def normalImg(image, label, resize=None):
    if resize is not None:
        image = tf.image.resize(image, (resize, resize))
    return tf.cast(image, tf.float32) / 255.0, label

def normalImg2(image, label, resize=224):
    if resize is not None:
        image = tf.image.resize(image, (resize, resize))
    return tf.cast(image, tf.float32) / 255.0, label


def augmentation(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:  # 10% din cazuri
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 1, 3])  # transformam imaginea din color in alb-negru
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:  # 10% din cazuri
        image = tf.image.flip_left_right(image)  # flip imagine
    return image, label
