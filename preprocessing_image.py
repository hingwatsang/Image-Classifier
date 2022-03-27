from PIL import Image
import tensorflow as tf
import numpy as np
from argument import image_path

im = Image.open(image_path)
im = np.asarray(im)
im = tf.image.resize(im, (224, 224))
im /= 255.
im = np.expand_dims(im, axis=0)
