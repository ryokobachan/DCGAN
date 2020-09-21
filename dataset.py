import tensorflow as tf
import glob
import numpy as np
from generator import make_generator_model,generator_loss
from discriminator import make_discriminator_model,discriminator_loss
from PIL import Image

def mnist_dataset():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    Images = []

    for i in train_images:
        img = train_images[i]
        img = Image.fromarray(np.uint8(img),mode='RGBA')
        img = img.convert("RGB")
        imgarr = np.asarray(img.resize((28, 28)))
        Images.append(imgarr)

    train_images = np.array(Images)

    return train_images

def dir_dataset(path):
    files = glob.glob(path)
    Images = []

    for file in files:
        img = Image.open(file)
        #img = img.convert("L").convert("RGB")
        img = img.convert("RGB")
        imgarr = np.asarray(img.resize((28, 28)))
        Images.append(imgarr)

    train_images = np.array(Images)

    return train_images

def mnist_dataset_gray():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    return train_images
