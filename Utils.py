import tensorflow as tf
from PIL.Image import Image
import os
import random
import zipfile
import io
import scipy.misc
import numpy as np
import imageio
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont


BATCH_SIZE = 32

import glob

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def convertfromNameOfDiseazeToOneHot(rowDataset):
    indexes = []
    for x in rowDataset.split("|"):
        try:
            indexes.append(CLASS_NAMES.index(x))
            onehot = tf.reduce_max(tf.one_hot(indexes, N_CLASSES, dtype=tf.int32), axis=0)
            return onehot
        except ValueError:
            return tf.convert_to_tensor(np.zeros(14), dtype=tf.int32)


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(io.BytesIO(img_data))
    bands = Image.Image.getbands(image)
    (im_width, im_height) = image.size
    # we have some images that are RGBA so we need to standardize them to 1 channel like the rest
    if (len(bands) > 3):
        image = image.convert('1')

    # return np.array(image.getdata()).reshape(
    #     (im_height, im_width)).astype(np.uint8)
    # in the paper they resized to 512 512 and normalized with the mean and Variance of Imagenet
    # Before inputting the images into the network,
    # the images were resized to 512 pixels by 512 pixels and normalized based on the mean and standard deviation (SD) of images in the ImageNet training set.
    # layer = tensorflow.keras.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406],
    #                                                                   variance=[np.square(0.299),
    #                                                                             np.square(0.224),
    #                                                                             np.square(0.225)])
    #https://stackoverflow.com/questions/67480507/tensorflow-equivalent-of-pytorchs-transforms-normalize --we use from here the 3rd option of taking the average Std and mean
    #255 because 8 bit  0.449 avg mean 0.226 avg std
    image = image.resize((512, 512))
    # image = np.array(image.getdata()).reshape((im_height, im_width)).astype(np.uint8)
    # image = ((image /255.0)-0.449)/0.226
    image = np.array(image.getdata()).reshape((512, 512)).astype(np.uint8)
    copyChannels = np.zeros(shape=(512,512,3), dtype=np.uint8)
    copyChannels[:,:,0] = image
    copyChannels[:, :, 1] = image
    copyChannels[:, :, 2] = image
    # image = ((tf.cast(np.array(image), tf.float32) / 255.0) - 0.449) / 0.226
    image = ((tf.cast(copyChannels, tf.float32) / 255.0) - 0.449) / 0.226
    return image


    # def rgbOrrgba2gray(rgb):
    #     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    #     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #
    #     return gray


def getDataset(images, oneHotLabels, bboxList):
    # not all have BBOX so no dataset object , have to put as none ?
    # dataset = tf.data.Dataset.from_tensor_slices( (images,oneHotLabels,bboxList))
    # 1800 1800 but still no , need to convert the images or everything to tensor

    dataset = tf.data.Dataset.from_tensor_slices((images, oneHotLabels))
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(BATCH_SIZE,
                            drop_remainder=True)  # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.prefetch(
        -1)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
    # xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    # ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    # image = tf.reshape(image, (28, 28, 1,))
    # image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    # image = tf.cast(image, tf.float32) / 255.0
    # xmin = tf.cast(xmin, tf.float32)
    # ymin = tf.cast(ymin, tf.float32)
    #
    # xmax = (xmin + 28) / 75
    # ymax = (ymin + 28) / 75
    # xmin = xmin / 75
    # ymin = ymin / 75
    # return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])
    return dataset
