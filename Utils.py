
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

import glob


N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']



def convertfromNameOfDiseazeToOneHot( rowDataset):
    indexes = []
    for x in rowDataset.split("|"):
        try:
            indexes.append(CLASS_NAMES.index(x))
            onehot = tf.reduce_max(tf.one_hot(indexes, N_CLASSES, dtype=tf.int32), axis=0)
            return onehot.numpy();
        except ValueError:
            return np.zeros(15);




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
    if(len(bands)>1):
        image = image.convert('1')

    return np.array(image.getdata()).reshape(
        (im_height, im_width, 1)).astype(np.uint8)

    # def rgbOrrgba2gray(rgb):
    #     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    #     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #
    #     return gray