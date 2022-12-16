import math

import sklearn
import tensorflow as tf
from PIL.Image import Image
import os
import random
import zipfile
import io
import scipy.misc
import numpy as np
import imageio
from matplotlib import pyplot as plt
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, \
    roc_curve, auc
from tensorflow.python.util.compat import as_text
import seaborn as sns

BATCH_SIZE = 8

import glob

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def convertFromPathAndLabelToTensor(imagepath, label):
    image = load_image_into_numpy_array(imagepath)
    return image, label


def convertfromNameOfDiseazeToOneHot(rowDataset):
    indexes = []
    try:
        for x in rowDataset.split("|"):
            indexes.append(CLASS_NAMES.index(x))
        onehot = tf.reduce_max(tf.one_hot(indexes, N_CLASSES, dtype=tf.float32), axis=0)

        return onehot
    except ValueError:
        return tf.convert_to_tensor(np.zeros(14), dtype=tf.float32)


def load_image_into_numpy_array1(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)

    in the paper they resized to 512 512 and normalized with the mean and Variance of Imagenet
    Before inputting the images into the network,
    the images were resized to 512 pixels by 512 pixels and normalized based on the mean and standard deviation (SD) of images in the ImageNet training set.
    layer = tensorflow.keras.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406],
                                                                      variance=[np.square(0.299),
                                                                                np.square(0.224),
                                                                                np.square(0.225)])
    https://stackoverflow.com/questions/67480507/tensorflow-equivalent-of-pytorchs-transforms-normalize --we use from here the 3rd option of taking the average Std and mean
    255 because 8 bit  0.449 avg mean 0.226 avg std

    I tried to read the image as Gray initially then copy the image in all 3 channels
    """

    image = tf.keras.utils.load_img(
        path,
        grayscale=False,
        color_mode='grayscale',
        target_size=None,
        interpolation='nearest',
        keep_aspect_ratio=False
    )
    # image = Image.open(io.BytesIO(img_data))

    bands = Image.Image.getbands(image)
    (im_width, im_height) = image.size
    # we have some images that are RGBA so we need to standardize them to 1 channel like the rest
    if (len(bands) > 3):
        image = image.convert('1')

    image = image.resize((512, 512))
    # image = np.array(image.getdata()).reshape((im_height, im_width)).astype(np.uint8)
    # image = ((image /255.0)-0.449)/0.226
    image = np.array(image.getdata()).reshape((512, 512)).astype(np.uint8)
    copyChannels = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    copyChannels[:, :, 0] = image
    copyChannels[:, :, 1] = image
    copyChannels[:, :, 2] = image
    image = ((tf.cast(copyChannels, tf.float32) / 255.0) - 0.449) / 0.226

    return image


def load_image_into_numpy_array(path):
    """
    in the paper they resized to 512 512 and normalized with the mean and Variance of Imagenet
    Before inputting the images into the network,
    the images were resized to 512 pixels by 512 pixels and normalized based on the mean and standard deviation (SD) of images in the ImageNet training set.

    A simpler way just in case
    https://stackoverflow.com/questions/67480507/tensorflow-equivalent-of-pytorchs-transforms-normalize --we use from here the 3rd option of taking the average Std and mean
    255 because 8 bit  0.449 avg mean 0.226 avg std
    """
    img = tf.io.read_file(path)
    image1 = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(image1, size=(512, 512), preserve_aspect_ratio=True, method='nearest')

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tf.cast(img, tf.float32) / 255.0
    img = img - mean
    image = img / std

    return image


def load_image_into_numpy_arrayNoNormalized(path):
    img = tf.io.read_file(path)
    image1 = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(image1, size=(512, 512), preserve_aspect_ratio=True, method='nearest')
    return img


def preProcessImage(image):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)

    in the paper they resized to 512 512 and normalized with the mean and Variance of Imagenet
    Before inputting the images into the network,
    the images were resized to 512 pixels by 512 pixels and normalized based on the mean and standard deviation (SD) of images in the ImageNet training set.

    A simpler way just in case
    https://stackoverflow.com/questions/67480507/tensorflow-equivalent-of-pytorchs-transforms-normalize --we use from here the 3rd option of taking the average Std and mean
    255 because 8 bit  0.449 avg mean 0.226 avg std

    image = np.array(image.getdata()).reshape((im_height, im_width)).astype(np.uint8)
    image = ((image /255.0)-0.449)/0.226
    image = ((tf.cast(np.array(image), tf.float32) / 255.0) - 0.449) / 0.226
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
    image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
    image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
    image = tf.cast(image, tf.float32)
    return image

    # def rgbOrrgba2gray(rgb):
    #     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    #     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #
    #     return gray


def plot_cm(testLabels, predictions, p=0.5):
    labels = CLASS_NAMES

    conf_mat_dict = {}

    for label_col in range(len(labels)):
        y_true_label = testLabels[:, label_col]
        y_pred_label = predictions[:, label_col]
        y_pred_label = np.where(y_pred_label > p, 1, 0)
        cm = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label, labels=[0, 1])
        conf_mat_dict[labels[label_col]] = cm

    f, axes = plt.subplots(2, 7, figsize=(40, 20))
    m = 0
    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)
        j = m
        i = m
        i = math.floor(i / 7)
        j = j % 7
        # ia cm si fa
        # sensitivity = Number of Correctly Predicted Positives / Number of Actual Positives
        # Specificity =Number of Correctly Predicted Negatives / Number of Actual Negatives
        tp = matrix[1][1]
        tn = matrix[0][0]
        fn = matrix[1][0]
        fp = matrix[0][1]
        sensitivity = round(tp / (tp + fn), 2)
        specificity = round(tn / (tn + fp), 2)

        disp = ConfusionMatrixDisplay(matrix,
                                      display_labels=None)
        disp.plot(ax=axes[i][j], xticks_rotation=45)

        disp.ax_.set_title(label + ' \n sensitivity ' + str(sensitivity) + ' \nspecificity ' + str(specificity))
        m += 1

    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()


def multiclass_roc_auc_score(testLabels, predictions, average="macro"):
    labels = CLASS_NAMES

    f, axes = plt.subplots(2, 7, figsize=(60, 30))
    m = 0
    for label_col in range(len(labels)):
        y_true_label = testLabels[:, label_col]
        y_pred_label = predictions[:, label_col]
        j = m
        i = m
        i = math.floor(i / 7)
        j = j % 7

        fp, tp, thresholds = roc_curve(y_true_label.astype(int), y_pred_label)
        print(labels[m] + ' ' + str(auc(fp, tp)))
        axes[i][j].plot(fp, tp, label='%s (AUC:%0.2f)' % (labels[m], auc(fp, tp)))
        axes[i][j].legend()
        axes[i][j].set_xlabel('False Positive Rate')
        axes[i][j].set_ylabel('True Positive Rate')
        m += 1
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()


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


def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed
