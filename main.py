import os
import tarfile
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import shutil
import csv
import wget
import zipfile
from google_drive_downloader import GoogleDriveDownloader
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
import scipy
import urllib.request
import pandas as pd
import Utils as util


def main():
    downloadAndPrepareWorkspace()
    print(f"There are {len(os.listdir('./images'))} images.")
    #there are 112120 images split in train and Test sets based on the lists in metadata
    #boundry box data is available only for 988
    #in the 1st version we will create a new dataset of 988 with Boudry box + 1012 random images for training (2k) and 200 for test
    trainingList = []
    testList = []
    trainigLabels = [];
    testLabels = [];
    trainingBbox= [];
    testingBbox = [];

    train_images_np = []
    test_images_np = []
    imageLocation = "./images"

    # df = pd.read_csv('./metadata/BBox_List_2017 (1).csv', index_col=0)
    # i = os.path.exists('./metadata/BBox_List_2017 (1).csv');
    # before we train with the full dataset we train with a smaller dataset for development purposes , this will be all dataset with Bounding box + 1001 examples without bounding box
    with open('./metadata/BBox_List_2017 (1).csv') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        next(csvReader)
        for row in csvReader:
            trainingList.append(row[0])
            trainigLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
            trainingBbox.append(np.array(row[2:]))

    # df = pd.read_csv('./metadata/trainGabi1012.csv', index_col=0)
    with open('./metadata/trainGabi1012.csv') as csvfile1:
        csvReader = csv.reader(csvfile1, delimiter=',')
        for row in csvReader:
            trainingList.append(row[0])
            trainigLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
    with open('./metadata/testgabi200.csv') as csvfile2:
        csvReader = csv.reader(csvfile2, delimiter=',')
        for row in csvReader:
            testList.append(row[0])
            testLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))


    for image in trainingList:
        image_path = os.path.join(imageLocation,image)
        train_images_np.append(util.load_image_into_numpy_array(image_path))
    for image in testList:
        image_path = os.path.join(imageLocation,image)
        test_images_np.append(util.load_image_into_numpy_array(image_path))

    #Now put all the data in dataset so the model can load it as input
    print("TOP")














def downloadAndPrepareWorkspace():
    # Define the training and validation base directories
    train_dir = './dataset/training'
    source_path = './'
    source_list_names = os.listdir(source_path)
    temp_path = os.path.join(source_path, 'temp')
    metadata_path = os.path.join(source_path, 'metadata')
    tempAlreadyExists = os.path.exists(temp_path)
    metaDataAlreadyExists = os.path.exists(metadata_path)
    if not tempAlreadyExists:
        os.makedirs(temp_path)
    if not metaDataAlreadyExists:
        os.makedirs(metadata_path)
    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    # check if files were already downloaded and moved to temp
    if len(os.listdir(temp_path)) == 0:
        for idx, link in enumerate(links):
            fn = 'images_%02d.tar.gz' % (idx + 1)
            print('downloading' + fn + '...')
            urllib.request.urlretrieve(link, fn)  # download the zip file
        # on first download move to temp the arhives of the images
        for filenameX in source_list_names:
            if filenameX.endswith("tar.gz"):
                dest = shutil.move(os.path.join(source_path, filenameX), os.path.join(source_path, temp_path))
        temp_list_names = os.listdir(temp_path)
        for filenameTemp in temp_list_names:
            if filenameTemp.endswith("tar.gz"):
                tar = tarfile.open(os.path.join(temp_path, filenameTemp), "r:gz")
                tar.extractall()
                tar.close()
    # !wget - -no - check - certificate
    # 'https://drive.google.com/uc?export=download&id=1QcGwCPZDl-soNlKXaCQcVFMhYnD1dP-U' - O
    # 'metadata.rar'
    originalDriveLink = 'https://drive.google.com/file/d/1QcGwCPZDl-soNlKXaCQcVFMhYnD1dP-U/view?usp=sharing'
    driveLink = 'https://drive.google.com/uc?export=download&id=1K-_oj7G9sLqTOt_IKoy6TanNKDLJ_occ'
    # GoogleDriveDownloader.download_file_from_google_drive(file_id='1QcGwCPZDl-soNlKXaCQcVFMhYnD1dP-U',
    #                                 dest_path='./temp',
    #                                 unzip=True)
    metaDataExists = os.path.exists('./metadata.zip')
    if not metaDataExists:
        file_name = wget.download(driveLink)
        print(file_name)
    if len(os.listdir('./metadata')) == 0:
        with zipfile.ZipFile('./metadata.zip', 'r') as zip_ref:
            zip_ref.extractall(metadata_path)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
