import math
import os
import tarfile
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
import shutil
import csv
import wget
import zipfile
import cv2
from google_drive_downloader import GoogleDriveDownloader
from keras.applications.densenet import DenseNet121
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
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
trainModel = False

def main():
    downloadAndPrepareWorkspace()
    print(f"There are {len(os.listdir('./images'))} images.")
    # there are 112120 images split in train and Test sets based on the lists in metadata
    # boundry box data is available only for 988
    # in the 1st version we will create a new dataset of 988 with Boudry box + 1012 random images for training (2k) and 200 for test
    trainingList = []
    testList = []
    trainigLabels = [];
    testLabels = [];
    trainingBbox = [];
    testingBbox = [];

    train_images_np = []
    test_images_np = []
    imageLocation = "./images"

    # df = pd.read_csv('./metadata/BBox_List_2017 (1).csv', index_col=0)
    # i = os.path.exists('./metadata/BBox_List_2017 (1).csv');
    # before we train with the full dataset we train with a smaller dataset for development purposes , this will be all dataset with Bounding box + 1001 examples without bounding box
    # I took 185 of the data entries from BBOX list and put them in a new file to create a test BBOX list

    # Training data
    # removed until ready for server
    # with open('./metadata/BBox_List_2017 (1).csv') as csvfile:
    #     csvReader = csv.reader(csvfile, delimiter=',')
    #     next(csvReader)
    #     for row in csvReader:
    #         trainingList.append(row[0])
    #         trainigLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
    #         trainingBbox.append(np.array(row[2:]))
    #
    # # df = pd.read_csv('./metadata/trainGabi1012.csv', index_col=0)
    # with open('./metadata/trainGabi1012.csv') as csvfile1:
    #     csvReader = csv.reader(csvfile1, delimiter=',')
    #     for row in csvReader:
    #         trainingList.append(row[0])
    #         trainigLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
    #
    # # Test data
    # with open('./metadata/BBox_List_2017_TEST.csv') as csvfile2:
    #     csvReader = csv.reader(csvfile2, delimiter=',')
    #     for row in csvReader:
    #         testList.append(row[0])
    #         testLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
    #         testingBbox.append(np.array(row[2:]))
    # with open('./metadata/testgabi200.csv') as csvfile3:
    #     csvReader = csv.reader(csvfile3, delimiter=',')
    #     for row in csvReader:
    #         testList.append(row[0])
    #         testLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))

    # Training data
    with open('./metadata/trainGabiDev80.csv') as csvfile1:
        csvReader = csv.reader(csvfile1, delimiter=',')
        for row in csvReader:
            trainingList.append(row[0])
            trainigLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))

    # # Test data
    with open('./metadata/testGabiDev20.csv') as csvfile3:
        csvReader = csv.reader(csvfile3, delimiter=',')
        for row in csvReader:
            testList.append(row[0])
            testLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))

    for image in trainingList:
        image_path = os.path.join(imageLocation, image)
        train_images_np.append(util.load_image_into_numpy_array(image_path))
    for image in testList:
        image_path = os.path.join(imageLocation, image)
        test_images_np.append(util.load_image_into_numpy_array(image_path))

    # Now put all the data in dataset so the model can load it as input
    print("TOP")
    train_images_np = tf.convert_to_tensor(train_images_np)
    test_images_np = tf.convert_to_tensor(test_images_np)
    # trainigLabels= tf.convert_to_tensor(np.array(trainigLabels))
    # testLabels= tf.convert_to_tensor(np.array(testLabels))
    trainingDataset = util.getDataset(train_images_np, trainigLabels, trainingBbox)
    testDataset = util.getDataset(test_images_np, testLabels, testingBbox)

    # DenseNet used  224×224 but the model in the paper 512X512
    # I copied the 2d Grayscale image 3 times to replicate it being in 3 chanebecause to use imagenet weiths we need 3 chaneles

    # We trained the networks with minibatches of size 8 and used an initial learning rate of 0.0001
    # that was decayed by a factor of 10 each time the loss on the tuning set plateaued after an epoch (a full pass over the training set).
    # In order to prevent the networks from overfitting,
    # early stopping was performed by saving the network after every epoch and choosing the saved network with the lowest loss on the tuning set.
    model = define_compile_model()
    model.summary()
    # densenetModel = model.layers[1]
    # densenetModel.summary()
    EPOCHS = 1
    BATCH_SIZE = 8
    # steps_per_epoch =tf.data.experimental.cardinality(trainingDataset).numpy()
    # validation_steps=tf.data.experimental.cardinality(testDataset).numpy()

    steps_per_epoch = math.ceil(len(trainigLabels) / BATCH_SIZE)
    validation_steps = math.ceil(len(testLabels) / BATCH_SIZE)

    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                          patience=5, verbose=1, min_lr=0.00001)
    checkpoint_filepath = './checkpoint/'

    # checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)


    if (trainModel):
        history = model.fit(trainingDataset,
                            steps_per_epoch=steps_per_epoch, validation_data=testDataset,
                            validation_steps=validation_steps, epochs=EPOCHS,
                            callbacks=[reduce_lr_plateau, model_checkpoint_callback])
    else:
        model.load_weights(checkpoint_filepath)

    loss, acc = model.evaluate(testDataset, batch_size=BATCH_SIZE, steps=validation_steps, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    model.save("CheXnetXt_replica_gabi")

    # add heat maps and viz util
    # To generate the CAMs, images were fed into the fully trained network
    # and the feature maps from the final convolutional layer were extracted
    # A map of the most salient features used in classifying the image
    # as having a specified pathology was computed by taking the weighted sum
    # of the feature maps using their associated weights in the fully connected layer
    # select all the layers for which you want to visualize the outputs and store it in a list
    #     outputLastConv = model.get_layer('bn').output
    #     vis_model = Model(model.input, outputLastConv)

    idx = 0
    sample_image = test_images_np[idx].numpy()
    actual_label = testLabels[idx].numpy()

    vizualizeCam(actual_label, model, sample_image)


def vizualizeCam(actual_label, model, sample_image):
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]
    heatmap = get_CAM(model, sample_image_processed, actual_label, layer_name='bn')
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap * 255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    converted_img = sample_image
    super_imposed_image = cv2.addWeighted(converted_img, 0.8, heatmap.astype('float32'), 2e-3, 0.0)
    sample_activation = get_CAM_simple(model, sample_image_processed)
    f, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax[0, 0].imshow(sample_image)
    ax[0, 0].set_title(f"True label: {actual_label} \n Predicted label: {pred_label}")
    ax[0, 0].axis('off')
    ax[0, 1].imshow(sample_activation)
    ax[0, 1].set_title("Class activation map")
    ax[0, 1].axis('off')
    ax[1, 0].imshow(heatmap)
    ax[1, 0].set_title("Heat Map")
    ax[1, 0].axis('off')
    ax[1, 1].imshow(super_imposed_image)
    ax[1, 1].set_title("Activation map superimposed")
    ax[1, 1].axis('off')
    plt.tight_layout()
    plt.show()


def get_CAM(model, processed_image, actual_label, layer_name='bn'):
    # we used the last batchNormalization Layer of the Densenet Model where
    # layer_name = 'conv5_block16_concat'
    x = model.get_layer(layer_name)
    model_grad = Model([model.inputs],
                       [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output_values, predictions = model_grad(processed_image)

        # watch the conv_output_values
        tape.watch(conv_output_values)

        expected_output = actual_label
        predictions = predictions[0]
        loss = tf.keras.losses.categorical_crossentropy(
            expected_output, predictions
        )

    # nu merge bine bp aici
    # get the gradient of the loss with respect to the outputs of the last conv layer
    grads_values = tape.gradient(loss, conv_output_values)
    # grads_values = tf.keras.backend.mean(grads_values, axis=(0, 1, 2))

    conv_output_values = np.squeeze(conv_output_values.numpy())
    grads_values = np.squeeze(grads_values)

    # weight the convolution outputs with the computed gradients
    for i in range(1024):
        # conv_output_values[:, :, i] *= grads_values[i]
        conv_output_values[:, :, i] *= grads_values[:, :, i]
    heatmap = np.mean(conv_output_values, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    del model_grad, conv_output_values, grads_values, loss

    return heatmap


def get_CAM_simple(model, processed_image):
    # we used the last batchNormalization Layer of the Densenet Model where
    layer_name = 'bn'

    vis_model = Model([model.inputs],
                      [model.get_layer(layer_name).output, model.layers[-1].output])
    features, predictions = vis_model.predict(processed_image)
    features = features[0]
    gap_weights = model.layers[-1].get_weights()[0]
    class_activation_features = sp.ndimage.zoom(features, (512 / 16, 512 / 16, 1), order=2)
    # compute the intensity of each feature in the CAM
    class_activation_weights = gap_weights
    cam_output = np.dot(class_activation_features, class_activation_weights)
    cam_output = cam_output.mean(axis=-1)
    cam_output -= cam_output.mean(axis=-1)
    cam_output /= cam_output.std()
    cam_output *= 255
    cam_output = np.clip(cam_output, 0, 255).astype(np.uint8)

    return cam_output


def feature_extractor(inputs):
    feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))(inputs)

    return feature_extractor


def classifier(inputs):
    # add flatten larye because  8, 16, 16, 14 shape of final model before
    averagePool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(14, activation='softmax', name="classification")(averagePool)
    return x


def final_model(inputs):
    # resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)  Usedfull if rezise

    densenet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(densenet_feature_extractor)

    return classification_output


def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))

    feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    averagePool = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor.output)
    output = tf.keras.layers.Dense(14, activation='softmax', name="classification")(averagePool)

    # classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=feature_extractor.input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def downloadAndPrepareWorkspace():
    # Define the training and validation base directories
    train_dir = './dataset/training'
    source_path = './'
    source_list_names = os.listdir(source_path)
    temp_path = os.path.join(source_path, 'temp')
    metadata_path = os.path.join(source_path, 'metadata')
    checkpoint_path = os.path.join(source_path, 'checkpoint')
    checkpointExists = os.path.exists(checkpoint_path)
    tempAlreadyExists = os.path.exists(temp_path)
    metaDataAlreadyExists = os.path.exists(metadata_path)
    if not tempAlreadyExists:
        os.makedirs(temp_path)
    if not metaDataAlreadyExists:
        os.makedirs(metadata_path)
    if not checkpointExists:
        os.makedirs(checkpoint_path)
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
