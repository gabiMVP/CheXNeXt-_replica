import math
import os
import tarfile
import zipfile
from random import random
import tabulate
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf
import shutil
import csv
import wget
import zipfile
import cv2
import tensorflow_ranking as tfr
from google_drive_downloader import GoogleDriveDownloader
from keras.applications.densenet import DenseNet121
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
import urllib.request
import pandas as pd
import Utils as util

trainModel = True


def main():
    downloadAndPrepareWorkspace()
    BATCH_SIZE = 8
    trainingMode = 0
    print(f"There are {len(os.listdir('./images'))} images.")
    # there are 112120 images split in train and Test sets based on the lists in metadata
    # boundry box data is available only for 988
    # in the 1st version we will create a new dataset of 988 with Boudry box + 1012 random images for training (2k) and 200 for test
    trainingList = []
    testList = []
    trainingImgages = []
    testListImages = []
    trainigLabels = [];
    testLabels = [];
    trainingBbox = [];
    testingBbox = [];
    testLabelsString = []
    trainigLabelsString = []
    train_images_np = []
    test_images_np = []
    imageLocation = "./images"

    # Load Data from Lists

    with open('./metadata/train_val_list.txt', "r") as f:
        trainingList = [line.strip() for line in f.read().split('\n')]
    with open('./metadata/test_list.txt', "r") as f:
        testList = [line.strip() for line in f.read().split('\n')]

    # get the data + convert the labest to one hot
    with open('./metadata/Data_Entry_2017_v2020UPDATED.csv') as csvfile1:
        csvReader = csv.reader(csvfile1, delimiter=',')
        next(csvReader)
        for row in csvReader:
            if (row[0] in trainingList):
                image_path = os.path.join(imageLocation, row[0])
                trainingImgages.append(image_path)
                trainigLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
                trainigLabelsString.append(row[1])
            elif (row[0] in testList):
                # testList.append(row[0])
                image_path = os.path.join(imageLocation, row[0])
                testListImages.append(image_path)
                testLabels.append(util.convertfromNameOfDiseazeToOneHot(row[1]))
                testLabelsString.append(row[1])

    print(f"There are {len(trainingList) + len(testList)} images images in total loaded .")
    print(f"There are {len(trainigLabels) + len(testLabels)} labels loaded.")

    if trainingMode == 0:

        #Train Dataset
        trainDataset = tf.data.Dataset.from_tensor_slices((trainingImgages, trainigLabels))
        trainDataset = trainDataset.map(util.convertFromPathAndLabelToTensor, num_parallel_calls=tf.data.AUTOTUNE)
        trainDataset = trainDataset.shuffle(5000, reshuffle_each_iteration=True)
        trainDataset = trainDataset.repeat()  # Mandatory for Keras for now
        trainDataset = trainDataset.batch(BATCH_SIZE,
                                          drop_remainder=True)  # drop_remainder is important on TPU, batch size must be fixed
        trainDataset = trainDataset.prefetch(
            tf.data.AUTOTUNE)

        #Test Dataset -- never repeat and shufle or else imposible to get as stand alone Tensor
        testDataset = tf.data.Dataset.from_tensor_slices((testListImages, testLabels))
        testDataset = testDataset.map(util.convertFromPathAndLabelToTensor, num_parallel_calls=tf.data.AUTOTUNE)
        # testDataset = testDataset.shuffle(5000, reshuffle_each_iteration=True)
        # testDataset = testDataset.repeat()  # Mandatory for Keras for now
        testDataset = testDataset.batch(BATCH_SIZE,
                                        drop_remainder=True)  # drop_remainder is important on TPU, batch size must be fixed
        testDataset = testDataset.prefetch(
            tf.data.AUTOTUNE)

    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    # old Method used to use ImageDataGenerator + FlowFrom Dataframte
    """
    if trainingMode == 1:
        trainigLabels = tf.convert_to_tensor(np.array(trainigLabels)).numpy()
        train_dict = {"pic": trainingImgages}
        traindf = pd.DataFrame(data=train_dict)
        trainDfOneHot = pd.DataFrame(trainigLabels, columns=CLASS_NAMES)
        traindf = traindf.join(trainDfOneHot)

        testLabels = tf.convert_to_tensor(np.array(testLabels)).numpy()
        test_dict = {'pic': testListImages}
        testdf = pd.DataFrame(data=test_dict)
        testDfOneHot = pd.DataFrame(testLabels, columns=CLASS_NAMES)
        testdf = testdf.join(testDfOneHot)

        datagen = ImageDataGenerator(preprocessing_function=util.preProcessImage,
                                     horizontal_flip=False)
        train_generator = datagen.flow_from_dataframe(
            dataframe=traindf,
            x_col="pic",
            y_col=CLASS_NAMES,
            color_mode='rgb',
            subset="training",
            batch_size=BATCH_SIZE,
            seed=42,
            shuffle=False,
            class_mode="raw",
            interpolation='bilinear',
            target_size=(1024,1024))

        datagen1 = ImageDataGenerator(preprocessing_function=util.preProcessImage)
        validation_generator = datagen1.flow_from_dataframe(
            dataframe=testdf,
            x_col="pic",
            y_col=CLASS_NAMES,
            batch_size=BATCH_SIZE,
            seed=42,
            shuffle=True,
            class_mode="raw",
            target_size=(512, 512))
    """
    #  Old method where all the data was loaded into Memory -- good for advice as how not to do it ,
    #  you can not use this approach with large Data as you will overflow the memory , the Dataset and ImageDatagerator objects load dinamically batches of data so they dont flood the memory
    """
    if trainingMode == 3:
        for image in trainingList:
            image_path = os.path.join(imageLocation, image)
            train_images_np.append(util.load_image_into_numpy_array(image_path))
        for image in testList:
            image_path = os.path.join(imageLocation, image)
            test_images_np.append(util.load_image_into_numpy_array(image_path))
        train_images_np = tf.convert_to_tensor(train_images_np)
        test_images_np = tf.convert_to_tensor(test_images_np)
        trainigLabels = tf.convert_to_tensor(np.array(trainigLabels))
        testLabels = tf.convert_to_tensor(np.array(testLabels))
        trainingDataset = util.getDataset(train_images_np, trainigLabels, trainingBbox)
        testDataset = util.getDataset(test_images_np, testLabels, testingBbox)
    """

    model = define_compile_model2()
    model.summary()
    model.load_weights(filepath="checkpoint/model.13-0.8617.h5")
    EPOCHS = 20

    steps_per_epoch = math.floor(len(trainigLabels) / BATCH_SIZE)
    validation_steps = math.floor(len(testLabels) / BATCH_SIZE)

    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                          patience=2, verbose=1, min_lr=0.00001)
    checkpoint_filepath = './checkpoint/'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoint/model.{epoch:02d}-{val_auc:.4f}.h5',
        save_weights_only=True,
        monitor='val_auc',
        mode='max',
        verbose=1,
        save_best_only=False)

    if trainModel:
        # history = model.fit(trainingDataset,
        #                     steps_per_epoch=steps_per_epoch, validation_data=testDataset,
        #                     validation_steps=validation_steps, epochs=EPOCHS,
        #                     callbacks=[reduce_lr_plateau, model_checkpoint_callback])

        # history = model.fit_generator(
        #     generator=train_generator, steps_per_epoch=steps_per_epoch, validation_data=validation_generator,
        #     validation_steps=validation_steps, epochs=EPOCHS,
        #     callbacks=[reduce_lr_plateau, model_checkpoint_callback]
        history = model.fit(
            trainDataset, steps_per_epoch=steps_per_epoch, validation_data=testDataset,
            validation_steps=validation_steps, epochs=3,
            callbacks=[model_checkpoint_callback]
        )
        model.save("CheXnetXt_replica_gabi")
    else:

        eval = model.predict(testDataset, steps=validation_steps)
        results = model.evaluate(testDataset, steps=validation_steps, verbose=2)
        for name, value in zip(model.metrics_names, results):
            print(name, ': ', value)
        # images, labels = tuple(zip(*testDataset))
        # labels = np.array(labels)
        # y = np.concatenate([y for x, y in testDataset], axis=0)
        testLabels = testLabels[:validation_steps * BATCH_SIZE]
        testLabels = tf.convert_to_tensor(testLabels).numpy()
        util.plot_cm(testLabels, eval)
        util.multiclass_roc_auc_score(testLabels, eval)

        idx = 98
        image_path = testListImages[idx]
        actual_label = testLabels[idx]

        vizualizeCam(actual_label, model, image_path)


def vizualizeCam(actual_label, model, image_path):
    """ add heat maps and viz util
    To generate the CAMs, images were fed into the fully trained network
    and the feature maps from the final convolutional layer were extracted
    A map of the most salient features used in classifying the image
    as having a specified pathology was computed by taking the weighted sum
    of the feature maps using their associated weights in the fully connected layer
    select all the layers for which you want to visualize the outputs and store it in a list
        outputLastConv = model.get_layer('bn').output
        vis_model = Model(model.input, outputLastConv)
    """
    actualImageforDisplayNotNormalized = util.load_image_into_numpy_arrayNoNormalized(image_path).numpy()
    sample_image = util.load_image_into_numpy_array(image_path).numpy()
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    pred_label = model.predict(sample_image_processed)[0]
    heatmap = get_CAM(model, sample_image_processed, actual_label, layer_name='bn')
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap * 255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    converted_img = sample_image
    super_imposed_image = cv2.addWeighted(converted_img, 0.8, heatmap.astype('float32'), 2e-3, 0.0)
    sample_activation = get_CAM_simple(model, sample_image_processed)
    f, ax = plt.subplots(2, 2, figsize=(40, 20))

    CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    data = {'lables': CLASS_NAMES,
            'Actual_Labeles': actual_label,
            'Predicted_label:': pred_label
            }
    printDf = pd.DataFrame(data=data)

    # f.legend(printDf.to_markdown())
    ax[0, 0].imshow(actualImageforDisplayNotNormalized)
    ax[0, 0].set_title("Original image ")
    ax[0, 0].axis('off')
    # ax[0, 0].set_title(f"True label: {actual_label} \n Predicted label: {pred_label}")
    ax[0, 1].imshow(sample_activation)
    ax[0, 1].set_title("Class activation map")
    ax[0, 1].axis('off')
    ax[1, 0].imshow(heatmap)
    ax[1, 0].set_title("Heat Map")
    ax[1, 0].axis('off')
    ax[1, 1].imshow(super_imposed_image)
    ax[1, 1].set_title("Activation map superimposed")
    ax[1, 1].axis('off')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.suptitle(printDf.to_markdown())
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

        # loss = multi_category_focal_loss2(gamma=2., alpha=.25)(expected_output, predictions)
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)(expected_output, predictions)

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
    predicted = np.argmax(predictions)
    # We get the weights for diseaze x then dot with the features the final conv layer extracted(batch normalization of different conv layers here)
    #Optional make a vector of predicted diseazes then get the weiths for them all . do the dot for each and add
    class_activation_weights = gap_weights[:, predicted]
    cam_output = np.dot(class_activation_features, class_activation_weights)
    # cam_output = cam_output.mean(axis=-1)
    # cam_output -= cam_output.mean(axis=-1)
    # cam_output /= cam_output.std()
    # cam_output *= 255
    # cam_output = np.clip(cam_output, 0, 255).astype(np.uint8)

    return cam_output


def feature_extractor(inputs):
    feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))(inputs)

    return feature_extractor


def classifier(inputs):
    # add flatten larye because  8, 16, 16, 14 shape of final model before
    averagePool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(14, activation='sigmoid', name="classification")(averagePool)
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
    output = tf.keras.layers.Dense(14, activation='sigmoid', name="classification")(averagePool)

    tf.keras.losses.BinaryFocalCrossentropy
    # loss = tfr.keras.losses.SigmoidCrossEntropyLoss()
    # classification_output = final_model(inputs)
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='Binary Accuracy'),
        tf.keras.metrics.CategoricalAccuracy(name='Categorical Accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    model = tf.keras.Model(inputs=feature_extractor.input, outputs=output)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  # loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)],
                  loss='binary_crossentropy',

                  metrics=METRICS)

    return model


def define_compile_model2():
    inputs = tf.keras.layers.Input(shape=(512, 512, 3))
    x = tf.keras.layers.RandomFlip("horizontal", seed=62)(inputs)
    x = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3), input_tensor=x)
    feature_extractor = x.output
    # feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))(x)

    averagePool = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    output = tf.keras.layers.Dense(14, activation='sigmoid', name="classification")(averagePool)

    # binary_crossentropy rupe
    # loss = tfr.keras.losses.SigmoidCrossEntropyLoss()
    # classification_output = final_model(inputs)
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='Binary Accuracy'),
        tf.keras.metrics.CategoricalAccuracy(name='Categorical Accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=5)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  # loss='binary_focal_crossentropy',
                  loss='binary_crossentropy',
                  # loss=[multi_category_focal_loss2(alpha=0.25, gamma=5)],
                  # loss = loss,
                  metrics=METRICS)

    return model


def downloadAndPrepareWorkspace():
    # Define the training and validation base directories
    train_dir = './dataset/training'
    source_path = './'
    source_list_names = os.listdir(source_path)
    temp_path = os.path.join(source_path, 'temp')
    images_path = os.path.join(source_path, 'images')
    metadata_path = os.path.join(source_path, 'metadata')
    checkpoint_path = os.path.join(source_path, 'checkpoint')
    checkpointExists = os.path.exists(checkpoint_path)
    tempAlreadyExists = os.path.exists(temp_path)
    metaDataAlreadyExists = os.path.exists(metadata_path)
    ImagesDataAlreadyExists = os.path.exists(images_path)
    if not tempAlreadyExists:
        os.makedirs(temp_path)
    if not metaDataAlreadyExists:
        os.makedirs(metadata_path)
    if not checkpointExists:
        os.makedirs(checkpoint_path)
    if not ImagesDataAlreadyExists:
        os.makedirs(images_path)
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

    driveLink = "https://drive.google.com/uc?export=download&id=1uA1DQP9Vj832djEZAoNU5YLm00Qf_rlQ"
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
