import math
import os
import shutil
import tarfile
import urllib.request
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import wget
from keras.applications.densenet import DenseNet121
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model

import Utils as util

BATCH_SIZE = 8
AUTO = tf.data.experimental.AUTOTUNE
trainModel = False
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


def loadImageAndPreprocess(imagepath, label):
    img = tf.io.read_file(imagepath)
    # img = tf.py_function(readPixel, [imagepath], [tf.uint8])
    # print(img)
    # img = tf.io.decode_raw(img, tf.string)
    image1 = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(image1, size=(512, 512), preserve_aspect_ratio=True, method='nearest')
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tf.cast(img, tf.float32) / 255.0
    img = img - mean
    image = img / std
    # flip with 50 % probability
    image = tf.image.flip_left_right(image)

    return image, label


def main():
    downloadAndPrepareWorkspace()
    imagePrefix = "./images/"
    df = pd.read_csv('./metadata/Data_Entry_2017_v2020UPDATED.csv', usecols=[0, 1])
    df['Image Index'] = imagePrefix + df['Image Index'].astype(str)
    y_entry = df.pop('Finding Labels')
    y = y_entry.str.get_dummies()
    y.pop('No Finding')
    df = pd.concat([df, y], axis=1)
    columns = df.columns.values
    X_train, X_test, Y_train, Y_test = train_test_split(df[columns[0]], df[columns[1:]], test_size=0.2, shuffle=False)
    # test for 1
    # loadImageAndPreprocess(X_train[0],None)
    # readPixel(X_train[0])
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    # Train
    trainDataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    trainDataset = trainDataset.with_options(option_no_order)
    trainDataset = trainDataset.map(loadImageAndPreprocess, num_parallel_calls=AUTO)
    trainDataset = trainDataset.shuffle(2000)
    # drop_remainder is important on TPU, batch size must be fixed
    trainDataset = trainDataset.batch(BATCH_SIZE, drop_remainder=True)
    trainDataset = trainDataset.prefetch(AUTO)

    # Test
    testDataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    testDataset = testDataset.map(loadImageAndPreprocess, num_parallel_calls=AUTO)
    # drop_remainder is important on TPU, batch size must be fixed
    testDataset = testDataset.batch(BATCH_SIZE, drop_remainder=True)
    testDataset = testDataset.prefetch(AUTO)
    Y_train = Y_train.to_numpy()

    # since we have multi label we sum per axis 0 argmax is incorect
    totalsPerDisease = np.sum(Y_train, axis=0)
    total = np.sum(totalsPerDisease)
    # formula per weight :  n_samples / (n_classes * np.bincount(y))
    weights = total / (len(CLASS_NAMES) * totalsPerDisease)

    class_weights = dict(enumerate(weights))
    with strategy.scope():
        model = define_compile_model()
    model.summary()

    # model.load_weights(
    #     filepath="trainings and results/TrainWithClassWeithsPlusCustomLoss/model.20-0.8029_bestRecall.h5")
    model.load_weights(
        filepath="trainings and results/trainSGDNoClassWeightCustomLoss2/model.05-0.8839.h5")

    EPOCHS = 20

    steps_per_epoch = math.floor(len(X_train) / BATCH_SIZE)
    validation_steps = math.floor(len(X_test) / BATCH_SIZE)

    reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
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
        history = model.fit(
            trainDataset, steps_per_epoch=steps_per_epoch, validation_data=testDataset,
            validation_steps=validation_steps, epochs=3,
            callbacks=[model_checkpoint_callback], class_weight=class_weights
        )
        model.save("CheXnetXt_replica_gabi")
    else:
        evaluate = False
        if evaluate:
            eval = model.predict(testDataset, steps=validation_steps)
            results = model.evaluate(testDataset, steps=validation_steps, verbose=2)
            for name, value in zip(model.metrics_names, results):
                print(name, ': ', value)
            # images, labels = tuple(zip(*testDataset))
            # labels = np.array(labels)
            # y = np.concatenate([y for x, y in testDataset], axis=0)
            testLabels = Y_test
            testLabels = tf.convert_to_tensor(testLabels).numpy()
            util.plot_cm(testLabels, eval)
            # util.multiclass_roc_auc_score(testLabels, eval)
        idx = 304
        df2 = pd.read_csv('./metadata/Old/BBox_List_2017 (1).csv')
        row = df2.iloc[idx]
        actual_label = util.convertfromNameOfDiseazeToOneHot(row[1])
        image_path = "./images/" + row[0]

        vizualizeCam(actual_label, model, image_path, np.array(row[2:6].tolist()))



def BinaryCrossentropy_extra_weigh_Positive_Example(timesBoost):
    timesBoost = tf.cast(timesBoost, tf.float32)

    def compute_loss_extra_weigh_Positive_Example(labels, predictions):
        labels = tf.cast(labels, tf.float32)
        loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        # we use class weights since our data is skewed
        weights = timesBoost * labels
        # compute your (unweighted) loss
        per_example_loss = loss_object(labels, predictions)
        print(per_example_loss)
        print(weights)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = per_example_loss * weights
        # reduce the result to get your final loss
        # loss = tf.reduce_mean(weighted_losses)
        loss = weighted_losses
        return loss

    return compute_loss_extra_weigh_Positive_Example


def vizualizeCam(actual_label, model, image_path, bboxCoordinates):
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
    # we rezise from 1024 to 512 so we divide the bonding box number by 2
    bboxCoordinates = bboxCoordinates / 2
    actualImageforDisplayNotNormalized = util.draw_bounding_box_on_image(actualImageforDisplayNotNormalized,
                                                                         bboxCoordinates[0], bboxCoordinates[1],
                                                                         bboxCoordinates[2], bboxCoordinates[3])

    sample_image = util.load_image_into_numpy_array(image_path).numpy()
    sample_image_processed = np.expand_dims(sample_image, axis=0)
    pred_label = model.predict(sample_image_processed)[0]
    heatmap = get_CAM_with_Saliency(model, sample_image_processed, actual_label, layer_name='bn')
    heatmap = cv2.resize(heatmap, (sample_image.shape[0], sample_image.shape[1]))
    heatmap = heatmap * 255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    converted_img = sample_image
    super_imposed_image = cv2.addWeighted(converted_img, 0.8, heatmap.astype('float32'), 2e-3, 0.0)
    sample_activation = get_CAM_simple(model, sample_image_processed)
    f, ax = plt.subplots(2, 2, figsize=(40, 20))

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
    ax[1, 0].set_title("Heat Map / Saliency ")
    ax[1, 0].axis('off')
    ax[1, 1].imshow(super_imposed_image)
    ax[1, 1].set_title("Activation map superimposed Saliency ")
    ax[1, 1].axis('off')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    f.suptitle(printDf.to_markdown())
    plt.show()


def get_CAM_with_Saliency(model, processed_image, actual_label, layer_name='bn'):
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
        loss = BinaryCrossentropy_extra_weigh_Positive_Example(2)(expected_output, predictions)

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
    # Optional make a vector of predicted diseazes then get the weiths for them all . do the dot for each and add
    class_activation_weights = gap_weights[:, predicted]
    cam_output = np.dot(class_activation_features, class_activation_weights)
    # cam_output = cam_output.mean(axis=-1)
    # cam_output -= cam_output.mean(axis=-1)
    # cam_output /= cam_output.std()
    # cam_output *= 255
    # cam_output = np.clip(cam_output, 0, 255).astype(np.uint8)

    return cam_output


def define_compile_model():
    feature_extractor = DenseNet121(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
    averagePool = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor.output)
    output = tf.keras.layers.Dense(14, activation='sigmoid', name="classification")(averagePool)
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
    optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.BinaryFocalCrossentropy( alpha = 0.75,gamma=2.0, apply_class_balancing=True)
    model.compile(optimizer=optimizer,
                  # loss=[BinaryCrossentropy_extra_weigh_Positive_Example(2)],
                  # loss='binary_crossentropy',
                  loss=loss,
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
