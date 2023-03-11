import os
import shutil
import tarfile
import urllib
import zipfile

import keras.applications.densenet
import tensorflow as tf
import numpy as np
import pandas as pd
import wget
from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

# Detect hardware
try:
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_address)  # TPU detection
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    # Going back and forth between TPU and host is expensive.
    # Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
except:
    print('TPU failed to initialize.')
    strategy = tf.distribute.MirroredStrategy()

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

BATCH_SIZE = 8
Global_BATCH_SIZE = strategy.num_replicas_in_sync * BATCH_SIZE
AUTO = tf.data.experimental.AUTOTUNE


class MyModel(tf.keras.Model):
    def __init__(self, classes):
        super(MyModel, self).__init__()

        self._feature_extractor = keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False,
                                                                          input_shape=(512, 512, 3))
        self._averagePool = tf.keras.layers.GlobalAveragePooling2D()
        self._classifier = tf.keras.layers.Dense(classes, activation='sigmoid', name="classification")

    def call(self, inputs):
        x = self._feature_extractor(inputs)
        x = self._averagePool(x)
        x = self._classifier(x)
        return x


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


def readPixel(imagepath):
    with open(imagepath, "rb") as local_file:
        img = local_file.read()

    return img


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
    trainDataset = strategy.experimental_distribute_dataset(trainDataset)

    # Test
    testDataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    testDataset = testDataset.map(loadImageAndPreprocess, num_parallel_calls=AUTO)
    # drop_remainder is important on TPU, batch size must be fixed
    testDataset = testDataset.batch(BATCH_SIZE, drop_remainder=True)
    testDataset = testDataset.prefetch(AUTO)
    testDataset = strategy.experimental_distribute_dataset(testDataset)
    Y_train = Y_train.to_numpy()

    totalsPerDisease = np.sum(Y_train, axis=0)
    total = np.sum(totalsPerDisease)
    totalsPerDisease1 = totalsPerDisease / total

    y_ints = [y.argmax() for y in Y_train]
    # class_weights= np.zeros(shape = (8,14))
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y_ints),
                                                      y=y_ints)

    # class_weights[0,: ] = a
    with strategy.scope():
        model = MyModel(classes=len(CLASS_NAMES))
        model.build(input_shape = (None,512,512,3))
        model.summary()
        loss_object = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            # we use class weights since our data is skewed
            # per_example_loss = loss_object(labels, predictions)
            # we use class weights since our data is skewed
            # adds weight only to the true label
            weights = tf.cast(tf.compat.v2.reduce_sum(class_weights * labels, axis=1), tf.float32)
            # compute your (unweighted) loss
            per_example_loss = loss_object(labels, predictions)
            # apply the weights, relying on broadcasting of the multiplication
            weighted_losses = per_example_loss * weights
            # reduce the result to get your final loss
            loss = tf.reduce_mean(weighted_losses)

            return tf.nn.compute_average_loss(per_example_loss,
                                              global_batch_size=BATCH_SIZE * strategy.num_replicas_in_sync)

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='Binary Accuracy'),
            tf.keras.metrics.CategoricalAccuracy(name='Categorical Accuracy'),
        ]
        optimizer = tf.keras.optimizers.Adam()

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            print(per_replica_losses)
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            strategy.run(test_step, args=(dataset_inputs,))

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images)
                loss = compute_loss(labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            for metric in METRICS:
                metric.update_state(labels, predictions)

            return loss

        def test_step(inputs):
            images, labels = inputs

            predictions = model(images)
            loss = loss_object(labels, predictions)

            test_loss.update_state(loss)
            for metric in METRICS:
                metric.update_state(labels, predictions)

    EPOCHS = 40
    with strategy.scope():
        for epoch in range(EPOCHS):
            # TRAINING LOOP
            total_loss = 0.0
            num_batches = 0
            for x in trainDataset:
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            # TESTING LOOP
            for x in testDataset():
                distributed_test_step(x)

            template = "Epoch {}, Loss: {:.2f},   Test Loss: {:.2f}, "
            print(template.format(epoch + 1, train_loss,
                                  test_loss.result() / strategy.num_replicas_in_sync
                                  ))
            for metric in METRICS:
                print(metric.name + "  " + metric.result().numpy())

            test_loss.reset_states()
            for metric in METRICS:
                metric.reset_state()
    return y


if __name__ == '__main__':
    main()
