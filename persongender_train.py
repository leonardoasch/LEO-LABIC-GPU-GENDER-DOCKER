# ----------------------------------------------
# Train age gender classifier
# ----------------------------------------------

from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Input
import keras.backend as backend
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.callbacks
from keras import layers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.layers import SeparableConv2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Conv2D

import efficientnet.keras as efn

import os.path
import sys
import numpy as np

import cv2
import os
import glob

os.environ['KERAS_BACKEND'] = 'tensorflow'


# ----------------------------------------------
# Settings
# ----------------------------------------------

ANNOTATIONS = ''
MODELS = ''
DATASET_NAME = ''
DATASET_ROOT_PATH = './'  # /Volumes/ST5/keras/
EXTRA_MODE = ''
OPTIMIZER = ''

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) >= 4:
    ANNOTATIONS = sys.argv[1]
    MODELS = sys.argv[2]
    DATASET_NAME = sys.argv[3]
    if len(sys.argv) >= 5:
        OPTIMIZER = sys.argv[4]
    if len(sys.argv) >= 6:
        DATASET_ROOT_PATH = sys.argv[5]
    if len(sys.argv) >= 7:
        EXTRA_MODE = sys.argv[6]
else:
    print(
        "usage: python agegender_train.py [gender/age/age101] [inceptionv3/vgg16/squeezenet/squeezenet2/mobilenet] [adience/imdb/utk/appareal/vggface2/merged] [datasetroot(optional)] [augumented/hdf5(optional)]")
    sys.exit(1)

if ANNOTATIONS != "gender" and ANNOTATIONS != "age" and ANNOTATIONS != "age101":
    print("unknown annotation mode")
    sys.exit(1)

if MODELS != "inceptionv3" and MODELS != "efficientnet" and MODELS != "vgg16" and MODELS != "squeezenet" and MODELS != "squeezenet2" and MODELS != "mobilenet":
    print("unknown network mode")
    sys.exit(1)

if DATASET_NAME != "adience" and DATASET_NAME != "imdb" and DATASET_NAME != "utk" and DATASET_NAME != "appareal" and DATASET_NAME != "vggface2" and DATASET_NAME != "merged":
    print("unknown dataset name")
    sys.exit(1)

if EXTRA_MODE != "" and EXTRA_MODE != "augumented" and EXTRA_MODE != "hdf5":
    print("unknown extra mode")
    sys.exit(1)

# ----------------------------------------------
# Model
# ----------------------------------------------

if EXTRA_MODE != "augumented":
    DATA_AUGUMENTATION = False
else:
    DATA_AUGUMENTATION = True

BATCH_SIZE = 8

if ANNOTATIONS == 'age101':
    EPOCS = 100
else:
    EPOCS = 500

EXTRA_PREFIX = ""
if EXTRA_MODE != "":
    EXTRA_PREFIX = "_"+EXTRA_MODE

PLOT_FILE = DATASET_ROOT_PATH+'pretrain/agegender_' + \
    ANNOTATIONS+'_'+MODELS+'_'+DATASET_NAME+EXTRA_PREFIX+'.png'
MODEL_HDF5 = DATASET_ROOT_PATH+'pretrain/agegender_' + \
    ANNOTATIONS+'_'+MODELS+'_'+DATASET_NAME+EXTRA_PREFIX+'.hdf5'

if ANNOTATIONS == 'age':
    N_CATEGORIES = 8
if ANNOTATIONS == 'gender':
    N_CATEGORIES = 2
if ANNOTATIONS == 'age101':
    N_CATEGORIES = 101

# ----------------------------------------------
# Limit GPU memory usage
# ----------------------------------------------


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
backend.set_session(sess)

# ----------------------------------------------
# Build Model
# ----------------------------------------------

if(MODELS == 'inceptionv3'):
    IMAGE_SIZE = 299
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = InceptionV3(
        weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(N_CATEGORIES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    layer_num = len(model.layers)
    print(layer_num)
    # for layer in model.layers[:15]:
    #     layer.trainable = False
    # for layer in model.layers[15:]:
    #     layer.trainable = True
elif(MODELS == 'vgg16'):
    IMAGE_SIZE = 224
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(N_CATEGORIES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers[:15]:
        layer.trainable = False
elif(MODELS == 'squeezenet'):
    IMAGE_SIZE = 227
    import sys
    sys.path.append('../keras-squeezenet-master')
    from keras_squeezenet import SqueezeNet
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = SqueezeNet(
        weights="imagenet", include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(N_CATEGORIES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
elif(MODELS == 'squeezenet2'):
    IMAGE_SIZE = 64
    import sys
    sys.path.append('../keras-squeezenet-master')
    from keras_squeezenet import SqueezeNet
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = SqueezeNet(include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = Dropout(0.5, name='drop9')(x)
    x = Convolution2D(N_CATEGORIES, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax', name='loss')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
elif(MODELS == 'mobilenet'):
    IMAGE_SIZE = 128
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    base_model = MobileNet(
        weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(N_CATEGORIES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
elif(MODELS == 'efficientnet'):
    IMAGE_SIZE = 299
    input_shape = shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    base_model = efn.EfficientNetB4(
        weights='imagenet', include_top=False, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(N_CATEGORIES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
else:
    raise Exception('invalid model name')

if(MODELS == 'inceptionv3' or MODELS == 'efficientnet' or MODELS == 'vgg16' or MODELS == 'squeezenet' or MODELS == 'squeezenet2' or MODELS == 'mobilenet'):
    # for fine tuning
    if OPTIMIZER == 'RMSprop':
        from keras.optimizers import RMSprop
        model.compile(optimizer=RMSprop(lr=0.001, rho=0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    elif OPTIMIZER == 'SGD':
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    elif OPTIMIZER == 'Adam':
        from keras.optimizers import Adam
        model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                     amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
else:
    # for full training
    from keras import Adagrad
    model.compile(optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# ----------------------------------------------
# Data Augumentation
# ----------------------------------------------

# reference from https://github.com/yu4u/age-gender-estimation/blob/master/random_eraser.py
# https://github.com/yu4u/age-gender-estimation/blob/master/LICENSE


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img
    return eraser

# ----------------------------------------------
# Load data
# ----------------------------------------------


def loadData(directory):

    subdirs = glob.glob(os.path.join(directory, "*", ""))
    trainX = []
    trainY = []

    for dr in tqdm(subdirs, desc="Folders"):
        data_path = os.path.join(dr, '*jpg')
        files = glob.glob(data_path)

        for f in tqdm(files, desc="Files"):
            img = cv2.imread(f)

            img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inputs = img.copy() / 255.0
            img_camera = cv2.resize(inputs, (227, 227))
            # img_camera = cv2.resize(inputs, (299, 299))

            label = dr.split('/')[-2]
            # print("File: ", f)
            trainX.append(img_camera)
            trainY.append(label)

    return trainX, trainY


# ----------------------------------------------
# Data
# ----------------------------------------------


if EXTRA_MODE != "hdf5":
    preprocessing_function = None
    if DATA_AUGUMENTATION:
        preprocessing_function = get_random_eraser(v_l=0, v_h=255)

    # preprocessing_function = preprocess_input

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=10
        # preprocessing_function=preprocessing_function
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_ROOT_PATH+'dataset/agegender_' +
        DATASET_NAME+'/annotations/'+ANNOTATIONS+'/train',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        DATASET_ROOT_PATH+'dataset/agegender_'+DATASET_NAME +
        '/annotations/'+ANNOTATIONS+'/validation',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    training_data_n = len(train_generator.filenames)
    validation_data_n = len(validation_generator.filenames)

    print("Training data count : "+str(training_data_n))
    print("Validation data count : "+str(validation_data_n))

    if DATASET_NAME != "imdb" and DATASET_NAME != "merged":
        training_data_n = training_data_n*4  # Data augumentation
        print("Training data augumented count : "+str(training_data_n))

# ----------------------------------------------
# Train
# ----------------------------------------------

el = EarlyStopping(monitor='val_loss', patience=5,
                   verbose=1, mode='min')
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2,
                        verbose=1, mode='min')

filepath = 'results/'+MODELS+'_'+ANNOTATIONS + \
    '_'+DATASET_NAME+'_'+OPTIMIZER+'.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

cb_list = [el, rlr, checkpoint]

if EXTRA_MODE == "hdf5":
    from keras.utils.io_utils import HDF5Matrix
    HDF5_PATH = DATASET_ROOT_PATH+"dataset/"+DATASET_NAME+"_"+ANNOTATIONS+".h5"
    x_train = HDF5Matrix(HDF5_PATH, 'training_x')
    y_train = HDF5Matrix(HDF5_PATH, 'training_y')
    x_validation = HDF5Matrix(HDF5_PATH, 'validation_x')
    y_validation = HDF5Matrix(HDF5_PATH, 'validation_y')
    fit = model.fit(
        epochs=EPOCS,
        x=x_train,
        y=y_train,
        validation_data=(x_validation, y_validation),
        batch_size=BATCH_SIZE,
        shuffle='batch'
    )
else:
    array_path = DATASET_ROOT_PATH+'dataset/agegender_' + \
        DATASET_NAME+'/annotations/'
    train_path = DATASET_ROOT_PATH+'dataset/agegender_' + \
        DATASET_NAME+'/annotations/'+ANNOTATIONS+'/train'

    validation_path = DATASET_ROOT_PATH+'dataset/agegender_' + \
        DATASET_NAME+'/annotations/'+ANNOTATIONS+'/validation'
    if os.path.exists(array_path+'trainXinception.npy'):
        trainX = np.load(array_path+'trainXinception.npy')
        trainY = np.load(array_path+'trainYinception.npy')
        validX = np.load(array_path+'validXinception.npy')
        validY = np.load(array_path+'validYinception.npy')
        label_encoder = LabelEncoder()
        trainY = label_encoder.fit_transform(trainY)
        validY = label_encoder.fit_transform(validY)
        trainY = to_categorical(trainY)
        validY = to_categorical(validY)
    else:
        # trainX, trainY = loadData(train_path)
        # validX, validY = loadData(validation_path)

        # np.save(array_path+'trainX.npy', trainX)
        # np.save(array_path+'trainY.npy', trainY)
        # np.save(array_path+'validX.npy', validX)
        # np.save(array_path+'validY.npy', validY)

        # # trainX = np.expand_dims(trainX, axis=-1)
        # # validX = np.expand_dims(validX, axis=-1)

        # print(trainX.shape, trainY.shape)

        # fit = model.fit(
        #     epochs=EPOCS,
        #     x=trainX,
        #     y=trainY,
        #     validation_data=(validX, validY),
        #     batch_size=BATCH_SIZE,
        #     shuffle=True,
        #     callbacks=[TQDMNotebookCallback()]
        # )

        fit = model.fit_generator(train_generator,
                                  epochs=EPOCS,
                                  verbose=1,
                                  workers=12,
                                  validation_data=validation_generator,
                                  steps_per_epoch=training_data_n//BATCH_SIZE,
                                  validation_steps=validation_data_n//BATCH_SIZE,
                                  callbacks=cb_list
                                  )


# else:
#   fit = model.fit_generator(train_generator,
#     epochs=EPOCS,
#     verbose=1,
#     workers=12,
#     validation_data=validation_generator,
#     steps_per_epoch=training_data_n//BATCH_SIZE,
#     validation_steps=validation_data_n//BATCH_SIZE
#   )

train_loss = fit.history['loss']
test_loss = fit.history['val_loss']

epoch_count = range(1, len(train_loss) + 1)

plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plotname = 'results/'+MODELS+'_'+ANNOTATIONS+'_'+DATASET_NAME+'_'+OPTIMIZER
plt.savefig(plotname+'.png', dpi=300)

model.save(MODEL_HDF5)

# ----------------------------------------------
# Plot
# ----------------------------------------------

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

# loss


def plot_history_loss(fit):
        # Plot the loss in the history
    axL.plot(fit.history['loss'], label="loss for training")
    axL.plot(fit.history['val_loss'], label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc


def plot_history_acc(fit):
    # Plot the loss in the history
    axR.plot(fit.history['acc'], label="accuracy for training")
    axR.plot(fit.history['val_acc'], label="accuracy for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')


plot_history_loss(fit)
plot_history_acc(fit)
fig.savefig(PLOT_FILE)
plt.close()
