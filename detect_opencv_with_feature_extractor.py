# ----------------------------------------------
# Predict age gender classifier
# ----------------------------------------------

import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers
from keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D, Input
from keras.preprocessing import image
import efficientnet.keras as efn
from keras.applications.vgg16 import VGG16
import keras.backend as K
import cv2
import cvlib as cv
import sys
import getopt
import numpy as np
import os
import glob

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tqdm import tqdm
import shutil

os.environ['KERAS_BACKEND'] = 'tensorflow'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def swish(x):
    return K.sigmoid(x) * x


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


class bounding_box:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

# crop


def crop(x, y, w, h, margin, img_width, img_height):
    xmin = int(x-w*margin)
    xmax = int(x+w*margin)
    ymin = int(y-h*margin)
    ymax = int(y+h*margin)
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > img_width:
        xmax = img_width
    if ymax > img_height:
        ymax = img_height
    return xmin, xmax, ymin, ymax


def showResults(img, results, img_width, img_height):
    img_cp = img.copy()
    cropp = list()

    for i in range(len(results)):
        # display detected face
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3])//2
        h = int(results[i][4])//2

        if(w < h):
            w = h
        else:
            h = w

        xmin, xmax, ymin, ymax = crop(x, y, w, h, 1.0, img_width, img_height)

        cropp.append([xmin, xmax, ymin, ymax])
        # cropp.append(xmax)
        # cropp.append(ymin)
        # cropp.append(ymax)

        cropped = img_cp[ymin:ymax, xmin:xmax]

        return cropped, cropp
        # cv2.imwrite(
        #     '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/test/'+img_name, cropped)


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap(
        [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap(
        [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union


# YOLOV2
# reference from https://github.com/experiencor/keras-yolo2
# https://github.com/experiencor/keras-yolo2/blob/master/LICENSE
def interpret_output_yolov2(output, img_width, img_height):
    anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
               5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    netout = output
    nb_class = 1
    obj_threshold = 0.4
    nms_threshold = 0.3

    grid_h, grid_w, nb_box = netout.shape[:3]

    size = 4 + nb_class + 1
    nb_box = 5

    netout = netout.reshape(grid_h, grid_w, nb_box, size)

    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * \
        _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    # center position, unit: image width
                    x = (col + _sigmoid(x)) / grid_w
                    # center position, unit: image height
                    y = (row + _sigmoid(y)) / grid_h
                    w = anchors[2 * b + 0] * \
                        np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * \
                        np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = bounding_box(x-w/2, y-h/2, x+w/2,
                                       y+h/2, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(
            reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    result = []
    for i in range(len(boxes)):
        if(boxes[i].classes[0] == 0):
            continue
        predicted_class = "face"
        score = boxes[i].score
        result.append([predicted_class, (boxes[i].xmax+boxes[i].xmin)*img_width/2, (boxes[i].ymax+boxes[i].ymin)
                       * img_height/2, (boxes[i].xmax-boxes[i].xmin)*img_width, (boxes[i].ymax-boxes[i].ymin)*img_height, score])

    return result


def loadData(directory, video_name):

    filenames = []
    x_test = []
    conf = []

    data_path = os.path.join(directory, '*jpg')
    all_files = glob.glob(data_path)

    frames_all = len(all_files)

    frames_number = 30

    frames_delta = 0#int(frames_all/frames_number)

    files = []

    j = 0
    for i in range(frames_number):
        files.append(all_files[i+j])
        j = frames_delta

    for image in tqdm(files, desc="Files"):

        if image is None:
            print("Could not read input image")
            exit()

        image_name = image.split('/')[-1]

        img = cv2.imread(image)

        face, confidence = cv.detect_face(img)

        if not face:
            os.symlink(image, NONE_FOLDER+video_name+"_"+image_name)
            continue
        else:
            for idx, f in enumerate(face):

                # get corner points of face rectangle
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face

                face_crop = np.copy(img[startY:endY, startX:endX])
                cv2.rectangle(img, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)
                try:
                    face_crop = cv2.resize(face_crop, (299, 299))
                    face_crop = face_crop[..., ::-1]  # BGR 2 RGB
                    x_test.append(face_crop)
                    filenames.append(image_name)
                    conf.append(confidence)
                except Exception as e:
                    print(str(e))
                    break

            cv2.imwrite(
                '/home/ssd/rafh/save/cropped/'+image_name, img)

        # img = cv2.resize(img, (299, 299))

        # cv2.imwrite(
        #     '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/cropped/'+image_name, img)

        # filenames.append(image_name)
        # x_test.append(img)

    return np.array(x_test), filenames


# def loadData(directory):

#     filenames = []
#     x_test = []

#     data_path = os.path.join(directory, '*jpg')
#     files = glob.glob(data_path)

#     for image in tqdm(files, desc="Files"):

#         if image is None:
#             print("Could not read input image")
#             exit()

#         image_name = image.split('/')[-1]

#         img = load_img(image)
#         img = img_to_array(img)

#         filenames.append(image_name)
#         x_test.append(img)

#     return filenames, np.array(x_test)

# ----------------------------------------------
# MODE
# ----------------------------------------------


ANNOTATIONS = ''
DATASET_NAME = ''
MODELS = ""
DATASET_ROOT_PATH = ""
OPTIONAL_MODE = ""
DATA_AUGUMENTATION = False

# DATASET_ROOT_PATH="/Volumes/TB4/Keras/"

# ----------------------------------------------
# Argument
# ----------------------------------------------

if len(sys.argv) >= 3:
    ANNOTATIONS = sys.argv[1]
    MODELS = sys.argv[2]
    if len(sys.argv) >= 4:
        DATASET_NAME = sys.argv[3]
    if len(sys.argv) >= 5:
        DATASET_ROOT_PATH = sys.argv[4]
    if len(sys.argv) >= 6:
        OPTIONAL_MODE = sys.argv[5]
else:
    print(
        "usage: python agegender_predict.py [gender/age/age101/emotion] [inceptionv3/vgg16/squeezenet/octavio] [adience/imdb/utk/appareal/vggface2/empty] [datasetroot(optional)] [benchmark/caffemodel(optional)]")
    sys.exit(1)

if ANNOTATIONS != "gender" and ANNOTATIONS != "age" and ANNOTATIONS != "age101" and ANNOTATIONS != "emotion":
    print("unknown annotation mode")
    sys.exit(1)

if MODELS != "efficientnet" and "inceptionv3" and MODELS != "vgg16" and MODELS != "squeezenet" and MODELS != "mobilenet" and MODELS != "octavio":
    print("unknown network mode")
    sys.exit(1)

if DATASET_NAME != "adience" and DATASET_NAME != "imdb" and DATASET_NAME != "utk" and DATASET_NAME != "appareal" and DATASET_NAME != "vggface2" and DATASET_NAME != "empty":
    print("unknown dataset name")
    sys.exit(1)

if OPTIONAL_MODE != "" and OPTIONAL_MODE != "benchmark" and OPTIONAL_MODE != "caffemodel":
    print("unknown optional mode")
    sys.exit(1)

if DATASET_NAME == "empty":
    DATASET_NAME = ""
else:
    DATASET_NAME = '_'+DATASET_NAME

# ----------------------------------------------
# converting
# ----------------------------------------------

AUGUMENT = ""
if(DATA_AUGUMENTATION):
    AUGUMENT = "augumented"

MODEL_MINE = '/home/ssd/rafh/model/gender_imdb_RMSprop.hdf5'
MODEL_ROOT_PATH = "/home/ssd/rafh/model/"

MODEL_HDF5 = DATASET_ROOT_PATH+'pretrain/agegender_' + \
    ANNOTATIONS+'_'+MODELS+DATASET_NAME+AUGUMENT+'.hdf5'
ANNOTATION_WORDS = 'words/agegender_'+ANNOTATIONS+'_words.txt'

if(ANNOTATIONS == "emotion"):
    ANNOTATION_WORDS = 'words/emotion_words.txt'

if(MODELS == "octavio"):
    if(ANNOTATIONS == "emotion"):
        MODEL_HDF5 = DATASET_ROOT_PATH+'pretrain/fer2013_mini_XCEPTION.102-0.66.hdf5'
    if(ANNOTATIONS == "gender"):
        MODEL_HDF5 = DATASET_ROOT_PATH+'pretrain/gender_mini_XCEPTION.21-0.95.hdf5'

if(MODELS == "mobilenet"):
    import keras
    from keras.utils.generic_utils import CustomObjectScope
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        keras_model = load_model(MODEL_HDF5)
if(MODELS == "efficientnet"):
    import keras
    keras_model = load_model(MODEL_MINE)
    model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')
else:
    print("Network Error!")
keras_model.summary()



# IMAGES_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/agegender_utk/annotations/gender/validation/'
# IMAGES_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/generator/test/'

# OKAY ONES
# video_name = 'vid1015'
# video_name = 'vid449'
#video_name = 'vid515'

# BAD ONES
# video_name = 'vid755'
# video_name = 'vid644'
video_name = 'vid554'
video_list = ['vid1015', 'vid449', 'vid515', 'vid755', 'vid644', 'vid554']


MALE_FOLDER = 'save/male/'
FEMALE_FOLDER = 'save/female/'
NONE_FOLDER = 'save/none/'
FOUND_FOLDER = 'save/found/'

shutil.rmtree(MALE_FOLDER)
shutil.rmtree(FEMALE_FOLDER)
shutil.rmtree(NONE_FOLDER)
shutil.rmtree(FOUND_FOLDER)

if os.path.exists(MALE_FOLDER) == False:
    os.makedirs(MALE_FOLDER)
if os.path.exists(FEMALE_FOLDER) == False:
    os.makedirs(FEMALE_FOLDER)
if os.path.exists(NONE_FOLDER) == False:
    os.makedirs(NONE_FOLDER)
if os.path.exists(FOUND_FOLDER) == False:
    os.makedirs(FOUND_FOLDER)



from keras.models import Model
layer_name = "global_average_pooling2d_1"
intermediate_model = Model(inputs=keras_model.input, outputs=keras_model.get_layer(layer_name).get_output_at(-1)) 


test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

IMAGE_SIZE = 299
BATCH_SIZE = 64
resultado = {}
video_list = ['vid1867', 'vid1966', 'vid1924', 'vid1926', 'vid1968', 'vid1969']
for video_name in video_list:
    FRAMES = '/home/users/datasets/youtubeclips-datasetV2/frames/' + video_name + '/'
    print("LENDO VIDEOS DE ", FRAMES)



    x_test, image_names = loadData(FRAMES, video_name)
    if x_test.shape[0] == 0:
        continue

    validation_generator = test_datagen.flow(x_test,
                                             batch_size=1,
                                             shuffle=False
                                             # save_to_dir='/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/2/'
                                             )

    print(len(validation_generator))

    # validation_data_n = len(validation_generator.filenames)
    validation_data_n = len(image_names)

    #EXTRACAO DE FEATURES
    Y_pred = intermediate_model.predict_generator(
        # validation_generator, validation_data_n, verbose=1)
        validation_generator, len(validation_generator), verbose=1)
    np.save("faces_"+video_name+".npy", Y_pred)
    with open("faces_"+video_name+".txt", 'w') as f:
        for item in image_names:
            f.write("%s\n" % item)
        f.close()


    print(Y_pred.shape)

    #PREDICAO
    validation_generator = test_datagen.flow(x_test,
                                             batch_size=1,
                                             shuffle=False
                                             # save_to_dir='/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/2/'
                                             )
    Y_pred = keras_model.predict_generator(
        # validation_generator, validation_data_n, verbose=1)
        validation_generator, len(validation_generator), verbose=1)

    
    idx = np.argmax(Y_pred, axis=1)

    prob = list()

    for i in Y_pred:
        prob.append(np.max(i))


    classes = ['m', 'f']
    pred_cls = list()
    result = {"m": 0, "pm": 0.0, "f":0, "pf": 0.0}
    for img_name, i in zip(image_names, idx):
        if i == 0:
            pred_cls.append('f')
            result["f"] +=1
            result["pf"] +=prob[i]
            try:
                os.symlink(FRAMES+img_name, FEMALE_FOLDER+video_name+"_"+img_name)
                os.symlink(FRAMES+img_name, FOUND_FOLDER+video_name+"_"+img_name)
            except FileExistsError:
                print("Arquivo já salvo!")
        elif i == 1:
            pred_cls.append('m')
            result["m"] +=1
            result["pm"] +=prob[i]
            try:
                os.symlink(FRAMES+img_name, MALE_FOLDER+video_name+"_"+img_name)
                os.symlink(FRAMES+img_name, FOUND_FOLDER+video_name+"_"+img_name)
            except FileExistsError:
                print("Arquivo já salvo!")
        else:
            print("EITA QUE NÃO ERA PARA ROLAR", i)

    print(len(image_names), len(pred_cls), len(prob))

    # pred_cls = classes[idx]
    filenames_to_cls = list(zip(image_names, pred_cls, prob))

    for i in filenames_to_cls:
        print(i)

    y_pred = np.argmax(Y_pred, axis=1)
    prob = np.max(Y_pred)
    print("Video ", video_name)
    if result['pm'] > 0:
        print("Masc", result["pm"] / result["m"])
    if result['pf'] > 0:
        print("FEM", result["pf"] / result["f"])
    print("\n\n")
