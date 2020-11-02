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
import sys
import getopt
import numpy as np
import os
import glob

from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ['KERAS_BACKEND'] = 'tensorflow'


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

        cropped = img_cp[ymin:ymax, xmin:xmax]

        return cropped
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

MODEL_MINE = '/home/users/ramosrafh/git/YoloKerasFaceDetection/pretrain/gender_imdb_RMSprop.hdf5'

MODEL_ROOT_PATH = "./pretrain/"
model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')

if(MODELS == "efficientnet"):
    import keras
    keras_model = load_model(MODEL_MINE)
else:
    print("Network Error!")
keras_model.summary()


#IMAGES_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/backup_predict/topred/'
IMAGES_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/generator/'

# IMAGE_SIZE = 299
# BATCH_SIZE = 64

data_path = os.path.join(IMAGES_FOLDER, '*jpg')
files = glob.glob(data_path)

# for image in files:

#     print(image, "nome da imagem")

#     image_split = image.split('/')[-1]

#     img = cv2.imread(image)
#     img = img[..., ::-1]  # BGR 2 RGB
#     inputs = img.copy() / 255.0

#     img_camera = cv2.resize(inputs, (416, 416))
#     img_camera = np.expand_dims(img_camera, axis=0)
#     out2 = model_face.predict(img_camera)[0]
#     results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])

#     # if not results:
#     #     os.symlink(image, none_folder+image_split)
#     #     continue

#     img = showResults(img, results, img.shape[1], img.shape[0])

#     shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()
#     try:
#         img = cv2.resize(img, (shape[1], shape[2]))
#         cv2.imwrite(
#             '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/test/'+image_split, img)
#     except Exception as e:
#         print(str(e))

#     data = np.array(img, dtype=np.float32)
#     # data.shape = (1,) + data.shape

#     data = data / 255.0

#     pred = keras_model.predict(data)[0]
#     prob = np.max(pred)
#     cls = pred.argmax()


classes = ['m', 'f']

y_true = []
y_pred = []

for classe in classes:

    data_path = os.path.join(IMAGES_FOLDER+classe, '*jpg')
    files = glob.glob(data_path)
    #files = ["/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/backup_predict/topred/m/ccp_0021.jpg"]
    print(files)

    for image in files:
        print("For da imagem")
        image_name = image.split('/')[-1]

        if image is None:
            print("Could not read input image")
            exit()

        img = cv2.imread(image)

        img = img[..., ::-1]  # BGR 2 RGB
        inputs = img.copy() / 255.0

        img_camera = cv2.resize(inputs, (416, 416))
        img_camera = np.expand_dims(img_camera, axis=0)
        out2 = model_face.predict(img_camera)[0]
        results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])

        if not results:
            continue

        img = showResults(img, results, img.shape[1], img.shape[0])
        img = img[..., ::-1]  # BGR 2 RGB

        # print(img)
        # input()

        cv2.imwrite(
            '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/testando/opaaaaa'+image_name, img)

        shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()
        try:
            img = cv2.resize(
                src=img, dsize=(shape[1], shape[2]), interpolation=0)
            cv2.imwrite(
                '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/testando/colooor'+image_name, img)
        except Exception as e:
            print(str(e))

        imagem = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/generator/f/fccp_0019.jpg'

        data = cv2.imread(imagem)

        data = np.array(img, dtype=np.float32)
        #data.shape = (1,) + data.shape
        data = np.expand_dims(data, axis=0)
        # print(data)
        # input()
        data = data / 255.0
        print(data)
        input()
        # apply gender detection on face

        conf = keras_model.predict(data)[0]
        prob = np.max(conf)

        # get label with max accuracy
        idx = np.argmax(conf)

        y_true.append(classe)
        y_pred.append(classes[idx])

        # if classe != classes[idx] and classe == 'man':
        # 		cv2.imwrite(
        # 				'/home/users/ramosrafh/git/gender-detection-keras/predict/wrongman/'+image_name, img)

        # if classe != classes[idx] and classe == 'woman':
        # 		cv2.imwrite(
        # 				'/home/users/ramosrafh/git/gender-detection-keras/predict/wrongwoman/'+image_name, img)

        print(image_name, prob, classe, classes[idx])

        # print(len(y_true), len(y_pred))
        # input()

# Confusion Matrix

matrix = confusion_matrix(y_true, y_pred)
print(matrix)
