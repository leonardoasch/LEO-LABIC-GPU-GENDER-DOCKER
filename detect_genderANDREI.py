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

os.environ['KERAS_BACKEND'] = 'tensorflow'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MODEL_MINE = '../rafh/model/gender_imdb_RMSprop.hdf5'
MODEL_ROOT_PATH = "../rafh/model/"


IMAGE_SIZE = 299
BATCH_SIZE = 64

MALE_FOLDER = '../rafh/save/male/'
FEMALE_FOLDER = '../rafh/save/female/'
NONE_FOLDER = '../rafh/save/none/'
FOUND_FOLDER = '../rafh/save/found/'


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

pessoas = []
def verificaPessoa(f):
    global pessoas
    threshold = 30

    if not len(pessoas):
        pessoas.append(f[:])
        return len(pessoas)

    for i in range(len(pessoas)):
        if not abs(f[0] - pessoas[i][0]) < threshold and abs(f[1] - pessoas[i][1]) < threshold and abs(f[2] - pessoas[i][2]) < threshold and abs(f[3] - pessoas[i][3]) < threshold:
            pessoas.append(f[:])

    return len(pessoas)

def loadData(directory):
    global pessoas

    filenames = []
    x_test = []
    conf = []
    pessoas = []

    data_path = os.path.join(directory, '*jpg')
    all_files = glob.glob(data_path)

    frames_all = len(all_files)

    frames_number = 30

    frames_delta = int(frames_all/frames_number)

    files = []

    j = 0
    for i in range(frames_number):
        files.append(all_files[i+j])
        j = frames_delta
    n_pessoas = 0 
    for image in tqdm(files, desc="Files"):

        if image is None:
            print("Could not read input image")
            exit()

        image_name = image.split('/')[-1]
        
        img = cv2.imread(image)

        face, confidence = cv.detect_face(img)

        if not face:
            #os.symlink(image, NONE_FOLDER+image_name)
            continue
        else:
            for idx, f in enumerate(face):

                n_pessoas = verificaPessoa(f)

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

    return np.array(x_test), filenames, n_pessoas

def detect_gender(FRAMES):

    import keras
    keras_model = load_model(MODEL_MINE)
    model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    x_test, image_names, n_pessoas = loadData(FRAMES)
    print(x_test)
    validation_generator = test_datagen.flow(x_test,
                                            batch_size=1,
                                            shuffle=False#,
                                            #save_to_dir='/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/2/'
                                            )


    validation_data_n = len(image_names)

    Y_pred = keras_model.predict_generator(
        validation_generator, len(validation_generator), verbose=1)

    idx = np.argmax(Y_pred, axis=1)

    prob = list()

    for i in Y_pred:
        prob.append(np.max(i))


    classes = ['m', 'f']
    pred_cls = list()

    for img_name, i in zip(image_names, idx):
        if i == 0:
            pred_cls.append('f')
            #try:
            #    os.symlink(FRAMES+img_name, FEMALE_FOLDER+img_name)
            #    os.symlink(FRAMES+img_name, FOUND_FOLDER+img_name)
            #except FileExistsError:
            #    print("Arquivo já salvo!")
        elif i == 1:
            pred_cls.append('m')
            #try:
            #    os.symlink(FRAMES+img_name, MALE_FOLDER+img_name)
            #    os.symlink(FRAMES+img_name, FOUND_FOLDER+img_name)
            #except FileExistsError:
            #    print("Arquivo já salvo!")
        else:
            print("EITA QUE NÃO ERA PARA ROLAR", i)


    filenames_to_cls = list(zip(image_names, pred_cls, prob))

    for i in filenames_to_cls:
        print(i)

    print('No video tem ', n_pessoas, ' pessoas.')

    y_pred = np.argmax(Y_pred, axis=1)
    prob = np.max(Y_pred)



detect_gender("/home/users/andrei/Video_To_Text_New/YouTube2Text/frames/vid157")
