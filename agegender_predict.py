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
if len(sys.argv) >= 1:
    ANNOTATIONS = sys.argv[1]
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

# if ANNOTATIONS!="gender" and ANNOTATIONS!="age" and ANNOTATIONS!="age101" and ANNOTATIONS!="emotion":
#   print("unknown annotation mode");
#   sys.exit(1)

# if MODELS!="inceptionv3" and MODELS!="vgg16" and MODELS!="squeezenet" and MODELS!="mobilenet" and MODELS!="octavio":
#   print("unknown network mode");
#   sys.exit(1)

# if DATASET_NAME!="adience" and DATASET_NAME!="imdb" and DATASET_NAME!="utk" and DATASET_NAME!="appareal" and DATASET_NAME!="vggface2" and DATASET_NAME!="empty":
#   print("unknown dataset name");
#   sys.exit(1)

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

MODEL_MINE = DATASET_ROOT_PATH+'pretrain/gender_efficientnet_utk.hdf5'

# MODEL_HDF5=DATASET_ROOT_PATH+'pretrain/agegender_'+ANNOTATIONS+'_'+MODELS+DATASET_NAME+AUGUMENT+'.hdf5'
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
else:
    keras_model = load_model(MODEL_MINE)
keras_model.summary()

# ----------------------------------------------
# convert to caffe model
# ----------------------------------------------

if OPTIONAL_MODE == "caffemodel":
    os.environ["GLOG_minloglevel"] = "2"
    import caffe
    import keras2caffe
    prototxt = DATASET_ROOT_PATH+'pretrain/agegender_' + \
        ANNOTATIONS+'_'+MODELS+DATASET_NAME+'.prototxt'
    caffemodel = DATASET_ROOT_PATH+'pretrain/agegender_' + \
        ANNOTATIONS+'_'+MODELS+DATASET_NAME+'.caffemodel'
    keras2caffe.convert(keras_model, prototxt, caffemodel)

# ----------------------------------------------
# Benchmark
# ----------------------------------------------

if OPTIONAL_MODE == "benchmark":
    BENCHMARK_DATASET_NAME = "imdb"
    BENCHMARK_DATASET_TARGET = "validation"
    BATCH_SIZE = 64

    shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()

    disp_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        DATASET_ROOT_PATH+'dataset/agegender_'+BENCHMARK_DATASET_NAME +
        '/annotations/'+ANNOTATIONS+'/'+BENCHMARK_DATASET_TARGET,
        target_size=(shape[1], shape[2]),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # DISTRIBUTION_FILE = DATASET_ROOT_PATH+'pretrain/benchmark_' +
    ANNOTATIONS+"_"+MODELS+DATASET_NAME+'.png'

    fig = plt.figure()
    ax1 = fig.add_axes((0.1, 0.6, 0.8, 0.3))
    ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.3))
    ax1.tick_params(labelbottom="on")
    ax2.tick_params(labelleft="on")

    max_cnt = len(disp_generator.filenames)
    # max_cnt=10

    x = np.zeros((max_cnt))
    y = np.zeros((max_cnt))
    t = np.zeros((max_cnt))

    cnt = 0
    heatmap = np.zeros((len(disp_generator.class_indices),
                        len(disp_generator.class_indices)))
    for x_batch, y_batch in disp_generator:
        for i in range(BATCH_SIZE):
            x[cnt] = y_batch[i][0]
            t[cnt] = y_batch[i].argmax()

            data = x_batch[i]
            data.shape = (1,) + data.shape
            pred = keras_model.predict(data)[0]
            cls = pred.argmax()

            y[cnt] = cls

            heatmap[int(y[cnt]), int(t[cnt])
                    ] = heatmap[int(y[cnt]), int(t[cnt])]+1

            cnt = cnt+1
            print(""+str(cnt)+"/"+str(max_cnt)+" ground truth:" +
                  str(y_batch[i].argmax())+" predicted:"+str(cls))
            if cnt >= max_cnt:
                break
        if cnt >= max_cnt:
            break

    ax1.pcolor(heatmap, cmap=plt.cm.Blues)
    if heatmap.shape[0] <= 2:
        for y in range(heatmap.shape[0]):
            for x in range(heatmap.shape[1]):
                ax1.text(x + 0.5, y + 0.5, '%.4f' % heatmap[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )

    ax1.set_title('ground truth vs predicted '+ANNOTATIONS)
    ax1.set_xlabel(ANNOTATIONS+'(ground truth)')
    ax1.set_ylabel(ANNOTATIONS+'(predicted)')
    ax1.legend(loc='upper right')

    ax2.hist(t, bins=len(disp_generator.class_indices))
    ax2.set_title('distribution of ground truth '+ANNOTATIONS)
    ax2.set_xlabel(ANNOTATIONS+'(ground truth)')
    ax2.set_ylabel('count')
    ax2.legend(loc='upper right')

    fig.savefig(DISTRIBUTION_FILE)
    sys.exit(1)

# ----------------------------------------------
# Normal test
# ----------------------------------------------

if(os.path.exists("./dataset/agegender_adience/")):
    DATASET_PATH_ADIENCE = ""
else:
    DATASET_PATH_ADIENCE = "/Volumes/TB4/Keras/"

if(os.path.exists("./dataset/agegender_imdb/")):
    DATASET_PATH_IMDB = ""
else:
    DATASET_PATH_IMDB = "/Volumes/TB4/Keras/"

# image_list=[
# 	DATASET_PATH_ADIENCE+'dataset/agegender_adience/annotations/agegender/validation/0_0-2_m/landmark_aligned_face.84.8277643357_43f107482d_o.jpg',
# 	DATASET_PATH_ADIENCE+'dataset/agegender_adience/annotations/agegender/validation/11_15-20_f/landmark_aligned_face.290.11594063605_713764ddeb_o.jpg',
# 	DATASET_PATH_ADIENCE+'dataset/agegender_adience/annotations/agegender/validation/3_15-20_m/landmark_aligned_face.291.11593667615_2cb80d1c2a_o.jpg',
# 	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/f/26707.jpg',
# 	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/f/26761.jpg',
# 	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/m/181.jpg',
# 	DATASET_PATH_IMDB+'dataset/agegender_imdb/annotations/gender/train/m/83.jpg'
# ]

image_list = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/images/'
male_folder = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/male/'
female_folder = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/female/'
none_folder = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/none/'

# for aFolder = folder_list:
# 	image_list = os.listdir("djaksldas/dasdas/dasdas/das/"+aFolder)
data_path = os.path.join(image_list, '*jpg')
files = glob.glob(data_path)

# for image in image_list:
# 	if not os.path.exists(image):
# 		print(image+" not found")
# 		continue

MODEL_ROOT_PATH = "./pretrain/"
model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')

# for image in files:
#     print(image, "nome da imagem")
#     image_split = image.split('/')[-1]
#     img = cv2.imread(image)
#     img = img[..., ::-1]  # BGR 2 RGB
#     inputs = img.copy() / 255.0

#     img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img_camera = cv2.resize(inputs, (416, 416))
#     img_camera = np.expand_dims(img_camera, axis=0)
#     out2 = model_face.predict(img_camera)[0]
#     results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
#     showResults(img, image_split, results, img.shape[1], img.shape[0])

# input()

for image in files:

    print(image, "nome da imagem")

    image_split = image.split('/')[-1]

    img = cv2.imread(image)
    img = img[..., ::-1]  # BGR 2 RGB
    inputs = img.copy() / 255.0

    img_camera = cv2.resize(inputs, (416, 416))
    img_camera = np.expand_dims(img_camera, axis=0)
    out2 = model_face.predict(img_camera)[0]
    results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])

    if not results:
        os.symlink(image, none_folder+image_split)
        continue

    img = showResults(img, results, img.shape[1], img.shape[0])

    cv2.imwrite(
        '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/test/opaaaaa'+image_split, img)

    shape = keras_model.layers[0].get_output_at(0).get_shape().as_list()
    try:
        img = cv2.resize(img, (shape[1], shape[2]))
        cv2.imwrite(
            '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/test/test/'+image_split, img)
    except Exception as e:
        print(str(e))

    data = np.array(img, dtype=np.float32)
    data.shape = (1,) + data.shape

    data = data / 255.0

    pred = keras_model.predict(data)[0]
    prob = np.max(pred)
    cls = pred.argmax()
    # y_pred.append(cls)
    # if aFolder == 'm'
    # 	y_true.append(1)
    # else:
    # 	y_true.append(0)

    # MALE
    if cls == 1 and not os.path.exists(male_folder+image_split):
        try:
            os.symlink(image, male_folder+image_split)
        except FileExistsError:
            print("Deu erro!")

    # FEMALE
    elif cls == 0 and not os.path.exists(female_folder+image_split):
        try:
            os.symlink(image, female_folder+image_split)
        except FileExistsError:
            print("Deu erro!")
    lines = open(ANNOTATION_WORDS).readlines()
    print("keras:", prob, cls, lines[cls])

# ----------------------------------------------
# Test caffemodel
# ----------------------------------------------

    if OPTIONAL_MODE == "caffemodel":
        net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        data = data.transpose((0, 3, 1, 2))
        out = net.forward_all(data=data)
        pred = out[net.outputs[0]]
        prob = np.max(pred)
        cls = pred.argmax()
        print("caffe:", prob, cls, lines[cls])
