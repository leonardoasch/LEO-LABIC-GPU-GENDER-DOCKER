import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import keras.backend as K
import cv2
import sys
import numpy as np
import os
import glob
import efficientnet.keras as efn

from tqdm import tqdm
from align import AlignDlib
from model import create_model

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


MODEL_GENDER = 'model/gender_adience.hdf5'
MODEL_AGE = 'model/age_adience.hdf5'
MODEL_ROOT_PATH = "model/"

IMAGE_SIZE = 299
BATCH_SIZE = 64

BB_FOLDER = 'save/bb/'
FOUND_FOLDER = 'save/found/'
COMPARE_FOLDER = 'save/compare/'

keras_model = load_model(MODEL_GENDER, compile=False)
model_age = load_model(MODEL_AGE, compile=False)
model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5', compile=False)


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
    obj_threshold = 0.78
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


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('model/nn4.small2.v1.h5')

alignment = AlignDlib('model/landmarks.dat')


def saveimg(path, image_name, img):
    cv2.imwrite(path+image_name, img)

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def vetorize(x_test, filenames, faces):
    copy_test = list()
    copy_filenames = list()
    copy_faces = list()
    x_vet = list()

    for img, img_name, face in zip(x_test, filenames, faces): 
        try:
            image = img
            img = align_image(img)
            img = (img / 255.).astype(np.float32)
            img = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
            x_vet.append(img)
            copy_test.append(image)
            copy_filenames.append(img_name)
            copy_faces.append(face)
        except Exception as e:
            continue

    if len(x_vet) < 3:
        copy_test = list()
        copy_filenames = list()
        copy_faces = list()
        x_vet = list()
        for img, img_name, face in zip(x_test, filenames, faces): 
            try:
                img = cv2.resize(img, (96, 96))
                img = (img / 255.).astype(np.float32)
                img = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
                x_vet.append(img)
                copy_test.append(image)
                copy_filenames.append(img_name)
                copy_faces.append(face)
            except Exception as e:
                continue
    
    return x_vet, np.array(copy_test), copy_filenames, copy_faces


def alreadyDetected(new, detected, actual, found_imgs, actual_face, detected_faces):

    treshold = 0.85
    metric_treshold = 45

    for i, det in enumerate(detected):
        dist = distance(new, det)
        # print(dist, "A distancia aqui!!")
        if dist > treshold:
            # saveimg(COMPARE_FOLDER, str(dist)+'mesmavet.jpg', np.copy(found_imgs[i][..., ::-1]))
            # saveimg(COMPARE_FOLDER, str(dist)+'mesmanative.jpg', np.copy(actual[..., ::-1]))
            detected[:][i] = (detected[i] + new) / 2
            detected_faces[:][i] = actual_face
            found_imgs[:][i] = actual
            return True
        else:
            if (((abs(actual_face[0] - detected_faces[i][0]) < metric_treshold) and \
                (abs(actual_face[2] - detected_faces[i][2]) < metric_treshold)) or \
                ((abs(actual_face[1] - detected_faces[i][1]) < metric_treshold) and \
                (abs(actual_face[3] - detected_faces[i][3]) < metric_treshold))):
                # saveimg(COMPARE_FOLDER, str(dist)+'diffvet.jpg', np.copy(found_imgs[i][..., ::-1]))
                # saveimg(COMPARE_FOLDER, str(dist)+'diffnative.jpg', np.copy(actual[..., ::-1]))
                detected[:][i] = (detected[i] + new) / 2
                detected_faces[:][i] = actual_face
                found_imgs[:][i] = actual
                return True

    return False


def findAll(find, total, gender, age, find_face, faces):

    treshold = 0.75
    metric_treshold = 55
    
    person_gender = []
    person_age = []

    for i, actual in enumerate(total):
        dist = distance(find, actual)
        if dist > treshold:
            if (((abs(find_face[0] - faces[i][0]) < metric_treshold) and \
                (abs(find_face[2] - faces[i][2]) < metric_treshold)) or \
                ((abs(find_face[1] - faces[i][1]) < metric_treshold) and \
                (abs(find_face[3] - faces[i][3]) < metric_treshold))):
                total[:][i] = (total[i] + find) / 2
                person_gender.append(gender[i])
                person_age.append(age[i])

    return person_gender, person_age

def distance(emb1, emb2):
    return np.dot(emb1, emb2) / (np.sqrt(np.dot(emb1, emb1)) * np.sqrt(np.dot(emb2, emb2))) # cossine
    # return np.sum(np.square(emb1 - emb2))                               # euclidian


def maximum(li):
    counts = dict()

    for item in li:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1

    word = max(counts, key=counts.get)

    return word

def verifyPeople(filenames, imgs, vets, faces, gender, age):

    people_gender = []
    people_age = []

    person_gender = []
    person_age = []

    found_vets = []
    found_faces = []
    found_imgs = []

    if not vets:
        return None, None
        
    found_vets.append(vets[0])
    found_faces.append(faces[0])
    found_imgs.append(imgs[0])

    for i in range(len(vets)):

        if not alreadyDetected(vets[i], found_vets, imgs[i], found_imgs, faces[i], found_faces):
            found_vets.append(vets[i])
            found_faces.append(faces[i])
            found_imgs.append(imgs[i])


    for i, f in enumerate(found_vets):
        person_gender, person_age = findAll(f, vets, gender, age, found_faces[i], faces)


        if len(person_gender) and len(person_age):
            # print(person_gender)
            people_gender.append(maximum(person_gender))
            people_age.append(maximum(person_age))

    # for i, im in enumerate(found_imgs):
    #     saveimg(FOUND_FOLDER, str(i)+'.jpg', im)


    return people_gender, people_age

def loadData(directory):

    faces = []
    filenames = []
    x_test = []
    x_vet = []
    # dash = 10
    dash = 0

    data_path = os.path.join(directory, '*jpg')
    all_files = glob.glob(data_path)

    frames_all = len(all_files)

    frames_number = 7

    frames_delta = int(frames_all/frames_number)

    files = []

    j = 0
    for i in range(frames_number):
        files.append(all_files[i+j])
        j = frames_delta

    files.sort()

    for image in files:
    # for image in tqdm(files, desc="Files"):

        if image is None:
            print("Could not read input image")
            exit()

        image_name = image.split('/')[-1]

        image = cv2.imread(image)
        img = np.copy(image[..., ::-1])  # BGR 2 RGB

        inputs = img.copy() / 255.0

        img_camera = cv2.resize(inputs, (416, 416))
        img_camera = np.expand_dims(img_camera, axis=0)
        out2 = model_face.predict(img_camera)[0]
        results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])

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

            startX, endX, startY, endY = crop(x, y, w, h, 1.0, img.shape[1], img.shape[0])
            vstartX, vendX, vstartY, vendY = (startX + dash), (endX + dash), (startY + dash), (endY + dash)

            face_bb = [startX, startY, endX, endY]
            #print(face_bb)
            #input()

            face_crop = np.copy(img[startY:endY, startX:endX])
            face_vet = np.copy(img[vstartY:vendY, vstartX:vendX])
            face_vet = np.copy(face_vet[..., ::-1])
            cv2.rectangle(image, (startX, startY),
                        (endX, endY), (0, 255, 0), 2)
            try:
                face_crop = cv2.resize(face_crop, (299, 299))
                x_test.append(face_crop)
                x_vet.append(face_vet)
                faces.append(face_bb)
                filenames.append(image_name)
            except Exception as e:
                continue

            # for im in x_vet:
            #     saveimg(COMPARE_FOLDER, 'vet'+image_name, im)

        # cv2.imwrite(
        #    '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/cropped/'+image_name, image)

    # print(faces)
    # input()
    return filenames, faces, x_test, x_vet

def detect_people(FRAMES):

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    filenames, faces, x_test, x_vet = loadData(FRAMES)

    x_vet, x_test, filenames, faces = vetorize(x_test, filenames, faces)

    if not len(x_test):
        print('No person found!')
        return None

    validation_gender = test_datagen.flow(x_test,
                                            batch_size=1,
                                            shuffle=False
                                            )


    validation_data_n = len(x_test)

    pred_gender_full = keras_model.predict_generator(
        validation_gender, len(validation_gender), verbose=0)

    idx_gender = np.argmax(pred_gender_full, axis=1)

    prob_gender_full = list()

    for i in pred_gender_full:
        prob_gender_full.append(np.max(i))






    validation_age = test_datagen.flow(x_test,
                                            batch_size=1,
                                            shuffle=False
                                            )


    validation_data_n = len(x_test)

    pred_age_full = model_age.predict_generator(
        validation_age, len(validation_age), verbose=0)

    idx_age = np.argmax(pred_age_full, axis=1)

    prob_age_full = list()

    for i in pred_age_full:
        prob_age_full.append(np.max(i))

    pred_gender = list()
    pred_age = list()
    after_names = list()
    after_faces = list()
    probability_gen = list()
    probability_age = list()
    after_imgs = list()
    vets = list()

    # print(filenames)
    # input()

    # print(len(filenames), len(x_test), len(x_vet), len(faces), len(idx_gender), len(prob_gender_full), len(idx_age), len(prob_age_full))
    # input('aaa')

    for img_name, img, vet, face, gender, p, age, p_age in zip(filenames, x_test, x_vet, faces, idx_gender, prob_gender_full, idx_age, prob_age_full):
        if gender == 0 and age == 0: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('adult')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 0 and age == 1: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('baby')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 0 and age == 2: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('child')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 0 and age == 3: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('old')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 1 and age == 0: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('adult')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 1 and age == 1: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('baby')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 1 and age == 2: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('child')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)
        elif gender == 1 and age == 3: # and p >= 0.48 and p_age >= 0.48:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('old')
            probability_age.append(p_age)
            after_imgs.append(img)
            vets.append(vet)

    people_gender, people_age = verifyPeople(after_names, after_imgs, vets, after_faces, pred_gender, pred_age)

    if people_gender is None:
        print("No person found!")
        return None


    final_people = []

    for pg, pa in zip(people_gender, people_age):
        final_person = []

        final_person.append(pg)
        final_person.append(pa)

        final_people.append(final_person)

    results = zip(after_names, pred_gender, probability_gen, pred_age, probability_age)

    # for i in results:
    #     print(i)

    return final_people
