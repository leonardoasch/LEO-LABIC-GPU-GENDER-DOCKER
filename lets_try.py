import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


MODEL_MINE = '/home/users/ramosrafh/git/YoloKerasFaceDetection/pretrain/gender_imdb_RMSprop.hdf5'
MODEL_AGE = '/home/users/ramosrafh/git/YoloKerasFaceDetection/pretrain/age_adience.hdf5'
MODEL_ROOT_PATH = "./pretrain/"


IMAGE_SIZE = 299
BATCH_SIZE = 64

MALE_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/male/'
FEMALE_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/female/'
NONE_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/none/'
FOUND_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/found/'
COMPARE_FOLDER = '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/compare/'


def saveimg(path, image_name, img):
    cv2.imwrite(path+image_name, img)

def maximum(li):
    counts = dict()

    for item in li:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1

    word = max(counts, key=counts.get)

    return word

def verifyPeople(filenames, imgs, faces, gender, age):

    people = []
    people_data = []
    people_gender = []
    people_age = []

    verif = []

    threshold = 20

    if not len(people_data):
        people_data.append(faces[0][:])
        people.append(filenames[0][:])
        saveimg(COMPARE_FOLDER, filenames[0], imgs[0])

    for i in range(len(faces)):
        for j in range(len(people_data)):
            if (((abs(faces[i][0] - people_data[j][0]) > threshold) and \
                    (abs(faces[i][2] - people_data[j][2]) > threshold)) and \
                    ((abs(faces[i][1] - people_data[j][1]) > threshold) and \
                    (abs(faces[i][3] - people_data[j][3]) > threshold))):
                verif.append('1')
            else:
                people_data[j] = faces[i]
                break


        if len(verif) == len(people_data):
            if faces[i] not in people_data:
                people_data.append(faces[i][:])
                people.append(filenames[i][:])
                saveimg(COMPARE_FOLDER, str(i)+filenames[i], imgs[i])


    for i in range(len(people_data)):
        person_gender = []
        person_age = []

        threshold = 60

        for j in range(len(faces)):
            if not (((abs(faces[j][0] - people_data[i][0]) > threshold) and \
                    (abs(faces[j][2] - people_data[i][2]) > threshold)) or \
                    ((abs(faces[j][1] - people_data[i][1]) > threshold) and \
                    (abs(faces[j][3] - people_data[i][3]) > threshold))):
                person_gender.append(gender[j])
                person_age.append(age[j])

        if len(person_gender) and len(person_age):
            people_gender.append(maximum(person_gender))
            people_age.append(maximum(person_age))

    return people, people_data, people_gender, people_age

def loadData(directory):

    faces = []
    filenames = []
    x_test = []
    conf = []

    data_path = os.path.join(directory, '*jpg')
    all_files = glob.glob(data_path)

    frames_all = len(all_files)

    frames_number = 50

    frames_delta = int(frames_all/frames_number)

    files = []

    j = 0
    for i in range(frames_number):
        files.append(all_files[i+j])
        j = frames_delta

    files.sort()

    for image in tqdm(files, desc="Files"):

        if image is None:
            print("Could not read input image")
            exit()

        image_name = image.split('/')[-1]

        img = cv2.imread(image)

        face, confidence = cv.detect_face(img)

        if not face:
            os.symlink(image, NONE_FOLDER+image_name)
            continue
        else:
            for idx, f in enumerate(face):



                #verifyPerson(f, image_name)

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
                    faces.append(f[:])
                    filenames.append(image_name)
                    conf.append(confidence)
                except Exception as e:
                    print(str(e))
                    break


            cv2.imwrite(
                '/home/users/ramosrafh/git/YoloKerasFaceDetection/dataset/video/cropped/'+image_name, img)

    return filenames, faces, np.array(x_test)

def detect_gender(FRAMES):

    # global people
    # global people_data

    import keras
    keras_model = load_model(MODEL_MINE, compile=False)
    model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5', compile=False)
    model_age = load_model(MODEL_AGE, compile=False)

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    filenames, faces, x_test = loadData(FRAMES)

    if not len(x_test):
        print('No person found!')
        return None

    validation_gender = test_datagen.flow(x_test,
                                            batch_size=1,
                                            shuffle=False
                                            )


    validation_data_n = len(x_test)

    pred_gender_full = keras_model.predict_generator(
        validation_gender, len(validation_gender), verbose=1)

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
        validation_age, len(validation_age), verbose=1)

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

    for img_name, img, face, gender, p, age, p_age in zip(filenames, x_test, faces, idx_gender, prob_gender_full, idx_age, prob_age_full):
        if gender == 0 and age == 0: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('adult')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 0 and age == 1: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('baby')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 0 and age == 2: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('child')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 0 and age == 3: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('f')
            probability_gen.append(p)
            pred_age.append('old')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 1 and age == 0: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('adult')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 1 and age == 1: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('baby')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 1 and age == 2: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('child')
            probability_age.append(p_age)
            after_imgs.append(img)
        elif gender == 1 and age == 3: # and p >= 0.55 and p_age >= 0.55:
            after_names.append(img_name)
            after_faces.append(face)
            pred_gender.append('m')
            probability_gen.append(p)
            pred_age.append('old')
            probability_age.append(p_age)
            after_imgs.append(img)

    # global people
    # global people_data

    people, people_data, people_gender, people_age = verifyPeople(after_names, after_imgs, after_faces, pred_gender, pred_age)


    final_people = []

    for pg, pa in zip(people_gender, people_age):
        final_person = []

        final_person.append(pg)
        final_person.append(pa)

        final_people.append(final_person)

    results = zip(after_names, pred_gender, probability_gen, pred_age, probability_age)

    for i in results:
        print(i)

    return final_people
