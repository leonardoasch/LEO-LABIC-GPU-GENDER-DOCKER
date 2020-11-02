import sys
from tqdm import tqdm
import os, glob

from detect_people import detect_people
import pickle


def countWithoutExceed(line, predict, right, total):

    initial_len = len(line)
    total += initial_len

    if not predict:
        return right, total

    for pred in predict:
        for p in pred:
            if p in line:
                line.remove(p)

    right += (initial_len - len(line))

    return right, total


def countPeople(line, predict, predicted, total, same):

    total += (len(line)/2)

    if not predict:
        return int(predicted), int(total), int(same)

    predicted += len(predict)

    if (len(line)/2) == len(predict):
        same += 1

    return int(predicted), int(total), int(same)

def count(line, predict, right, total, video_name):

    initial_len = len(line)
    total += initial_len

    if not predict:
        return right, total

    test = list()
    test = line.copy()

    for pred in predict:
        aux_pred = pred.copy()
        for p in pred:
            # print(p)
            # print(aux_pred)
            # print(line)
            if p in line:
                aux_pred.remove(p)
                line.remove(p)
                #input()
        exceed = len(aux_pred)
        total += exceed

    right += (initial_len - len(line))

    return right, total

def readcsv(filename):

    lis = list()

    f = open(filename)
    for line in f:
        a = line.split()

        for item in a:
            b = item.split(',')
            lis.append(b)


    return lis


data_path = os.path.join('/home/users/datasets/youtubeclips-datasetV2/frames/', '*/')
paths = sorted(glob.glob(data_path))


results = list()
total = 0
right = 0
predicted = 0
same = 0

for video in tqdm(paths, desc='Files'):

    a = list()


    #video = "/home/users/datasets/youtubeclips-datasetV2/frames/vid1376/"
    video_name = video.split('/')[-2]

    people = detect_people(video)

    a.append(video_name)
    a.append(people)
    print(video, a)
    results.append(a)
with open('msvg_gender.pkl', 'wb') as f:
   pickle.dump(results, f)


# PARA O COUNTPEOPLE()

# print('Total: ', total)
# print('Predicted: ', predicted)
# print('Correct people count in videos: ', same)


# for p in results:
#     print(p)

# PARA O COUNT() E COUNT_WITHOUT_EXCEED


