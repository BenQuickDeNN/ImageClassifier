# provide IO API to other files

import pandas as pd
from PIL import Image
import HOGExtractor as He
import ColorFeatureExtractor as cfe
import os
from tqdm import *


def read_label_data(label_path):
    if not os.path.exists(label_path):
        print('label file does not exist: ' + label_path)
        return -1
    label_data = pd.read_csv(label_path, header=None)
    if label_data is None:
        print('label is empty')
        return -1
    return label_data.get_values()


def read_hog_data(hog_feature_path):
    if not os.path.exists(hog_feature_path):
        print('HOG file does not exist!')
        return -1
    print('reading ' + hog_feature_path)
    hog_data = pd.read_csv(hog_feature_path, header=None)
    print('reading completed')
    if hog_data is None:
        print('HOG feature table is empty')
        return -1
    return hog_data.get_values()


def read_color_data(color_feature_path):
    if not os.path.exists(color_feature_path):
        print('HOG file does not exist!')
        return -1
    print('reading ' + color_feature_path)
    hog_data = pd.read_csv(color_feature_path, header=None)
    print('reading completed')
    if hog_data is None:
        print('HOG feature table is empty')
        return -1
    return hog_data.get_values()


def generate_hog_csv(destination_path, image_directory):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    file_list = os.listdir(image_directory)
    csv_writer = open(destination_path, 'a')
    file_list_len = len(file_list)
    for i in tqdm(range(0, file_list_len), desc='generating ' + destination_path):
        path = os.path.join(image_directory, file_list[i])
        image = Image.open(path)
        csv_writer.write(file_list[i] + ',')
        try:
            fd = He.extract_hog_feature(image)
        except Exception as e:
            tqdm.write(str(e) + "image: " + file_list[i])
            continue
        fd_len = len(fd)  # prefetch to register
        # save HOG feature to CSV file
        for j in range(0, fd_len):
            if j == fd_len - 1:
                csv_writer.write(str(fd[j]) + '\n')
            else:
                csv_writer.write(str(fd[j]) + ",")
    csv_writer.close()
    return


def generate_color_csv(destination_path, image_directory):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    file_list = os.listdir(image_directory)
    csv_writer = open(destination_path, 'a')
    file_list_len = len(file_list)
    for i in tqdm(range(0, file_list_len), desc='generating ' + destination_path):
        path = os.path.join(image_directory, file_list[i])
        image = Image.open(path)
        csv_writer.write(file_list[i] + ',')
        try:
            fd = cfe.extract_color_feature(image)
        except Exception as e:
            tqdm.write(str(e) + "image: " + file_list[i])
            continue
        fd_len = len(fd)  # prefetch to register
        # save HOG feature to CSV file
        for j in range(0, fd_len):
            if j == fd_len - 1:
                csv_writer.write(str(fd[j]) + '\n')
            else:
                csv_writer.write(str(fd[j]) + ",")
    csv_writer.close()
    return


def generate_prediction(predict_data, destination_path):
    if os.path.exists(destination_path):
        os.remove(destination_path)
    csv_writer = open(destination_path, 'w')
    for i in tqdm(range(0, predict_data.shape[0]), 'generating prediction'):
        for j in range(0, predict_data.shape[1]):
            if j == predict_data.shape[1] - 1:
                csv_writer.write(str(predict_data[i][j]) + '\n')
            else:
                csv_writer.write(str(predict_data[i][j]) + ",")
    csv_writer.close()
    return


# test region
He.extract_hog_feature(Image.open('../test/images/MWI_kbVt6Sc8y1Kqb8wO.jpg'))
