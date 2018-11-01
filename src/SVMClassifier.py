# classify HOG feature by using support vector machine
# author: Ben Quick
# email: benquickdenn@foxmail.com
# date: 2018-10-31

from tqdm import *
import IOMethod as iom
import numpy as np
import os
from sklearn import svm
from sklearn.externals import joblib

# terrain enumerate
# KEY_TERRAIN = ['DESSERT', 'OCEAN', 'MOUNTAIN', 'FARMLAND', 'LAKE', 'CITY']

dir_hog_feature_2000 = "../training-2000/HOG_feature/HOG_feature.csv"  # path of HOG feature table
dir_color_feature_2000 = "../training-2000/color_feature/color_feature.csv"

dir_label_2000 = "../training-2000/label/training-2000-label.csv"

dir_hog_feature_test = "../test/HOG_feature/HOG_feature.csv"
dir_color_feature_test = "../test/color_feature/color_feature.csv"

dir_prediction_test = "../test/prediction/prediction.csv"
dir_color_prediction_test = "../test/prediction/color_prediction.csv"

dir_predict_model = "../training-2000/training_module/hog_based_svm"
dir_predict_color_model = "../training-2000/training_module/color_based_svm"

SVM_KERNEL = 'rbf'
SVM_GAMMA = 0.5
SVM_DECISION_FUNCTION_SHAPE = 'ovo'


def training_svm_module_by_hog( hog_feature_path, label_path, training_module_output_path):
    hog_data = iom.read_hog_data(hog_feature_path)
    hog_data_len_w = int(hog_data.shape[1])
    label_data = iom.read_label_data(label_path)
    label_data_len_h = label_data.shape[0]
    label_data_reshaped = []
    for i in tqdm(range(0, label_data_len_h), desc='SVM - reshaping label data'):
        if label_data[i][0] in hog_data[:, 0]:
            label_data_reshaped.append(label_data[i])
    label_data_reshaped = np.array(label_data_reshaped)
    classifier = svm.SVC(kernel=SVM_KERNEL, gamma=SVM_GAMMA, decision_function_shape=SVM_DECISION_FUNCTION_SHAPE)
    print('SVM - fitting')
    classifier.fit(hog_data[:, 1:hog_data_len_w - 1], label_data_reshaped[:, 1])
    if os.path.exists(training_module_output_path):
        os.remove(training_module_output_path)
    open(training_module_output_path, 'w')
    joblib.dump(classifier, training_module_output_path)
    print('model dumped to ' + training_module_output_path)
    return


def training_svm_module_by_color(color_feature_path, label_path, training_module_output_path):
    color_data = iom.read_color_data(color_feature_path)
    color_data_len_w = int(color_data.shape[1])
    label_data = iom.read_label_data(label_path)
    label_data_len_h = label_data.shape[0]
    label_data_reshaped = []
    for i in tqdm(range(0, label_data_len_h), desc='SVM - reshaping label data'):
        if label_data[i][0] in color_data[:, 0]:
            label_data_reshaped.append(label_data[i])
    label_data_reshaped = np.array(label_data_reshaped)
    classifier = svm.SVC(kernel=SVM_KERNEL, gamma=SVM_GAMMA, decision_function_shape=SVM_DECISION_FUNCTION_SHAPE)
    print('SVM - fitting')
    classifier.fit(color_data[:, 1:color_data_len_w - 1], label_data_reshaped[:, 1])
    if os.path.exists(training_module_output_path):
        os.remove(training_module_output_path)
    open(training_module_output_path, 'w')
    joblib.dump(classifier, training_module_output_path)
    print('model dumped to ' + training_module_output_path)
    return


def svm_predict(model_path, feature_data_path, output_file_path):
    test_hog_data = iom.read_hog_data(feature_data_path)
    classifier = joblib.load(model_path)
    print('SVM - predicting')
    result = np.array(classifier.predict(test_hog_data[:, 1:test_hog_data.shape[1] - 1]))
    predict_result = []
    for i in tqdm(range(0, len(result)), 'reshaping predicting result'):
        predict_result.append([test_hog_data[i, 0], result[i]])
    predict_result = np.array(predict_result)
    print(predict_result)
    iom.generate_prediction(predict_result, output_file_path)
    return predict_result


# test region
read_gamma = 0.5
training_svm_module_by_hog(dir_hog_feature_2000, dir_label_2000, dir_predict_model + '_gamma-' + str(SVM_GAMMA))
svm_predict(dir_predict_model + '_gamma-' + str(read_gamma), dir_hog_feature_test, dir_prediction_test)
# training_svm_module_by_color(dir_color_feature_2000, dir_label_2000, dir_predict_color_model + '_gamma-' + str(SVM_GAMMA))
# svm_predict(dir_predict_color_model + '_gamma-' + str(read_gamma), dir_color_feature_test, dir_color_prediction_test)

