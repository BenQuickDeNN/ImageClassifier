# cal the Euculid distance between a given point and other points, and show the distribution of all samples
# author: Ben Quick
# email: benquickdenn@foxmail.com
# date: 2018-10-30
import numpy as np
from tqdm import *
import operator as op
import matplotlib.pyplot as plt
import IOMethod as iom


def cal_euculid_distance(array1, array2):
    if len(array1) != len(array2):
        print('array length does not match!')
        return -1;
    dist_2 = 0
    array_length = len(array1)  # prefetch to register
    for i in range(0, array_length):
        dist_2 += pow(array1[i] - array2[i], 2)
    return np.sqrt(dist_2)


def get_euculid_distance_array( hog_feature_path, image_name ):
    hog_data = iom.read_hog_data(hog_feature_path)
    hog_data_len_h = hog_data.shape[0]  # prefetch to register
    # binary searching
    up_bound = hog_data_len_h - 1
    lower_bound = 0
    last_up_bound = up_bound
    last_lower_bound = lower_bound
    i = int(hog_data_len_h / 2)
    while up_bound != lower_bound:
        if op.eq(image_name, hog_data[i][0]):
            break  # get the result
        elif op.gt(image_name, hog_data[i][0]):
            lower_bound = i + 1;
        else:
            up_bound = i - 1
        if last_up_bound == up_bound and last_lower_bound == lower_bound:
            break
        i = int((up_bound + lower_bound) / 2)  # i must be an integer
    if not op.eq(image_name, hog_data[i][0]):
        print('cannot find ' + image_name + 'in ' + hog_feature_path)
        return -1
    # generate distance array
    hog_data_len_w = hog_data.shape[1]
    main_array = hog_data[i][1:hog_data_len_w - 1]  # prefetch
    distance_array = [0] * (hog_data_len_h - 1)
    flag_meet_main_array = False
    for j in tqdm(range(0, hog_data_len_h), desc='calculating Euculid distance'):
        if j == i:
            flag_meet_main_array = True
            continue
        if flag_meet_main_array:
            distance_array[j - 1] = cal_euculid_distance(main_array, hog_data[j][1:hog_data_len_w - 1])
        else:
            distance_array[j] = cal_euculid_distance(main_array, hog_data[j][1:hog_data_len_w - 1])
    return distance_array


def plot_euculid_distance_figure( distance_array ):
    ele_count = len(distance_array)
    point_range = [0] * ele_count
    plt.scatter(point_range, distance_array)
    plt.show()
    return


# test region
dir_hog_feature = "../training-2000/HOG_feature/HOG_feature.csv" # path of HOG feature table
dir_color_feature_2000 = "../training-2000/color_feature/color_feature.csv"
temp = get_euculid_distance_array(dir_color_feature_2000, 'MWI_00Y7L33j7KcV6fTv.jpg')
plot_euculid_distance_figure(temp)
