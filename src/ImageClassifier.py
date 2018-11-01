# the main execute file of this project

import IOMethod as iom


image_directory_2000 = "../training-2000/images/"
dir_hog_feature_2000 = "../training-2000/HOG_feature/HOG_feature.csv"  # path of HOG feature table
dir_color_feature_2000 = "../training-2000/color_feature/color_feature.csv"

image_directory_test = "../test/images/"
dir_hog_feature_test = "../test/HOG_feature/HOG_feature.csv"
dir_color_feature_test = "../test/color_feature/color_feature.csv"


def generate_hog_for_training_2000():
    iom.generate_hog_csv(dir_hog_feature_2000, image_directory_2000)
    return


def generate_color_for_training_2000():
    iom.generate_color_csv(dir_color_feature_2000, image_directory_2000)
    return


def generate_hog_for_test():
    iom.generate_hog_csv(dir_hog_feature_test, image_directory_test)
    return


def generate_color_for_test():
    iom.generate_color_csv(dir_color_feature_test, image_directory_test)
    return


# main
generate_hog_for_training_2000()
generate_hog_for_test()
# generate_color_for_training_2000()
# generate_color_for_test()
