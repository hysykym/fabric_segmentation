import os
import numpy as np
import cv2


# making testing data from images folder
def create_test_data(img_path, img_shape):
    # testing data: x=image, filename=filename
    test_x = []
    test_filename = []

    # list dir of image dir
    img_filelist = os.listdir(img_path)

    # calculate images numbers
    file_num = len(img_filelist)

    index = np.arange(file_num)

    for i in index:
        # read images
        img = cv2.imread(img_path + '/' + img_filelist[i], cv2.IMREAD_GRAYSCALE)

        # append img into testing data
        test_x.append(img)
        test_filename.append(img_filelist[i])

    test_x = np.array(test_x).reshape(-1, img_shape[0], img_shape[1], img_shape[2])

    return test_x, test_filename


if __name__ == "__main__":
    pass
