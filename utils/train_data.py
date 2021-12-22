import os
import numpy as np
import cv2


# making training & validation data from images folders
def create_training_data(img_path, label_path, img_shape, val, seed, rand=False):
    # training data: x=image, y=label, filename=filename
    train_x = []
    train_y = []
    train_filename = []

    # validation data: x=image, y=label, filename=filename
    val_x = []
    val_y = []
    val_filename = []

    # list dir of image and label folders
    img_filelist = os.listdir(img_path)
    label_filelist = os.listdir(label_path)

    # calculate images numbers
    file_num = len(img_filelist)

    # picking validation data after val_trigger numbers of images. val = validation percentage(should between 0 and 1)
    val_trigger = int(file_num * (1 - val))

    # random
    if rand:
        np.random.seed(seed)
        index = np.random.permutation(file_num)
    else:
        index = np.arange(file_num)

    count = 0
    for i in index:
        # read images and labels
        img = cv2.imread(img_path + '/' + img_filelist[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path + '/' + label_filelist[i], cv2.IMREAD_GRAYSCALE)

        # append img into testing data or validation data
        if count < val_trigger:
            train_x.append(img)
            train_y.append(label)
            train_filename.append(img_filelist[i])
        else:
            val_x.append(img)
            val_y.append(label)
            val_filename.append(img_filelist[i])

        count += 1

    # reshape into img_shape
    train_x = np.array(train_x).reshape(-1, img_shape[0], img_shape[1], img_shape[2])
    train_y = np.array(train_y).reshape(-1, img_shape[0], img_shape[1], img_shape[2])
    val_x = np.array(val_x).reshape(-1, img_shape[0], img_shape[1], img_shape[2])
    val_y = np.array(val_y).reshape(-1, img_shape[0], img_shape[1], img_shape[2])

    return train_x, train_y, train_filename, val_x, val_y, val_filename


if __name__ == "__main__":
    pass
