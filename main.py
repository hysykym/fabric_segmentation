import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping

from utils.check_folder import check_folder
from utils.train_data import create_training_data
from utils.test_data import create_test_data
from utils.show_train_history import show_train_history

# Build Segmentation model
def segmentation(img_shape=(320,400,1)):
    inputs = Input(img_shape)

    conv1_1 = Conv2D(48, 3, activation='relu', padding='same')(inputs)
    conv1_2 = Conv2D(48, 3, activation='relu', padding='same')(conv1_1)
    drop1 = Dropout(0.4)(conv1_2)
    pool1 = MaxPool2D((2,2))(drop1)

    conv2_1 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(32, 3, activation='relu', padding='same')(conv2_1)
    drop2 = Dropout(0.5)(conv2_2)
    pool2 = MaxPool2D((2, 2))(drop2)

    conv3_1 = Conv2D(24, 3, activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(24, 3, activation='relu', padding='same')(conv3_1)
    drop3 = Dropout(0.5)(conv3_2)

    up4 = Conv2D(32, 2, activation='relu', padding='same')(UpSampling2D((2,2))(drop3))
    merge4 = concatenate([conv2_2, up4], axis=3)
    conv4_1 = Conv2D(32, 3, activation='relu', padding='same')(merge4)
    conv4_2 = Conv2D(32, 3, activation='relu', padding='same')(conv4_1)

    up5 = Conv2D(48, 2, activation='relu', padding='same')(UpSampling2D((2,2))(conv4_2))
    merge5 = concatenate([conv1_2, up5], axis=3)
    conv5_1 = Conv2D(48, 3, activation='relu', padding='same')(merge5)
    conv5_2 = Conv2D(48, 3, activation='relu', padding='same')(conv5_1)
    conv5_3 = Conv2D(2, 3, activation='relu', padding='same')(conv5_2)

    conv6 = Conv2D(1, 1, activation='sigmoid')(conv5_3)

    model = Model(inputs, conv6)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

if __name__ == "__main__":
    img_shape = (320, 400, 1)
    epochs = 100
    batch_size = 8
    validation_split = 0.1

    img_path = './data/train/img'
    label_path = './data/train/label'

    print('create training data')
    # create training data
    train_x, train_y, train_filelist, val_x, val_y, val_filelist = create_training_data(img_path, label_path, img_shape, validation_split, seed=10, rand=True)
    print('train shape:{}'.format(train_x.shape))
    print('val shape:{}'.format(val_x.shape))

    print('training model')
    # check weight path
    check_folder('./weight')
    model_path = './weight/segmentation.h5'

    # create model
    # model = segmentation(img_shape)

    # or load pre-trained model
    model = tf.keras.models.load_model('./weight/pre_trained.h5')

    # Early Stop
    early_stopping_monitor = EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=20,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    # train model and show history
    history = model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_x, val_y), callbacks=[early_stopping_monitor])
    show_train_history(history)

    # save model
    model.save(model_path)

    # test data
    print('test model')
    test_path = './data/test/img'
    test_x, test_filelist = create_test_data(test_path, img_shape)

    # result path
    result_path = './result'
    check_folder(result_path)

    # segmentation threshold
    pixel_threshold = 0.4

    for i in range(len(test_filelist)):
        # predict testing img
        img = model.predict(np.array(test_x[i]).reshape(1, img_shape[0], img_shape[1], img_shape[2]))
        img = img[0]

        # set pixel to black if less than threshold, set white vice versa (for visualization)
        img[img <= pixel_threshold] = 0
        img[img > pixel_threshold] = 255

        # save result
        cv2.imwrite(result_path + '/' +test_filelist[i], img)