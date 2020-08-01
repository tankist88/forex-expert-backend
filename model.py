import pickle
from os import rename, remove
from os.path import join, dirname, realpath, isfile

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import datasets
from logger_config import logger

MODEL_FILENAME = "model.h5"
SCALER_FILENAME = "scaler.dump"
FRAME_LENGTH = 10
FRAME_COLUMNS = 5
N_FEATURES = 4
PRICE_DELTA = 100
MAX_MODELS_COUNT = 7
LABELS = [
    0,  # short
    1,  # long
    2   # others
]
DROP_COLUMNS = [
    'time'
]


def copy_sub_frame(start, end, src, dest):
    line = []
    for j in range(start, end):
        line.append(src[j])
    line = np.asarray(line)
    dest.append(line)


def get_labels(label, shape):
    arr = np.zeros(shape)
    for n in range(len(arr)):
        arr[n][label] = 1
    return arr


def create_scaler_filename(instrument, period):
    dir_path = join(dirname(realpath(__file__)), datasets.DATASET_DIR)
    return dir_path + "/" + str(instrument) + "_" + str(period) + "_" + SCALER_FILENAME


def create_model_filename(instrument, period, temp=False, index=None):
    dir_path = join(dirname(realpath(__file__)), datasets.DATASET_DIR)
    if temp:
        return dir_path + "/TEMP_" + str(instrument) + "_" + str(period) + "_" + ("" if index is None else (str(index) + "_")) + MODEL_FILENAME
    else:
        return dir_path + "/" + str(instrument) + "_" + str(period) + "_" + ("" if index is None else (str(index) + "_")) + MODEL_FILENAME


# https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
def f1(y_true, y_predict):
    y_pred_rnd = K.round(y_predict)
    tp = K.sum(K.cast(y_true * y_pred_rnd, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred_rnd, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred_rnd), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1_res = 2 * p * r / (p + r + K.epsilon())
    f1_res = tf.where(tf.is_nan(f1_res), tf.zeros_like(f1_res), f1_res)
    return K.mean(f1_res)


def build_classifier(input_shape):
    inp = Input(shape=input_shape)

    filters = [1, 3]

    conv_layers = []
    for f in filters:
        if f > 1:
            conv = Conv2D(32, (f, f), kernel_initializer='glorot_normal', activation='sigmoid', padding='same')(inp)
            conv = BatchNormalization()(conv)
        else:
            conv = Conv2D(32, (f, f), kernel_initializer='glorot_normal', activation='sigmoid')(inp)
            conv = BatchNormalization()(conv)
        conv = Flatten()(conv)
        conv_layers.append(conv)

    x = Concatenate(axis=1)(conv_layers)
    x = Dense(units=32, kernel_initializer='glorot_normal', activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(units=16, kernel_initializer='glorot_normal', activation='sigmoid')(x)
    x = BatchNormalization()(x)
    outp = Dense(units=3, kernel_initializer='glorot_normal', activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[f1])

    return model


def create_features(data, point):
    new_data = np.zeros((len(data), N_FEATURES), dtype='float64')

    max_volume = data.iloc[:, [4]].values.max()

    for i in range(len(data)):
        new_data[i][0] = (data.iloc[i][0] - data.iloc[i][3]) / point
        new_data[i][1] = (data.iloc[i][1] - data.iloc[i][2]) / point
        new_data[i][2] = pow(new_data[i][0], 2)
        new_data[i][3] = (data.iloc[i][4] / max_volume) * 100

    return new_data


def read_data(files, instrument, period, point):
    data_frames = []

    for file in files:
        data_frames.append(pd.read_csv(file))

    data = pd.concat(data_frames, axis=0)
    data = data.drop(columns=DROP_COLUMNS)

    scaler = StandardScaler()
    scaler.fit(create_features(data, point))

    frames_short = []
    frames_long = []
    frames_other = []

    for i in range(FRAME_LENGTH, len(data)):
        close0 = round(data.iloc[i][3] / point)
        close1 = round(data.iloc[i - 1][3] / point)

        frame = []
        for j in range(i - FRAME_LENGTH, i):
            frame.append(data.iloc[j])
        frame = np.asarray(frame)

        scaled_data = scaler.transform(create_features(pd.DataFrame(frame), point))

        if close0 - close1 >= PRICE_DELTA:
            frames_long.append(scaled_data)
        elif close1 - close0 >= PRICE_DELTA:
            frames_short.append(scaled_data)
        else:
            frames_other.append(scaled_data)

    frames_short = np.asarray(frames_short)
    frames_long = np.asarray(frames_long)
    frames_other = np.asarray(frames_other)[0: max(len(frames_short), len(frames_long)) * 3]

    y_short = get_labels(LABELS[0], (len(frames_short), 3))
    y_long = get_labels(LABELS[1], (len(frames_long), 3))
    y_others = get_labels(LABELS[2], (len(frames_other), 3))

    x = np.concatenate((frames_short, frames_other, frames_long))
    y = np.concatenate((y_short, y_others, y_long))

    x_shuffled, y_shuffled = shuffle(x, y)

    joblib.dump(scaler, create_scaler_filename(instrument, period))

    return x_shuffled, y_shuffled


def train_model(x, y, instrument, period, val_loss_limit=0.88, verbose=2):
    for i in range(MAX_MODELS_COUNT):
        model_file = create_model_filename(instrument, period, index=i)
        if isfile(model_file):
            remove(model_file)

    logger.info("PREVIOUS MODELS REMOVED")

    temp_model_file = create_model_filename(instrument, period, temp=True)

    if isfile(temp_model_file):
        remove(temp_model_file)

    x = np.expand_dims(x, axis=3)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42)

    models_count = 0
    while models_count < MAX_MODELS_COUNT:
        model = build_classifier((x.shape[1], x.shape[2], 1))
        history = model.fit(x_train, y_train,
                            batch_size=10,
                            epochs=20,
                            validation_data=(x_valid, y_valid),
                            callbacks=[
                                ModelCheckpoint(
                                    filepath=temp_model_file,
                                    save_best_only=True,
                                    monitor='val_loss',
                                    save_weights_only=False,
                                    mode='min')],
                            verbose=verbose)

        for val_loss in history.history["val_loss"]:
            if val_loss < val_loss_limit:
                rename(temp_model_file, create_model_filename(instrument, period, index=models_count))
                models_count += 1
                break

    logger.info("******* MODEL %s %s TRAINED SUCCESS ********", instrument, period)

    K.clear_session()


def predict_trend(frame, instrument, period, point, models=None, scaler=None, clear_session=True):
    if frame.shape[0] != FRAME_LENGTH or frame.shape[1] != FRAME_COLUMNS:
        logger.error("Invalid frame shape!")
        return "NONE"

    models_exists = True
    model_files = []
    for i in range(MAX_MODELS_COUNT):
        model_file = create_model_filename(instrument, period, index=i)
        if not isfile(model_file):
            models_exists = False
        model_files.append(model_file)

    if models is None and models_exists:
        models = []
        for model_file in model_files:
            models.append(load_model(model_file, custom_objects={'f1': f1}))
    elif models is None:
        logger.error("Models not found!")
        return "NONE"

    scaler_file = create_scaler_filename(instrument, period)
    if scaler is None and isfile(scaler_file):
        scaler = joblib.load(scaler_file)
    elif scaler is None:
        logger.error("Scaler not found!")
        return "NONE"

    scaled_data = scaler.transform(create_features(pd.DataFrame(frame), point))

    frames = []
    copy_sub_frame(0, FRAME_LENGTH, scaled_data, frames)

    predicts = []
    for model in models:
        predicts.append(model.predict(np.expand_dims(np.asarray(frames), axis=3)))

    predicts = np.concatenate(predicts)

    y_pred = np.zeros((1, 3))

    y_pred[0][0] = predicts[:, [0]].mean()
    y_pred[0][1] = predicts[:, [1]].mean()
    y_pred[0][2] = predicts[:, [2]].mean()

    if clear_session:
        K.clear_session()

    label = LABELS[np.argmax(y_pred)]

    if label == 0:
        return "DOWN"
    elif label == 1:
        return "UP"
    else:
        return "NONE"


def test_model(data_file, instrument, period, point, plot_results=False):
    scaler_file = create_scaler_filename(instrument, period)

    models = []
    for i in range(MAX_MODELS_COUNT):
        models.append(load_model(create_model_filename(instrument, period, index=i), custom_objects={'f1': f1}))

    scaler = joblib.load(scaler_file)

    data = pd.read_csv(data_file)
    data = data.drop(columns=DROP_COLUMNS)

    true_base_points = 0.0
    false_base_points = 0.0
    true_predicts = 0
    false_predicts = 0
    lb = np.zeros(len(data))
    lb.fill(-1)
    for i in range(FRAME_LENGTH, len(data)):
        frame = []
        for j in range(i - FRAME_LENGTH, i):
            frame.append(data.iloc[j])
        frame = np.asarray(frame)

        trend = predict_trend(frame, instrument, period, point, models, scaler, False)

        close0 = round(data.iloc[i][3] / point)
        close1 = round(data.iloc[i - 1][3] / point)

        if trend == "UP":
            lb[i] = 1
            if close0 - close1 >= 0:
                true_predicts += 1
                true_base_points += abs(close0 - close1)
            else:
                false_predicts += 1
                false_base_points += abs(close0 - close1)
        elif trend == "DOWN":
            lb[i] = 0
            if close1 - close0 >= 0:
                true_predicts += 1
                true_base_points += abs(close1 - close0)
            else:
                false_predicts += 1
                false_base_points += abs(close1 - close0)
        else:
            lb[i] = 2

    logger.info("***********************************")
    logger.info("* DATA FILE: " + str(data_file) + "       *")
    logger.info("***********************************")
    logger.info("true_predicts: " + str(true_predicts))
    logger.info("false_predicts: " + str(false_predicts))
    logger.info("predicts rate: " + str(true_predicts/false_predicts))
    logger.info("true_base_points: " + str(true_base_points))
    logger.info("false_base_points: " + str(false_base_points))

    if plot_results:
        plt.figure(figsize=(50, 20))
        for i in range(len(lb)):
            if lb[i] == 0:
                plt.axvline(i, 0, 1.5, color='red')
            elif lb[i] == 1:
                plt.axvline(i, 0, 1.5, color='green')
        plt.plot(data.iloc[:, [3]].values, color='black')
        plt.savefig("res.png")
        plt.close()

    return true_predicts, false_predicts


if __name__ == '__main__':
    features, labels = read_data(['data/train.csv'], "EURUSD", "M5", point=0.00001)

    with open('data/train_proc_data.pickle', 'wb') as f:
        pickle.dump((features, labels), f)

    with open('data/train_proc_data.pickle', 'rb') as f:
        (features, labels) = pickle.load(f)

    # tp = 1
    # fp = 100
    # while tp < 50 or tp/fp < 1.5:
    #     train_model(features, labels, "EURUSD", "M5", val_loss_limit=0.88)
    #     tp, fp = test_model('data/test1.csv', "EURUSD", "M5", point=0.00001, plot_results=True)

    train_model(features, labels, "EURUSD", "M5", val_loss_limit=0.88)
    test_model('data/test1.csv', "EURUSD", "M5", point=0.00001, plot_results=True)
