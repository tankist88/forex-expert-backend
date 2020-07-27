import os
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization, Dropout
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.externals import joblib


MODEL_FILENAME = "model.hdf5"
SCALER_FILENAME = "scaler.dump"
FRAME_LENGTH = 10
N_FEATURES = 3
PRICE_DELTA = 100
SCALE = pow(10, 5)
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

    filters = [1, 2, 3]

    conv_layers = []
    for f in filters:
        if f > 1:
            conv = Conv2D(32, (f, f), kernel_initializer='he_normal', activation='relu', padding='same')(inp)
            conv = BatchNormalization()(conv)
            # conv = Conv2D(32, (f, f), kernel_initializer='he_normal', activation='relu', padding='same')(conv)
            # conv = BatchNormalization()(conv)
            # conv = MaxPooling2D((2, 2))(conv)
        else:
            conv = Conv2D(32, (f, f), kernel_initializer='he_normal', activation='relu')(inp)
            conv = BatchNormalization()(conv)
        conv = Flatten()(conv)
        conv_layers.append(conv)

    x = Concatenate(axis=1)(conv_layers)
    # x = Dense(units=16, init='he_normal', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    # x = Dense(units=16, init='he_normal', activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    x = Dense(units=32, init='he_normal', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outp = Dense(units=3, init='glorot_normal', activation='softmax')(x)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[f1])

    return model


def create_features(data):
    new_data = np.zeros((len(data), N_FEATURES))

    for i in range(len(data)):
        new_data[i][0] = data.iloc[i][0] - data.iloc[i][3]
        new_data[i][1] = data.iloc[i][1] - data.iloc[i][2]
        new_data[i][2] = pow(new_data[i][0], 2)

    return new_data


def read_data(files):
    data_frames = []

    for file in files:
        data_frames.append(pd.read_csv(file))

    data = pd.concat(data_frames, axis=0)
    data = data.drop(columns=DROP_COLUMNS)

    new_data = create_features(data)

    print("features shape: " + str(new_data.shape))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(new_data)

    frames_short = []
    frames_long = []
    frames_other = []

    for i in range(FRAME_LENGTH, len(data)):
        close0 = data.iloc[i][3] * SCALE
        close1 = data.iloc[i - 1][3] * SCALE
        if close0 - close1 >= PRICE_DELTA:
            copy_sub_frame(i - FRAME_LENGTH, i, scaled_data, frames_long)
        elif close1 - close0 >= PRICE_DELTA:
            copy_sub_frame(i - FRAME_LENGTH, i, scaled_data, frames_short)
        else:
            copy_sub_frame(i - FRAME_LENGTH, i, scaled_data, frames_other)

    frames_short = np.asarray(frames_short)
    frames_long = np.asarray(frames_long)
    frames_other = np.asarray(frames_other)[0: max(len(frames_short), len(frames_long)) * 7]

    y_short = get_labels(LABELS[0], (len(frames_short), 3))
    y_long = get_labels(LABELS[1], (len(frames_long), 3))
    y_others = get_labels(LABELS[2], (len(frames_other), 3))

    x = np.concatenate((frames_short, frames_other, frames_long))
    y = np.concatenate((y_short, y_others, y_long))

    x_shuffled, y_shuffled = shuffle(x, y)

    joblib.dump(scaler, SCALER_FILENAME)

    return x_shuffled, y_shuffled


def train_model(x, y):
    x = np.expand_dims(x, axis=3)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42)

    model = build_classifier((x.shape[1], x.shape[2], 1))
    model.fit(x_train, y_train,
              batch_size=40,
              epochs=60,
              validation_data=(x_valid, y_valid),
              callbacks=[
                  ModelCheckpoint(
                      filepath=MODEL_FILENAME,
                      save_best_only=True,
                      monitor='val_loss',
                      save_weights_only=False,
                      mode='min')],
              verbose=2)


def predict_trend(frame, model=None, scaler=None):
    if frame.shape[0] != FRAME_LENGTH:
        print("Invalid frame length!")
        return "NONE"

    if model is None and os.path.isfile(MODEL_FILENAME):
        model = load_model(MODEL_FILENAME, custom_objects={'f1': f1})
    elif model is None:
        print("Model not found!")
        return "NONE"

    if scaler is None and os.path.isfile(SCALER_FILENAME):
        scaler = joblib.load(SCALER_FILENAME)
    elif scaler is None:
        print("Scaler not found!")
        return "NONE"

    scaled_data = scaler.transform(create_features(pd.DataFrame(frame)))

    frames = []
    copy_sub_frame(0, len(scaled_data), scaled_data, frames)

    y_pred = model.predict(np.expand_dims(np.asarray(frames), axis=3))

    label = LABELS[np.argmax(y_pred)]

    if label == 0:
        return "DOWN"
    elif label == 1:
        return "UP"
    else:
        return "NONE"


def test_model(data_file):
    model = load_model(MODEL_FILENAME, custom_objects={'f1': f1})
    scaler = joblib.load(SCALER_FILENAME)

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

        trend = predict_trend(frame, model, scaler)

        close0 = data.iloc[i][3] * SCALE
        close1 = data.iloc[i - 1][3] * SCALE

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

    print("true_predicts: " + str(true_predicts))
    print("false_predicts: " + str(false_predicts))
    print("m: " + str(true_predicts/false_predicts))
    print("true_base_points: " + str(true_base_points))
    print("false_base_points: " + str(false_base_points))

    plt.figure(figsize=(50, 20))
    for i in range(len(lb)):
        if lb[i] == 0:
            plt.axvline(i, 0, 1.5, color='red')
        elif lb[i] == 1:
            plt.axvline(i, 0, 1.5, color='green')
    plt.plot(data.iloc[:, [3]].values, color='black')
    plt.savefig("res.png")
    plt.close()


if __name__ == '__main__':
    features, labels = read_data(['data/train.csv'])
    train_model(features, labels)
    test_model('data/test.csv')
