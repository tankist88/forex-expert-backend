import atexit

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request
from os.path import join, dirname, realpath, isfile
from threading import Lock, RLock

from datetime import datetime, timedelta

import datasets
import model

SUPPORTED_INSTRUMENTS = []

REQUESTED_DATA_LENGTH = 25000

LEARN_PERIOD_HOURS = 3
LAST_LEARN_FILE = "last_learn.txt"

DATA_UPDATE_PERIOD_MINUTES = 20
LAST_DATA_UPDATE_FILE = "last_data_update.txt"

predict_lock = Lock()
supported_instruments_lock = RLock()


def write_time(kind, instrument, period):
    dir_path = join(dirname(realpath(__file__)), datasets.DATASET_DIR)
    f = open(dir_path + "/" + instrument + "_" + period + "_" + kind, "w")
    f.write(datetime.now().strftime("%Y.%m.%d %H:%M:%S"))
    f.close()


def read_time(kind, instrument, period):
    dir_path = join(dirname(realpath(__file__)), datasets.DATASET_DIR)
    file_path = dir_path + "/" + instrument + "_" + period + "_" + kind

    if isfile(file_path):
        f = open(file_path, "r")
        return datetime.strptime(f.read(), "%Y.%m.%d %H:%M:%S")
    else:
        return datetime.now() - timedelta(hours=24)


def get_base_params():
    instrument = request.json["instrument"].split(".")[0]
    period = request.json["period"].split("_")[1]
    return instrument, period


def job_function():
    for element in SUPPORTED_INSTRUMENTS:
        instrument = element["instrument"]
        period = element["period"]
        point = element["point"]

        last_learn_time = read_time(LAST_LEARN_FILE, instrument, period)
        if ((datetime.now() - last_learn_time).total_seconds() / 60 / 60) > LEARN_PERIOD_HOURS:
            filename = datasets.get_daily_dataset_file(instrument, period)
            if datasets.daily_dataset_exists(instrument, period):
                print("job_function: Start learning " + instrument + " " + period)
                x, y = model.read_data([filename], instrument, period, point)
                model.train_model(x, y, instrument, period, verbose=0)
                write_time(LAST_LEARN_FILE, instrument, period)
            else:
                print("job_function: Daily dataset " + str(filename) + " not found. Waiting...")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = datasets.DATASET_DIR
sched = BackgroundScheduler(daemon=True)
sched.add_job(job_function, 'cron', minute='*/15')
sched.start()


atexit.register(lambda: sched.shutdown(wait=False))


def get_rates():
    rates = request.json["rates"]
    arr = np.zeros((len(rates), 6), dtype='object')
    for i in range(len(rates)):
        arr[i][0] = datetime.strptime(rates[i]["time"], "%Y.%m.%d %H:%M:%S")
        arr[i][1] = rates[i]["open"]
        arr[i][2] = rates[i]["high"]
        arr[i][3] = rates[i]["low"]
        arr[i][4] = rates[i]["close"]
        arr[i][5] = rates[i]["tickVolume"]

    return np.asarray(sorted(arr, key=lambda x: x[0]))


@app.route('/forex-expert/predict', methods=['POST'])
def predict():
    instrument, period = get_base_params()

    last_learn_time = read_time(LAST_LEARN_FILE, instrument, period)
    if ((datetime.now() - last_learn_time).total_seconds() / 60 / 60) > LEARN_PERIOD_HOURS:
        print("predict: model " + instrument + " " + period + " is outdated")
        answer = "NONE"
    else:
        rates = get_rates()[:, [1, 2, 3, 4, 5]]

        predict_lock.acquire()
        trend = model.predict_trend(rates, instrument, period)
        predict_lock.release()

        if trend == "UP":
            answer = "OP_BUY"
        elif trend == "DOWN":
            answer = "OP_SELL"
        else:
            answer = "NONE"

    print("predict: " + str(answer))

    return jsonify(
        {
            "status": "success",
            "desc": "success",
            "answer": answer
        }
    )


@app.route('/forex-expert/upload', methods=['POST'])
def upload():
    instrument, period = get_base_params()
    datasets.save_rates(instrument, period, get_rates())
    write_time(LAST_DATA_UPDATE_FILE, instrument, period)
    return jsonify(
        {
            "status": "success",
            "desc": "success"
        }
    )


@app.route('/forex-expert/datacheck', methods=['POST'])
def datacheck():
    instrument, period = get_base_params()

    last_data_update_time = read_time(LAST_DATA_UPDATE_FILE, instrument, period)

    data_is_actual = ((datetime.now() - last_data_update_time).total_seconds() / 60) <= DATA_UPDATE_PERIOD_MINUTES
    if datasets.daily_dataset_exists(instrument, period) and data_is_actual:
        answer = "NONE"
    else:
        answer = "UPLOAD"

    print("datacheck: " + str(answer))

    return jsonify(
        {
            "status": "success",
            "desc": "success",
            "answer": answer,
            "train_length": REQUESTED_DATA_LENGTH,
            "predict_length": model.FRAME_LENGTH
        }
    )


@app.route('/forex-expert/appendinstrument', methods=['POST'])
def append_instrument():
    instrument, period = get_base_params()

    supported_instruments_lock.acquire()
    SUPPORTED_INSTRUMENTS.append(
        {
            "instrument": instrument,
            "period": period,
            "point": request.json["point"]
        }
    )
    supported_instruments_lock.release()

    print("append_instrument: SUPPORTED_INSTRUMENTS: " + str(SUPPORTED_INSTRUMENTS))

    return jsonify(
        {
            "status": "success",
            "desc": "success"
        }
    )


@app.route('/forex-expert/removeinstrument', methods=['POST'])
def remove_instrument():
    instrument, period = get_base_params()

    for_remove = None
    for element in SUPPORTED_INSTRUMENTS:
        if element["instrument"] == instrument and element["period"] == period:
            for_remove = element

    supported_instruments_lock.acquire()
    if for_remove is not None:
        SUPPORTED_INSTRUMENTS.remove(for_remove)
    supported_instruments_lock.release()

    print("remove_instrument: SUPPORTED_INSTRUMENTS: " + str(SUPPORTED_INSTRUMENTS))

    return jsonify(
        {
            "status": "success",
            "desc": "success"
        }
    )


if __name__ == '__main__':
    # app.run(port=8080, debug=False)
    app.run(host='0.0.0.0', port=80, debug=False)
