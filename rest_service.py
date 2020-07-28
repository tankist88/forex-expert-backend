import atexit

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

from datetime import datetime

import datasets
import model

SUPPORTED_INSTRUMENTS = [
    {
        "instrument": "EURUSD",
        "period": "M5"
    }
]


def get_base_params():
    instrument = request.json["instrument"].split(".")[0]
    period = request.json["period"].split("_")[1]
    return instrument, period


def job_function():
    for element in SUPPORTED_INSTRUMENTS:
        instrument = element["instrument"]
        period = element["period"]
        filename = datasets.get_daily_dataset_file(instrument, period)
        if datasets.daily_dataset_exists(instrument, period):
            print('Start learning')
            x, y = model.read_data([filename], instrument, period)
            model.train_model(x, y, instrument, period)
        else:
            print("Daily dataset " + str(filename) + " not found. Waiting...")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = datasets.DATASET_DIR
sched = BackgroundScheduler(daemon=True)
sched.add_job(job_function, 'cron', minute='*')
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


@app.route('/forex-expert/whatshouldido', methods=['POST'])
def what_should_i_do():
    instrument, period = get_base_params()

    rates = get_rates()[:, [1, 2, 3, 4, 5]]
    trend = model.predict_trend(rates, instrument, period)

    if trend == "UP":
        answer = "OP_BUY"
    elif trend == "DOWN":
        answer = "OP_SELL"
    else:
        answer = "NONE"

    print("what should i do?: " + str(answer))

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
    return jsonify(
        {
            "status": "success",
            "desc": "success"
        }
    )


@app.route('/forex-expert/doyouneeddata', methods=['POST'])
def do_you_need_data():
    instrument, period = get_base_params()

    if datasets.daily_dataset_exists(instrument, period):
        answer = "no"
    else:
        answer = "yes"

    print("do you need data?: " + str(answer))

    return jsonify(
        {
            "status": "success",
            "desc": "success",
            "answer": answer
        }
    )


if __name__ == '__main__':
    app.run(port=8080, debug=False)
    # app.run(host='0.0.0.0', port=80, debug=False)
