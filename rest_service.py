import numpy as np
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

import datasets
import model

SUPPORTED_INSTRUMENTS = [
    {
        "instrument": "EURUSD",
        "period": "M5"
    }
]


def job_function():
    for element in SUPPORTED_INSTRUMENTS:
        instrument = element["instrument"]
        period = element["period"]
        filename = datasets.get_daily_dataset_file(instrument, period)
        if datasets.daily_dataset_exists(instrument, period):
            print('******************')
            print('* Start learning *')
            print('******************')
            x, y = model.read_data([filename], instrument, period)
            model.train_model(x, y, instrument, period)
        else:
            print("Daily dataset " + str(filename) + " not found. Waiting...")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = datasets.DATASET_DIR
sched = BackgroundScheduler(daemon=True)
sched.add_job(job_function, 'cron', hour='*')
sched.start()


atexit.register(lambda: sched.shutdown(wait=False))


def get_rates(without_time=False):
    rates = request.json["rates"]
    if without_time:
        frame = np.zeros((len(rates), 5), dtype='object')
        for i in range(len(rates)):
            frame[i][0] = rates[i]["open"]
            frame[i][1] = rates[i]["high"]
            frame[i][2] = rates[i]["low"]
            frame[i][3] = rates[i]["close"]
            frame[i][4] = rates[i]["tickVolume"]
    else:
        frame = np.zeros((len(rates), 6), dtype='object')
        for i in range(len(rates)):
            frame[i][0] = rates[i]["time"]
            frame[i][1] = rates[i]["open"]
            frame[i][2] = rates[i]["high"]
            frame[i][3] = rates[i]["low"]
            frame[i][4] = rates[i]["close"]
            frame[i][5] = rates[i]["tickVolume"]

    return frame[::-1]


@app.route('/forex-expert/whatshouldido', methods=['POST'])
def what_should_i_do():
    instrument = request.json["instrument"].split(".")[0]
    period = request.json["period"].split("_")[1]

    trend = model.predict_trend(get_rates(without_time=True), instrument, period)

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
    instrument = request.json["instrument"].split(".")[0]
    period = request.json["period"].split("_")[1]

    datasets.save_rates(instrument, period, get_rates())
    return jsonify(
        {
            "status": "success",
            "desc": "success"
        }
    )


@app.route('/forex-expert/doyouneeddata', methods=['POST'])
def do_you_need_data():
    instrument = request.json["instrument"].split(".")[0]
    period = request.json["period"].split("_")[1]

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
    # app.run(port=8080, debug=True)
    app.run(host='0.0.0.0', port=80, debug=False)
