import numpy as np
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, request

import datasets
import model


def job_function():
    print('******************')
    print('* Start learning *')
    print('******************')

    if datasets.daily_dataset_exists():
        x, y = model.read_data([datasets.get_daily_dataset_file()])
        model.train_model(x, y)
    else:
        print("Daily dataset " + str(datasets.get_daily_dataset_file()) + " not found. Waiting...")


app = Flask(__name__)
sched = BackgroundScheduler(daemon=True)
sched.add_job(job_function, 'cron', minute='*')
sched.start()


atexit.register(lambda: sched.shutdown(wait=False))


def get_rates():
    rates = request.json["rates"]
    frame = np.zeros((model.FRAME_LENGTH, 6))
    for i in range(len(rates)):
        frame[i][0] = rates[i]["time"]
        frame[i][1] = rates[i]["open"]
        frame[i][2] = rates[i]["high"]
        frame[i][3] = rates[i]["low"]
        frame[i][4] = rates[i]["close"]
        frame[i][5] = rates[i]["volume"]

    return frame


@app.route('/forex-expert/whatshouldido', methods=['POST'])
def what_should_i_do():
    trend = model.predict_trend(get_rates())

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
    datasets.save_rates(get_rates())
    return jsonify(
        {
            "status": "success",
            "desc": "success"
        }
    )


@app.route('/forex-expert/doyouneeddata', methods=['POST'])
def do_you_need_data():
    if datasets.daily_dataset_exists():
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
    app.run(port=8080, debug=True)
    # app.run(host='0.0.0.0', port=80, debug=True)
