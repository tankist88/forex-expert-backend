from datetime import datetime
from os.path import join, dirname, realpath, isfile

import pandas as pd

DATASET_PATTERN = "data_<instrument>_<period>_<date>.csv"
DATASET_DIR = "data"


def daily_dataset_exists(instrument, period):
    if isfile(get_daily_dataset_file(instrument, period)):
        return True
    else:
        return False


def get_daily_dataset_file(instrument, period):
    return join(dirname(realpath(__file__)), DATASET_DIR) + "/" + DATASET_PATTERN\
        .replace("<date>", datetime.now().strftime("%d%m%Y"))\
        .replace("<instrument>", instrument)\
        .replace("<period>", period)


def save_rates(instrument, period, rates):
    frame = pd.DataFrame(data=rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    frame.to_csv(get_daily_dataset_file(instrument, period), index=False)
