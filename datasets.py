import os

import pandas as pd

from datetime import datetime

DATASET_PATTERN = "data_<date>.csv"
DATASET_DIR = "data"


def daily_dataset_exists():
    if os.path.isfile(get_daily_dataset_file()):
        return True
    else:
        return False


def get_daily_dataset_file():
    return DATASET_DIR + "/" + DATASET_PATTERN.replace("<date>", datetime.now().strftime("%d%m%Y"))


def save_rates(rates):
    frame = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    frame.to_csv(get_daily_dataset_file(), index=False)
