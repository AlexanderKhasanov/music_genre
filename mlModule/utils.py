import math
import pandas as pd


def get_data():
    return pd.read_csv('./datasets/kaggle_music_genre_train.csv')


def log_column(column):
    if column > 0:
        return math.log(column)
    return column
