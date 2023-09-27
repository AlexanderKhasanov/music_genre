import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from mlModule.constants import RANDOM_STATE


def preprocessing_string_column(data, columns):
    for column in columns:
        data[column] = data[column].apply(lambda value: ' '.join(value.split()).lower())
    return data


def delete_unnecessary_columns(data, columns):
    return data.drop(columns, axis=1)


def get_features_target(data):
    return data.drop('music_genre', axis=1), data['music_genre']


def get_samples(data, is_three_samples=False):
    features, target = get_features_target(data)
    features_train, feature_valid, target_train, target_valid = train_test_split(
        features, target,
        test_size=.2, stratify=target,
        random_state=RANDOM_STATE
    )
    if is_three_samples:
        features_train, features_test, target_train, target_test = train_test_split(
            features_train, target_train,
            test_size=.25, stratify=target_train,
            random_state=RANDOM_STATE
        )
        return features_train, feature_valid, features_test, target_train, target_valid, target_test
    return features_train, feature_valid, target_train, target_valid


def ohe(df, encoder, columns):
    encoder_categorical = pd.DataFrame(
        encoder.transform(df[columns]).toarray(),
        index=df.index,
        columns=encoder.get_feature_names_out()
    )
    df = df.join(encoder_categorical)
    df.drop(columns, inplace=True, axis=1)
    return df


def removing_outliers_by_category(data, columns):
    for column in columns:
        for genre in data['music_genre'].unique():
            statistics = data.loc[
                data['music_genre'] == genre, column
            ].describe()
            q1 = statistics['25%']
            q3 = statistics['75%']
            iqr = q3 - q1
            min = q1 - 1.5 * iqr
            max = q3 + 1.5 * iqr
            data = data.query(f'music_genre == @genre & @min <= {column} <= @max | music_genre != @genre')
    return data


def upsample(features, target):
    features['music_genre'] = target
    max_samples = features.groupby('music_genre').count().max().iloc[0]
    data_upsample = pd.DataFrame()
    for group_name, group_data in features.groupby('music_genre'):
        count_sample = group_data.count().iloc[0]
        repeat = round(max_samples / count_sample, 1)
        if repeat % 10 <= 6:
            repeat = math.floor(repeat)
        else:
            repeat = math.ceil(repeat)
        data_upsample = pd.concat([data_upsample] + [group_data] * repeat)
    data_upsample = shuffle(data_upsample, random_state=RANDOM_STATE)
    return get_features_target(data_upsample)


def normalize_columns(data, columns):
    scaler = MinMaxScaler()
    data[columns] = pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns, index=data.index)
    return data


def get_tempo(row):
    if row['acousticness'] < 0.4:
        if row['energy'] > 0.75:
            if row['speechiness'] > 0.1:
                # Медиана для Electronic
                return 125
            # Медиана для Anime
            return 128
        # Медиана для остальных жанров
        return 120
    elif row['acousticness'] < 0.8:
        # Медиана для Jazz
        return 105
    # Медиана для classical
    return 95


def get_duration(row):
    if row['acousticness'] < 0.4:
        if row['speechiness'] > 0.1:
            # Медиана для hip-hop и rap
            return 207000
        if row['instrumentalness'] > 0.1:
            # Медиана для Electronic
            return 236000
        if row['energy'] > 0.75:
            # Медиана для Anime
            return 230000
        elif row['energy'] > 0.7:
            if row['danceability'] < 0.5:
                # Медиана для Anime (еще один вариант)
                return 230000
            # Медиана для alternative, rock
            return 220000
        if row['valence'] < 0.54 and row['loudness'] > -7:
            # Медиана для country
            return 207000
        elif row['valence'] > 0.55:
            # Медиана для blues и country
            return 213000
        # Медиана для blues
        return 220000
    elif row['acousticness'] < 0.8:
        # Медиана для Jazz
        return 236000
    # Медиана для classical
    return 245000
