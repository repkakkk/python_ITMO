import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import math

def distance_to_center(lat, lon):
    # Convert latitude and longitude from degrees to radians
    lat_center, lon_center = 55.75222, 37.61556
    lat_center, lon_center = map(math.radians, [lat_center, lon_center])

    lat, lon = map(math.radians, [lat, lon])

    # Haversine formula
    dlon = lon - lon_center
    dlat = lat - lat_center

    a = math.sin(dlat / 2)**2 + math.cos(lat) * math.cos(lat_center) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers (use 3956 for miles)
    r = 6371.0

    return c * r

def prepare_data():
    df = pd.read_csv('data/realty_data.csv')

    df["rooms"] = df["rooms"].fillna(1)
    df["floor"] = df["floor"].fillna(1)
    df['distance_to_center'] = df.apply(lambda x: distance_to_center(x['lat'], x['lon']), axis = 1)

    df = df.drop(columns = ['period', 'postcode','address_name', 'object_type', 'settlement', 'district', 'area', 'description', 'source', 'city', 'product_name', 'lat', 'lon'])

    df['price'] = df['price'].astype(np.uint32)
    df['total_square'] = df['total_square'].astype(np.uint16)
    df['rooms'] = df['rooms'].astype(np.uint8)
    df['floor'] = df['floor'].astype(np.uint8)

    return df


def train_model(df):
    y = df["price"]
    X = df.drop(columns = ['price'])


    model = LinearRegression()
    model.fit(X, y);

    with open('model_fitted.pkl', 'wb') as file:
        pickle.dump(model, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model
