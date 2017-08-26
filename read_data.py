import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geopy.distance import vincenty


def read_data_base():
    train = pd.read_csv("data/train.csv", parse_dates=['pickup_datetime', 'dropoff_datetime'])
    test = pd.read_csv("data/test.csv", parse_dates=['pickup_datetime'])

    numeric_variables = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                         'dropoff_latitude', 'store_and_fwd_flag' ]

    target = "trip_duration"

    return train[numeric_variables], train[target], test[numeric_variables]

#Create some distance related columns
def compute_distance(x):
    #'pickup_longitude' 'pickup_latitude' 'dropoff_longitude' 'dropoff_latitude'
    lat_1 = x["pickup_latitude"]
    lat_2 = x["dropoff_latitude"]
    long_1 = x["pickup_longitude"]
    long_2 = x["dropoff_longitude"]

    return vincenty((lat_1, long_1), (lat_2, long_2)).miles

#Compute bearing from https://gist.github.com/jeromer/2005586
def calculate_initial_compass_bearing(pointA, pointB):

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def compute_bearing(x):
    #'pickup_longitude' 'pickup_latitude' 'dropoff_longitude' 'dropoff_latitude'
    lat_1 = x["pickup_latitude"]
    lat_2 = x["dropoff_latitude"]
    long_1 = x["pickup_longitude"]
    long_2 = x["dropoff_longitude"]

    return calculate_initial_compass_bearing((lat_1, long_1), (lat_2, long_2))

def read_data_add_features():
    train = pd.read_csv("data/train.csv", parse_dates=['pickup_datetime', 'dropoff_datetime'])
    test = pd.read_csv("data/test.csv", parse_dates=['pickup_datetime'])

    numeric_variables = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                         'dropoff_latitude', 'store_and_fwd_flag' ]

    target = "trip_duration"

    train["day_of_week"] = train.pickup_datetime.dt.dayofweek
    train["month"] = train.pickup_datetime.dt.month
    train["day_of_month"] = train.pickup_datetime.dt.day
    train["day_of_year"] = train.pickup_datetime.dt.dayofyear
    train["hour"] = train.pickup_datetime.dt.hour
    train["minute"] = train.pickup_datetime.dt.minute
    train["distance"] = train.apply(lambda x: compute_distance(x), axis=1)
    train["bearing"] = train.apply(lambda x: compute_bearing(x), axis=1)

    test["day_of_week"] = test.pickup_datetime.dt.dayofweek
    test["month"] = test.pickup_datetime.dt.month
    test["day_of_month"] = test.pickup_datetime.dt.day
    test["day_of_year"] = test.pickup_datetime.dt.dayofyear
    test["hour"] = test.pickup_datetime.dt.hour
    test["minute"] = test.pickup_datetime.dt.minute
    test["distance"] = test.apply(lambda x: compute_distance(x), axis=1)
    test["bearing"] = test.apply(lambda x: compute_bearing(x), axis=1)

    add_features = ["day_of_week", "month", "day_of_month", "day_of_year", "hour", "minute", "distance", "bearing"]

    return train[numeric_variables+add_features], train[target], test[numeric_variables+add_features]


