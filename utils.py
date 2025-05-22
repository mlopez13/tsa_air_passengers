import numpy as np
import pandas as pd


def read_air_passengers_csv():

    path = "data/AirPassengers.csv"
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df.drop(columns=["Month"])

    df["Month"] = np.arange(len(df))

    df = df.rename(columns={"#Passengers": "Passengers"})

    return df


def train_val_test_split(df, cutoffs):

    train = df[
        (df["Date"] >= pd.to_datetime(cutoffs[0], format="%Y"))
        & (df["Date"] < pd.to_datetime(cutoffs[1], format="%Y"))
    ].copy()
    val = df[
        (df["Date"] >= pd.to_datetime(cutoffs[1], format="%Y"))
        & (df["Date"] < pd.to_datetime(cutoffs[2], format="%Y"))
    ].copy()
    test = df[df["Date"] >= pd.to_datetime(cutoffs[2], format="%Y")].copy()

    return train, val, test


def get_data(cutoffs):
    df = read_air_passengers_csv()
    train, val, test = train_val_test_split(df, cutoffs)
    return df, train, val, test


def get_stationary_data(cutoffs):
    df_sta = read_air_passengers_csv()
    df_sta["Passengers"] = df_sta["Passengers"].diff(1).diff(12)
    train_sta, val_sta, test_sta = train_val_test_split(df_sta, cutoffs)
    return df_sta, train_sta, val_sta, test_sta
