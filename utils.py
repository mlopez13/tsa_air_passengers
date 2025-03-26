import pandas as pd


def read_air_passengers_csv():
    path = 'data/AirPassengers.csv'
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Month'], format='%Y-%m')
    df = df.drop(columns=['Month'])
    df = df.set_index('Date')
    return df

def train_val_test_split(df, cutoffs):

    train = df.copy().reset_index()
    val = df.copy().reset_index()
    test = df.copy().reset_index()

    train = train[
        (train['Date'] >= pd.to_datetime(cutoffs[0], format='%Y')) &
        (train['Date'] < pd.to_datetime(cutoffs[1], format='%Y'))
    ]
    val = val[
        (val['Date'] >= pd.to_datetime(cutoffs[1], format='%Y')) &
        (val['Date'] < pd.to_datetime(cutoffs[2], format='%Y'))
    ]
    test = test[test['Date'] >= pd.to_datetime(cutoffs[2], format='%Y')]

    train = train.set_index('Date')
    val = val.set_index('Date')
    test = test.set_index('Date')

    return train, val, test

def get_data(cutoffs):
    df = read_air_passengers_csv()
    train, val, test = train_val_test_split(df, cutoffs)
    return df, train, val, test

def get_stationary_data(cutoffs):
    s_df = read_air_passengers_csv()
    s_df['#Passengers'] = s_df['#Passengers'].diff(1).diff(12)
    s_train, s_val, s_test = train_val_test_split(s_df, cutoffs)
    return s_df, s_train, s_val, s_test
