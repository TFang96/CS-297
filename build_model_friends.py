from packaging import version
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_twitter_data():
    file = Path("Synthetic__Bootstrap__preview.csv")
    return pd.read_csv(file)

def print_model(model):
    return list(zip(model.feature_names_in_.tolist(), model.coef_.tolist()[0])), model.intercept_[0]

def main():
    twitter_data = load_twitter_data()
    ## dealing with na's
    nan_rows = twitter_data.isna().any(axis=1)
    twitter_data.loc[nan_rows]
    twitter_data.dropna(inplace=True)
    twitter_data.loc[nan_rows]
    twitter_data.reset_index(inplace=True)
    twitter_data.drop('index', axis=1, inplace=True)
    twitter_data.plot(kind="scatter", x="No Of Friends", y = "Fake Or Not Category", grid=True, alpha=0.2)
    plt.show()

    train_set, test_set = train_test_split(twitter_data, test_size=0.2, random_state=1237)
    len(train_set), len(test_set)

    label = 'Fake Or Not Category'
    train_label = train_set[[label]]
    features = ['No Of Friends']
    train_features = train_set[features]

    model = LinearRegression()
    model.fit(train_features, train_label)

    print_model(model)

    joblib.dump(value = model, filename='model_friends.pkl')

main()