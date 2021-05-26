# 1 hidden layer, 3 nodes

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train_nn(csv):
    # csv = input("Path to preprocessed CSV dataset: ")
    df = pd.read_csv(csv)

    ind_train = df[df.year.isin(range(1980, 2000))].index  # 1980 to 1999
    ind_test = df[df.year.isin(range(2000, 2020))].index  # 2000 to 2019

    df_train = df.loc[ind_train, :].copy().reset_index(drop=True)
    df_test = df.loc[ind_test, :].copy().reset_index(drop=True)

    feats_not_to_use = ["permno", "year", "month", "next_ret",
                        "pe_op_dil", "date", "SICCD", "COMNAM", "TSYMBOL", "SECTOR"]
    feats = [feat for feat in df.columns if feat not in feats_not_to_use]
    # feats = ["mmt6", "divyield", "gprof"]

    target = 'next_ret'

    """
  Data Normalization
  """

    def normalize(series):
        return (series - series.mean(axis=0)) / series.std(axis=0)

    mean = df_train[feats].mean(axis=0)
    df_train[feats] = df_train[feats].fillna(mean)

    data_train = df_train[feats].apply(normalize).values

    mean = df_test[feats].mean(axis=0)
    df_test[feats] = df_test[feats].fillna(mean)

    data_test = df_test[feats].apply(normalize).values

    """
  Create TensorFlow Train and Test Datasets
  """

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (data_train, df_train[target].values))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (data_test, df_test[target].values))

    # print("_" * 65)
    # print("First five elements of train dataset:")
    # for feat, targ in train_dataset.take(5):
    #   print ('Features: {}, Target: {}'.format(feat, targ))

    """
  Constructing the Model
  """

    nfeats = len(feats)
    nhid = 3

    def build_model():
        model = keras.Sequential([
            layers.Dense(nhid, activation='tanh', input_shape=[nfeats]),
            layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.SGD(0.005)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mse'])
        return model

    model = build_model()

    # Initialize model weights to random values
    weights = model.weights
    # print("_" * 65)
    # print(weights)

    # theta = [w1, b1, w2, b2]
    np.random.seed(12345)
    w = [np.random.uniform(-0.01, 0.01, size=(nfeats, nhid)),  # w1
         np.random.uniform(-0.01, 0.01, size=nhid),  # b1
         np.random.uniform(-0.01, 0.01, size=(nhid, 1)),  # w2
         np.random.uniform(-0.01, 0.01, size=1)]  # b2
    model.set_weights(w)

    """
  Inspecting the Model
  """

    model.summary()

    """
  Training the Model
  """

    model.fit(train_dataset.batch(1), epochs=1)

    # Trained model weights
    weights = model.weights
    # print("_" * 65)
    # print(weights)

    """
  Make Predictions
  """

    # Larger batch size (100) for faster predictions
    test_predictions = model.predict(test_dataset.batch(100)).flatten()

    """
  Model Evaluation
  """

    def R2(y, y_hat):
        R2 = 1 - np.sum((y - y_hat)**2) / np.sum(y**2)
        return R2

    R2_Val = R2(df_test[target].values, test_predictions)
    # print("_" * 65)
    # print(R2_Val)

    return weights, feats, R2_Val
