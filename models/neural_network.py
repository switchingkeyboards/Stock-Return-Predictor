import eli5
import numpy as np
import pandas as pd
import tensorflow as tf
# from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
from eli5.sklearn import PermutationImportance
from tensorflow import keras
from tensorflow.keras import layers


def train_nn(csv):
    df = pd.read_csv(csv)

    ind_train = df[df.year.isin(range(1980, 2000))].index  # 1980 to 1999
    ind_test = df[df.year.isin(range(2000, 2020))].index  # 2000 to 2019

    df_train = df.loc[ind_train, :].copy().reset_index(drop=True)
    df_test = df.loc[ind_test, :].copy().reset_index(drop=True)

    feats_not_to_use = ["permno", "year", "month", "next_ret",
                        "pe_op_dil", "DATE", "COMNAM", "TICKER", "SICCD",  "SECTOR"]
    feats = [feat for feat in df.columns if feat not in feats_not_to_use]
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

    """
    Constructing the Model
    """

    nfeats = len(feats)

    # Geometric pyramid rule (Masters 1993)
    nhid = [32, 16, 8, 4, 2]

    def build_models():
        models = []
        layers_stack = [layers.Dense(nhid[i], activation="tanh")
                        for i, _ in enumerate(nhid)]

        for i in range(1, 6):
            layers_arr = [layers.Dense(
                nhid[0], activation='tanh', input_shape=[nfeats])]
            for j in range(1, i):
                layers_arr.append(layers_stack[j])
            layers_arr.append(layers.Dense(1))

            model = keras.Sequential(layers_arr)
            optimizer = tf.keras.optimizers.SGD(0.005)
            model.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['mse'])

            models.append(model)

        return models

    NN1, NN2, NN3, NN4, NN5 = build_models()

    # Initialize model weights to random values
    NN1_weights = NN1.weights
    NN2_weights = NN2.weights
    NN3_weights = NN3.weights
    NN4_weights = NN4.weights
    NN5_weights = NN5.weights

    np.random.seed(12345)

    weights_arr = []

    for i in range(1, 6):

        w = [np.random.uniform(-0.01, 0.01, size=(nfeats, nhid[0]))]

        for j in range(0, i):
            w.append(np.random.uniform(-0.01, 0.01, size=nhid[j]))
            if j == i - 1:
                w.append(
                    np.random.uniform(-0.01, 0.01, size=(nhid[j], 1)))
            else:
                w.append(
                    np.random.uniform(-0.01, 0.01, size=(nhid[j], nhid[j + 1])))

        w.append(np.random.uniform(-0.01, 0.01, size=1))
        weights_arr.append(w)

    NN1.set_weights(weights_arr[0])
    NN2.set_weights(weights_arr[1])
    NN3.set_weights(weights_arr[2])
    NN4.set_weights(weights_arr[3])
    NN5.set_weights(weights_arr[4])

    """
    Inspecting the Model
    """

    NN1.summary()
    NN2.summary()
    NN3.summary()
    NN4.summary()
    NN5.summary()

    """
    Training the Model
    """

    NN1.fit(train_dataset.batch(1), epochs=1)
    NN2.fit(train_dataset.batch(1), epochs=1)
    NN3.fit(train_dataset.batch(1), epochs=1)
    NN4.fit(train_dataset.batch(1), epochs=1)
    NN5.fit(train_dataset.batch(1), epochs=1)

    # Trained model weights
    NN1_weights = NN1.weights
    NN2_weights = NN2.weights
    NN3_weights = NN3.weights
    NN4_weights = NN4.weights
    NN5_weights = NN5.weights

    # """
    # Make Predictions
    # """

    # Larger batch size (100) for faster predictions
    NN1_test_predictions = NN1.predict(test_dataset.batch(100)).flatten()
    NN2_test_predictions = NN2.predict(test_dataset.batch(100)).flatten()
    NN3_test_predictions = NN3.predict(test_dataset.batch(100)).flatten()
    NN4_test_predictions = NN4.predict(test_dataset.batch(100)).flatten()
    NN5_test_predictions = NN5.predict(test_dataset.batch(100)).flatten()

    # """
    # Model Evaluation
    # """

    def R2(y, y_hat):
        R2 = 1 - np.sum((y - y_hat)**2) / np.sum(y**2)
        return R2

    NN1_R2_Val = R2(df_test[target].values, NN1_test_predictions)
    NN2_R2_Val = R2(df_test[target].values, NN2_test_predictions)
    NN3_R2_Val = R2(df_test[target].values, NN3_test_predictions)
    NN4_R2_Val = R2(df_test[target].values, NN4_test_predictions)
    NN5_R2_Val = R2(df_test[target].values, NN5_test_predictions)

    all_R2_Val = [NN1_R2_Val, NN2_R2_Val, NN3_R2_Val, NN4_R2_Val, NN5_R2_Val]

    def NN1_score(X, y):
        y_pred = NN1.predict(X)
        return R2(y, y_pred)

    def NN2_score(X, y):
        y_pred = NN2.predict(X)
        return R2(y, y_pred)

    def NN3_score(X, y):
        y_pred = NN3.predict(X)
        return R2(y, y_pred)

    def NN4_score(X, y):
        y_pred = NN4.predict(X)
        return R2(y, y_pred)

    def NN5_score(X, y):
        y_pred = NN5.predict(X)
        return R2(y, y_pred)

    _, NN1_score_decreases = get_score_importances(
        NN1_score, data_test, df_test[target].values)
    _, NN2_score_decreases = get_score_importances(
        NN2_score, data_test, df_test[target].values)
    _, NN3_score_decreases = get_score_importances(
        NN3_score, data_test, df_test[target].values)
    _, NN4_score_decreases = get_score_importances(
        NN4_score, data_test, df_test[target].values)
    _, NN5_score_decreases = get_score_importances(
        NN5_score, data_test, df_test[target].values)

    NN1_feat_imps = np.mean(NN1_score_decreases, axis=0)
    NN2_feat_imps = np.mean(NN2_score_decreases, axis=0)
    NN3_feat_imps = np.mean(NN3_score_decreases, axis=0)
    NN4_feat_imps = np.mean(NN4_score_decreases, axis=0)
    NN5_feat_imps = np.mean(NN5_score_decreases, axis=0)

    all_importances = []

    for feat_imps in [NN1_feat_imps, NN2_feat_imps, NN3_feat_imps, NN4_feat_imps, NN5_feat_imps]:
        importances = {}

        for index, feat_imp in enumerate(feat_imps):
            importances[feats[index]] = feat_imp

        all_importances.append(importances)

    return all_importances, all_R2_Val
