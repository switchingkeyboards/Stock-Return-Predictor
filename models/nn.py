import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

raw = input("Path to CSV dataset: ") 
df = pd.read_csv(raw)
  
feats_not_to_use=["permno","year","month","next_ret","pe_op_dil"]
feats_to_use = [feat for feat in df.columns if feat not in feats_not_to_use]
target = 'next_ret'
feats = ["mmt6","divyield","gprof"]

"""
Data Normalization
"""

def normalize(series):
  return (series-series.mean(axis=0))/series.std(axis=0)

mean = df_train[feats].mean(axis=0)
df_train[feats] = df_train[feats].fillna(mean)      

data_train = df_train[feats].apply(normalize).values

mean = df_test[feats].mean(axis=0)
df_test[feats] = df_test[feats].fillna(mean)      

data_test = df_test[feats].apply(normalize).values

"""
Create TensorFlow Train and Test Datasets
"""

train_dataset = tf.data.Dataset.from_tensor_slices((data_train, df_train[target].values))
test_dataset = tf.data.Dataset.from_tensor_slices((data_test, df_test[target].values))

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

weights = model.weights
print(weights)

np.random.seed(12345)
w = [np.random.uniform(-0.01,0.01,size = (nfeats,nhid)),np.random.uniform(-0.01,0.01,size = nhid),np.random.uniform(-0.01,0.01,size = (nhid,1)),np.random.uniform(-0.01,0.01,size = 1)]
model.set_weights(w)

"""
Inspecting the Model
"""

model.summary()

"""
Training the Model
"""

model.fit(train_dataset.batch(1), epochs=1)

weights = model.weights
print(weights)

"""
Make Predictions
"""

test_predictions = model.predict(test_dataset.batch(100)).flatten()

"""
Model Evaluation
"""

def R2(y, y_hat):
  R2 = 1 - np.sum((y - y_hat)**2) / np.sum(y**2)
  return R2

R2_Val = R2(df_test[target].values,test_predictions)
print(R2_Val)