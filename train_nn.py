from models.neural_network import train_nn
import os
import pandas as pd
import numpy as np

cwd = os.getcwd()
dir = os.path.join(cwd, 'data/sectors')

full_paths, names = [], []

for f in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, f)) and f.endswith('.csv'):
        # if (f == "mining.csv"):
        full_paths.append(os.path.join(dir, f))
        names.append(f.replace('.csv', ''))

result = []

for i, path in enumerate(full_paths):
    weights, feats, R2 = train_nn(path)
    result.append(R2)
    # print(weights[0].numpy())

df = pd.DataFrame(data=result, index=names, columns=['R2'])
df.to_csv('result.csv', index=False)