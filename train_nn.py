from models.neural_network import train_nn
import os
import pandas as pd
import numpy as np

cwd = os.getcwd()
dir = os.path.join(cwd, 'data/sectors')

full_paths, names = [], []

for f in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, f)) and f.endswith('.csv'):
        # if (f != "all.csv"):
        #     continue
        full_paths.append(os.path.join(dir, f))
        names.append(f.replace('.csv', ''))


result = []

# Each sector csv file train 5 NN models
for i, path in enumerate(full_paths):
    all_importances, all_R2_Val = train_nn(path)

    for j, importance in enumerate(all_importances):
        importance["sector"] = names[i]
        importance["R2"] = all_R2_Val[j]

    result = [*result, *all_importances]

df = pd.DataFrame(result)
df.to_csv('result.csv', index=False)
