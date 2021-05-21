"""
Data Preprocessing
"""

import pandas as pd
import numpy as np

raw = input("Path to CSV dataset: ")
df = pd.read_csv(raw)

df.next_ret = pd.to_numeric(df.next_ret, errors='coerce')
df = df[~np.isnan(df.next_ret)].reset_index(drop=True)

df['divyield'] = df['divyield'].str.replace('%', '').astype('float')

df.columns = [name.lower() for name in df.columns]

df.to_csv(r'data/data_preprocessed.csv')
