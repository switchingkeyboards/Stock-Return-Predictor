"""
Data Preprocessing
"""

import pandas as pd

df = pd.read_csv("data/data_with_siccd.csv")

df.next_ret = pd.to_numeric(df.next_ret,errors='coerce')
df = df[~np.isnan(df.next_ret)].reset_index(drop=True)

df['divyield'] = df['divyield'].str.replace('%', '').astype('float')

df.columns = [name.lower() for name in df.columns]

ind_train = df[df.year.isin(range(1980,2000))].index # 1980 to 1999
ind_test = df[df.year.isin(range(2000,2020))].index # 2000 to 2019

df_train = df.loc[ind_train,:].copy().reset_index(drop=True)
df_test = df.loc[ind_test,:].copy().reset_index(drop=True)
  
result.to_csv(r'data_with_siccd_cleaned.csv')