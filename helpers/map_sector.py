import pandas as pd
import numpy as np

raw = input("Path to CSV dataset: ") 
df = pd.read_csv(raw)
df_siccd = pd.read_csv(r'data/siccd.csv')


def get_sector(x):
    digits = str(x)[:2]
    try:
        digits = int(digits)
    except (TypeError, ValueError):
        return "Unclassified"
    if digits > 0 and digits < 10:
        return "Agriculture, Forestry, Fishing"
    elif digits > 9 and digits < 15:
        return "Mining"
    elif digits > 14 and digits < 18:
        return "Construction"
    elif digits > 19 and digits < 40:
        return "Manufacturing"
    elif digits > 39 and digits < 50:
        return "Transportation & Public Utilities"
    elif digits > 49 and digits < 52:
        return "Wholesale Trade"
    elif digits > 51 and digits < 60:
        return "Retail Trade"
    elif digits > 59 and digits < 68:
        return "Finance, Insurance, Real Estate"
    elif digits > 69 and digits < 90:
        return "Services"
    elif digits > 90 and digits < 100:
        return "Public Administration"


df_siccd['SECTOR'] = [get_sector(siccd) for siccd in df_siccd['SICCD']]
result = pd.concat([df, df_siccd], axis=1, join='inner')

print(result.groupby(by="SECTOR")['SECTOR'].agg('count'))
"""
SECTOR
Finance, Insurance, Real Estate       13980
Manufacturing                        113725
Mining                                 6468
Public Administration                  8816
Retail Trade                           5314
Services                              50478
Transportation & Public Utilities     24033
Unclassified                           1237
Wholesale Trade                        3044
"""

result.to_csv(r'data/data_with_siccd.csv')
