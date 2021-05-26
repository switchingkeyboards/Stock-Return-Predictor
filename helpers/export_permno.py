import pandas as pd
import numpy as np

# raw = input("Path to CSV dataset: ")
# df = pd.read_csv(raw)

df = pd.read_csv(r'data.csv')

# Export txt file for permno to siccd conversion on WRDS
np.savetxt(r'data/permno.txt', df['permno'], fmt='%i')
