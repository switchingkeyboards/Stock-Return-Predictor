import pandas as pd
import numpy as np

raw = input("Path to CSV dataset: ") 
df = pd.read_csv(raw)

# Export txt file for permno to siccd conversion on WRDS
np.savetxt(r'data/permno.txt', df['PERMNO'], fmt='%i')