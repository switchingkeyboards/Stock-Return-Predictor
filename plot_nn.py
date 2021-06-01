import pandas as pd
import seaborn as sns
from helpers.importance_plot import importance_plot

df = pd.read_csv(r'result.csv')

sns.heatmap(df, annot=True)
