import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data.csv', header=None)
# sns.histplot(data=range(0, 256), weights=df[0].values.tolist())
bin_values = list(range(0, 256))
bin_entries = df[0].values.tolist()
assert len(bin_values) == len(bin_entries)
plt.hist(x=bin_values, weights=bin_entries, bins=256)
plt.show()
