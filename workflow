import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as sc
from IPython.display import display

df = pd.read_csv('https://raw.githubusercontent.com/leksmart/NGHRM/main/SNIP1_1-82.csv')
print(df.head(5))

plt.show(plt.plot(df.iloc[:, [0]], df.iloc[:, 1:]))
#plt.show(plt.plot(df.iloc[:, [0]], df.iloc[:, 1:15]))

df_melt = df.loc[(df.iloc[:, 0] > 73) & (df.iloc[:, 0] < 88)]
df_data = df_melt.iloc[:, 1:]
plt.plot(df_melt.iloc[:, [0]], df_data)
plt.show()

df_norm = (df_data - df_data.min()) / (df_data.max()-df_data.min())*100
plt.plot(df_melt.iloc[:, [0]], df_norm)
plt.show()

dfdif = df_norm.sub(df_norm['A1'], axis=0)



plt.show(plt.plot(df_melt.iloc[:, [0]], (df_norm.sub(df_norm['B1'], axis=0))))
print("B1")


mat = dfdif.T.values
hc = sc.KMeans(n_clusters=3)
hc.fit(mat)

labels = hc.labels_
results = pd.DataFrame([dfdif.T.index, labels])
display(results.loc[:0, results.iloc[1] == 0])
display(results.loc[:0, results.iloc[1] == 1])
display(results.loc[:0, results.iloc[1] == 2])



import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(df_melt.iloc[:, [0]],df_melt.iloc[:, [4]], c=colors, alpha=0.5)
plt.show()
