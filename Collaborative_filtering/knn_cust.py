#%%
import pandas as pd
import os.path
import numpy as np
#%%
DATA_DIR='Data'
sr_1_fn = os.path.join(DATA_DIR, 'sr_1.csv')
with open(sr_1_fn, encoding='cp950', errors='replace') as f:
    df = pd.read_csv(f, nrows=1000000, usecols=['CUST_NO', 'TXN_DESC'])

#%%
df.head()
#%%
TXN = df.TXN_DESC.str.extract(r'([^\s０－９（５１２）]+)')

#%%
df.TXN_DESC.nunique()
#%%
df['TXN_DESC'] = TXN

#%%
df.TXN_DESC.nunique()

#%%
txn_counts = df.TXN_DESC.value_counts()
txn_counts.quantile(np.arange(.5,1,.01))
#%%
good_txn = txn_counts[txn_counts>50]
#df = df.loc[df.TXN_DESC.isin(good_txn.index)]
df.drop(df.loc[~df.TXN_DESC.isin(good_txn.index)].index, inplace=True)
df.TXN_DESC.nunique()
#%%
cust_counts = df.CUST_NO.value_counts()
cust_counts.quantile(np.arange(0,1,.01))

#%%
good_cust = cust_counts[cust_counts>15]
#df = df.loc[df.CUST_NO.isin(good_cust.index)]
df.drop(df.loc[~df.CUST_NO.isin(good_cust.index)].index, inplace=True)
#%%
df.TXN_DESC.nunique()

#%%
df.CUST_NO.nunique()

#%%
df = df.pivot_table(index='CUST_NO', columns='TXN_DESC', aggfunc=len, fill_value=0 )

#%%
def print_cust(n):
    print(df.iloc[n][df.iloc[n] > 0])

#%%
print_cust(0)

#%%
from sklearn.neighbors import NearestNeighbors

#%%
model_knn = NearestNeighbors(metric='correlation', algorithm='brute')
model_knn.fit(df)

#%%
distances, indices = model_knn.kneighbors(df.iloc[1:2], n_neighbors=6)

#%%
distances

#%%
for i in indices[0]:
    print_cust(i)
    print()

#%%
print_cust(621)
#%%
indices.shape

#%%
