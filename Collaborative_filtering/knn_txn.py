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
df = df.pivot_table(index='TXN_DESC', columns='CUST_NO', aggfunc=len, fill_value=0 )

#%%
def print_txn(n):
    print(df.iloc[n].name)

#%%
print_txn(0)

#%%
from sklearn.neighbors import NearestNeighbors

#%%
model_knn = NearestNeighbors(metric='correlation', algorithm='brute')
model_knn.fit(df)

#%%
def find_related(name):
    distances, indices = model_knn.kneighbors(df.loc[[name]], n_neighbors=6)
    for i, d in zip(indices[0][1:], distances[0][1:]):
        print(df.iloc[i].name, d)
#%%
find_related('台中新光影城')
#%%
find_related('友邦人壽')

#%%
find_related('AGODA')

#%%
find_related('一卡通自動加值')

#%%
find_related('中友百貨公司')
#%%


#%%
