# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:29:26 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:59:37 2020

@author: Admin
"""
import numpy as np
import pandas as pd

df = pd.read_excel(r"C:\Users\Admin\Desktop\Real Estate Valuation Prediction\original.xlsx", index_col=0)

## checking missing values
null_col = df.columns[df.isnull().any()]
null_col
df[null_col].isnull().sum()

df.info()

df_cls = df.iloc[:,4:6]

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
df_cls_scaled = scaler.fit_transform(df_cls)

df_cls_scaled = pd.DataFrame(df_cls_scaled,
                          columns=df_cls.columns,
                          index=df_cls.index)

# Import KMeans
from sklearn.cluster import KMeans

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2019)
    model.fit(df_cls_scaled)
    Inertia.append(model.inertia_)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

##from the graph, we shall do four clusters of data.

# Create a KMeans instance with clusters: model
model = KMeans(n_clusters=3, random_state=123, verbose=2)
model.fit(df_cls_scaled)
model.cluster_centers_
# Determine the cluster labels of new_points: labels
labels = model.predict(df_cls_scaled)
print(labels)


# Variation
print(model.inertia_)


clusterID = pd.DataFrame({'ClustID':labels})
clusterID['ClustID'].value_counts()
updated_df = pd.concat([df.reset_index(drop=True),clusterID],axis=1)

updated_df.info()
updated_df['ClustID'] = updated_df['ClustID'].astype(str)
updated_df.info()




updated_df = updated_df.drop(columns = ['X1 transaction date',
                                        'X5 latitude',
                                        'X6 longitude'])


updated_df = updated_df.rename(columns = {'ClustID': 'Area'})


updated_df['Area'] = updated_df['Area'].replace({'0': 'Area_1',
                                        '1': 'Area_2',
                                        '2': 'Area_3' })
                                        
    
updated_df.info()



### Now updated_df is the data which we can use for further analysis.

updated_df.to_csv(r'C:\Users\Admin\Desktop\Real Estate Valuation Prediction\3_clusters\Updated.csv', index = False)
