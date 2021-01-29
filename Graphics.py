# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:50:35 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:50:02 2020

@author: Admin
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\Admin\Desktop\Real Estate Valuation Prediction\3_clusters\Updated.csv')
df = df.rename(columns = {'X2 house age': 'Age',
                          'X3 distance to the nearest MRT station': 'nearest_mrt',
                          'X4 number of convenience stores': 'no_stores',
                          'Y house price of unit area': 'Value'})



df.columns
  ##     'Age', 'nearest_mrt', 'no_stores', 'Value', 'Area'], dtype='object'


df.corr()
## no need to remove any column


# 1)
df['Age'].describe()
sns.boxplot(df['Age'])
# no outliers



# 2)
df['nearest_mrt'].describe()
# huge difference betwuun Q3 & max value,  this is an indication of outlier, then also confirm by boxplot
sns.boxplot(df['nearest_mrt'])
# outliers detected.
# removing outliers can lead to loose the information, instead we do appropriate imputation.


df['nearest_mrt'] = np.where(df['nearest_mrt']>2500,2000, df['nearest_mrt']) 
sns.boxplot(df['nearest_mrt'])



# 3) 
df['Area'].value_counts()
sns.countplot('Area',data=df)
plt.show()


#######################################################################################

df.to_csv(r'C:\Users\Admin\Desktop\Real Estate Valuation Prediction\3_clusters\No_outliers.csv', index = False)
















