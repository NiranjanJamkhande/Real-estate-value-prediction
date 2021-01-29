# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:56:15 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 01:54:44 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 27 02:13:13 2020

@author: Admin
"""


import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Admin\Desktop\Real Estate Valuation Prediction\3_clusters\No_outliers.csv')




dum_df = pd.get_dummies(df, drop_first=True)

dum_df.columns
X = dum_df.drop(columns = 'Value')
y = dum_df['Value']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[['Age', 'nearest_mrt',
   'no_stores']] = scaler.fit_transform(X[['Age', 'nearest_mrt',
                                         'no_stores']])
   
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, 
                                                    random_state=123)

####################################################################################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


#########################K-Fold CV####################################
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, random_state=123)
results = cross_val_score(regressor, X, y, cv=kfold, 
                          scoring='r2')
R2 = results
print(R2)
print("R-Squared: %.2f" % (R2.mean()))
###############################################################################

from sklearn.linear_model import Ridge
clf = Ridge(alpha=2)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))



parameters = dict(alpha=np.linspace(0.01,40))
from sklearn.model_selection import GridSearchCV
clf = Ridge()
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=123)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

cv.best_estimator_
#################################################################################

from sklearn.linear_model import Lasso

clf = Lasso(alpha=2)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


parameters = dict(alpha=np.linspace(0.01,100))
from sklearn.model_selection import GridSearchCV
clf = Lasso()
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2019)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)

# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)
##############################################################################

from sklearn.linear_model import ElasticNet


clf = ElasticNet(alpha=2, l1_ratio=0.6)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))



parameters = dict(alpha=np.linspace(0,20,25),
                  l1_ratio=np.linspace(0,1,25))
from sklearn.model_selection import GridSearchCV
clf = ElasticNet()
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2019)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

cv.best_estimator_
################################################################################


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

parameters = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
# OR
parameters = {'n_neighbors': np.arange(1,16)}
print(parameters)

knn = KNeighborsRegressor()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5 , random_state=42)

cv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='r2',
                  verbose=3)
cv.fit( X , y )
#pd.DataFrame(cv.cv_results_  )
print(cv.best_params_)
print(cv.best_score_)

print(cv.best_estimator_)





knn =KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=7, p=2,
                    weights='uniform')
knn.fit( X_train , y_train )
y_pred = knn.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print( np.sqrt( mean_squared_error(y_test, y_pred)))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

############################  Neural Network

from sklearn.neural_network import MLPRegressor


mlp = MLPRegressor(hidden_layer_sizes=(4,3,2),random_state=2018, verbose=2)
mlp.fit( X_train , y_train )
y_pred = mlp.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


import numpy as np
from sklearn.model_selection import GridSearchCV
lr_range = np.linspace(0.01,0.10,10)
hl_range = [(4,3,2),(3,2),(5,4,3,2)]
lr_method = ['constant','invscaling','adaptive']

parameters = dict(learning_rate=lr_method,
                  hidden_layer_sizes = hl_range,
                  learning_rate_init=lr_range)

mlp = MLPRegressor(random_state=2019)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=42)

mlpGrid = GridSearchCV(mlp, param_grid=parameters, cv=kfold,
                       scoring='r2')
mlpGrid.fit(X, y)

# Best Parameters
print(mlpGrid.best_params_)

print(mlpGrid.best_score_)
