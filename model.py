import pickle
import optuna
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, RFECV
os.chdir("D:\Shrey\Python\SPYDER\Datasets")

data = pd.read_csv('cars_sampled.csv')
data

data1 = data.copy(deep = True)
data1


# SETTING THE WORKING RANGE
data1 = data1[(data1['price'] >= 100) & (data1['price'] <= 150000) & (data1['yearOfRegistration'] >= 1950) & (data1['yearOfRegistration'] <= 2021) & (data1['powerPS'] >= 10) & (data1['powerPS'] <= 500)]

# CONVERTING MONTH AND YEAR INTO AGE
data1['age'] = (2021 - data1['yearOfRegistration']) + data1['monthOfRegistration']
data1['age'] = round(data1['age'], 2)

data1 = data1[['yearOfRegistration', 'monthOfRegistration', 'powerPS', 'model', 'kilometer', 'dateCreated', 'lastSeen', 'fuelType', 'vehicleType', 'gearbox', 'notRepairedDamage', 'dateCrawled', 'name', 'seller', 'offerType', 'abtest', 'brand', 'postalCode', 'age', 'price']]


r_list = ['seller', 'abtest', 'offerType', 'name', 'lastSeen', 'monthOfRegistration', 'yearOfRegistration', 'dateCreated', 'dateCrawled', 'postalCode']
data2 = data1.drop(r_list, axis = 1)


data2.drop_duplicates(keep = 'first', inplace = True)

y = np.log(data2['price'])
data2.drop(['price'], axis = 1, inplace = True)


num_cols = [cname for cname in data2.columns if data2[cname].dtype in ['int64', 'float64']]
cat_cols = [cname for cname in data2.columns if data2[cname].dtype == 'object']

num_trans = SimpleImputer(strategy = 'mean')
cat_trans = Pipeline(steps = [('impute', SimpleImputer(strategy = 'most_frequent')), 
                              ('onehotencode', OneHotEncoder(handle_unknown = 'ignore'))])


preproc = ColumnTransformer(transformers = [('cat', cat_trans, cat_cols), 
                                            ('num', num_trans, num_cols)])


dtr_model = DecisionTreeRegressor(random_state = 69, criterion = 'mse')

#PIPELINE FOR PREPROCESSING 
dtr_pipe = Pipeline(steps = [('preproc', preproc), ('model', dtr_model)])

train_x, test_x, train_y, test_y = train_test_split(data2, y, test_size = 0.2, random_state = 69)


#OPTUNA
def objective(trial):
    
    model__max_depth = trial.suggest_int('model__max_depth', 2, 32)
    model__max_leaf_nodes = trial.suggest_int('model__max_leaf_nodes', 50, 500)
    model__max_features = trial.suggest_float('model__max_features', 0.0, 1.0)
    model__min_samples_leaf = trial.suggest_int('model__min_samples_leaf', 1, 50)
    
    params = {'model__max_depth' : model__max_depth, 
              'model__max_leaf_nodes' : model__max_leaf_nodes,
              'model__max_features' : model__max_features,
              'model__min_samples_leaf' : model__min_samples_leaf}
    
    dtr_pipe.set_params(**params)
    
    return np.mean(-1 * cross_val_score(dtr_pipe, train_x, train_y,
                                     cv = 5, n_jobs = -1, scoring = 'neg_mean_squared_error'))
    

dtr_study = optuna.create_study(direction = 'minimize')
dtr_study.optimize(objective, n_trials = 10)

dtr_pipe.set_params(**dtr_study.best_params)
dtr_pipe.fit(train_x, train_y)

pickle.dump(dtr_pipe, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))