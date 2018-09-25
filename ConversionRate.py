#Website conversion rate
#1. The purpose is to understand what feature(s) contribute to conversion and how to improve conversion rate.
#2. This solution explore various data sampling techniques upon the highly imbalanced dataset, all of which have been
#feeded to machine learning algorithms for training. The trained models were used for predicting the raw test data, which
#has been evaluated by various metrics.
#3. The results were summarized with both model selection and business insight suggestions.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalanceCascade, EasyEnsemble
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, roc_curve, confusion_matrix, precision_recall_fscore_support
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#import data
data = pd.read_csv('conversion_data.csv')

#Data preprocessing#
#According to data describe, very clean data, no missing values, 'age' has abnormal large value, 'converted' mean is 0.03.

#1. Double-check missing data
data.isnull().sum() #none, no data impute required.

#2. Clean age data
data=data[data.age<80] #2 data points with age over 110, not very likely, treated as outliers and deleted.

#3. Check magnitude of data imbalance
pd.value_counts(data['converted'], sort=True).sort_index() #0: 305836; 1: 10198 (~3%).
#(Imbalanced, will try sampling techniques to get it balanced.)

#4. Check skewness of numeric features and reduce their skewness
numeric_features = ['age', 'total_pages_visited']
skewness = data[numeric_features].apply(lambda x: skew(x.dropna()))  #age: 0.4903, total_pages_visited: 1.44289
#(not highly skewed, but can be improved by log)

data['age'] = np.log1p(data['age']) #reduced to -0.0529 after transformation.
data['total_pages_visited'] = np.log1p(data['total_pages_visited']) #to reduce skewness to -0.015
#(conversion rate reaches unity when total_pages_visited >=21. However, no prediction improvement by clipping 'total_pages_visited [1,21])

#5. Transform categorical data
data = pd.get_dummies(data)

#6. Re-organize columns
cols = data.columns.tolist()
cols = cols[:3] + cols[4:] + cols[3:4]
data = data[cols]


#Part 1. Model with all features
#train and test data split

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

test_size = 0.3 #30% data to be assigned as test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

##Data sampling (all features): explore data sampling techniques for improving model prediction

def Sampling(sampler, X, y):           #define a data sampling function
    x_resampled, y_resampled = sampler.fit_sample(X,y)
    return sampler.__class__.__name__, pd.DataFrame(x_resampled), pd.DataFrame(y_resampled)

data_sampling = []
data_sampling.append(('base', x_train, y_train))
data_sampling.append(Sampling(RandomUnderSampler(), x_train, y_train)) #undersampling
data_sampling.append(Sampling(RandomOverSampler(), x_train, y_train)) #oversampling
data_sampling.append(Sampling(SMOTE(n_jobs=-1), x_train, y_train)) #oversampling
data_sampling.append(Sampling(SMOTEENN(n_jobs=-1), x_train, y_train)) #oversampling followed by undersampling
data_sampling.append(Sampling(SMOTETomek(n_jobs=-1), x_train, y_train)) #oversampling followed by undersampling

ee = EasyEnsemble(random_state=0)  #ensemble sampling
ee.fit(x_train, y_train)
x_resampled_ee, y_resampled_ee = ee.sample(x_train, y_train)
x_resampled_ee, y_resampled_ee = x_resampled_ee.reshape(144960,10), y_resampled_ee.reshape(144960)
data_sampling.append(('EasyEnsemble', x_resampled_ee, y_resampled_ee))

bc=BalanceCascade(estimator=LogisticRegression(), n_max_subset=10, random_state=0) #ensemble sampling
bc.fit(x_train, y_train)
x_resampled_bc, y_resampled_bc = bc.sample(x_train, y_train)
x_resampled_bc, y_resampled_bc = x_resampled_bc.reshape(144960,10), y_resampled_bc.reshape(144960)
data_sampling.append(('BalanceCascade', x_resampled_bc, y_resampled_bc))

#Check data balance after sampling
data_balance = []
for t, X, y in data_sampling:
    y_count = np.count_nonzero(y)
    c = y_count * 1.0 / len(y)
    data_balance.append((t,c))
#Training data after balance:
#[('base', 0.03276332026959222),('RandomUnderSampler', 0.5),('RandomOverSampler', 0.5),('SMOTE', 0.5),
#('SMOTEENN', 0.48939352158931976),('SMOTETomek', 0.5),('EasyEnsemble', 0.5),('BalanceCascade', 0.5)]

# Modeling
# Random Forest

rf_scores = []
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(figsize=(8, 8))
ax[0].plot([0, 1], [0, 1], linestyle='--', color='k')
ax[0].set_title('Receiver Operating Characteristics: RandomForest')
ax[0].set_xlabel('False positive rate')
ax[0].set_ylabel('True positive rate')
for s, X, y in data_sampling:
    rfc = RandomForestClassifier(max_features='log2', max_depth=3, n_estimators=1000, min_samples_split=2,
                                 oob_score=False, n_jobs=4, random_state=0)
    #     params = {'max_features': ['log2', 'sqrt'], 'max_depth': [3,5], 'n_estimators': [100,1000],
    #               'min_samples_split': [2,3]}
    #     rfc = GridSearchCV(rfc, params, scoring='accuracy', cv=3, n_jobs=4)
    X = pd.DataFrame(X)
    rfc.fit(X.values, y)
    y_hat_rfc = rfc.predict(x_test.values)  # for test prediction, feed with raw data, not resampled data.
    y_hat_rfc_proba = rfc.predict_proba(x_test.values)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_hat_rfc_proba)
    rf_scores.append((rfc.__class__.__name__,
                      s,
                      precision_score(y_test, y_hat_rfc),
                      recall_score(y_test, y_hat_rfc),
                      f1_score(y_test, y_hat_rfc),
                      accuracy_score(y_test, y_hat_rfc),
                      auc(fpr, tpr),
                      confusion_matrix(y_test, y_hat_rfc)))
    ax[0].plot(fpr, tpr, label=s + ': AUC= %.4f' % auc(fpr, tpr))
    ax[0].legend(loc='lower right')

rf_scores = pd.DataFrame(rf_scores, columns=['model', 'sampling', 'precision', 'recall', 'f1', 'accuracy', 'auc',
                                             'confusion_matrix'])

#XGBoost
xgb_scores = []
ax[1].plot([0, 1], [0, 1], linestyle='--', color='k')
ax[1].set_title('Receiver Operating Characteristics: XGBoost')
ax[1].set_xlabel('False positive rate')
ax[1].set_ylabel('True positive rate')
for s, X, y in data_sampling:
    xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                             booster='gbtree', n_jobs=4, random_state=0)
    #     params = {'max_depth': [3,5], 'learning_rate': [0.05,0.1], 'n_estimators': [100,500], 'objective': ['binary:logistic'],
    #              'booster':['gbtree']}
    #     xgbc = GridSearchCV(xgbc, params, n_jobs=4, verbose=1)
    X = pd.DataFrame(X)
    xgbc.fit(X.values, y)
    y_hat_xgb = xgbc.predict(x_test.values)
    y_hat_xgb_proba = xgbc.predict_proba(x_test.values)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_hat_xgb_proba)
    xgb_scores.append((xgbc.__class__.__name__,
                       s,
                       precision_score(y_test, y_hat_xgb),
                       recall_score(y_test, y_hat_xgb),
                       f1_score(y_test, y_hat_xgb),
                       accuracy_score(y_test, y_hat_xgb),
                       auc(fpr, tpr),
                       confusion_matrix(y_test, y_hat_xgb)))
    ax[1].plot(fpr, tpr, label=s + ': AUC= %.4f' % auc(fpr, tpr))
    ax[1].legend(loc='lower right')
xgb_scores = pd.DataFrame(xgb_scores, columns=['model', 'sampling', 'precision', 'recall', 'f1', 'accuracy', 'auc',
                                               'confusion_matrix'])

scores = pd.concat([rf_scores, xgb_scores], axis=0)
#scores

feature_importance_rf = rfc.feature_importances_/rfc.feature_importances_.max()
feature_importance_xgb = xgbc.feature_importances_/xgbc.feature_importances_.max()
feature_importance = pd.DataFrame({'feature': x_train.columns.tolist(),'RandomForest': feature_importance_rf,
                                   'XGBoost': feature_importance_xgb}).sort_values('XGBoost', ascending=False)
#feature_importance

####Part 2. Modeling without the feature 'total_pages_visited'.####

#data sampling: drop feature 'total_pages_visited'
Xt = data.iloc[:, :-1].drop('total_pages_visited', axis=1)
yt = data.iloc[:, -1]
xt_train, xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=test_size, random_state=1)

data_sampling_t = []
data_sampling_t.append(('base', xt_train, yt_train))
data_sampling_t.append(Sampling(RandomUnderSampler(), xt_train, yt_train)) #undersampling
data_sampling_t.append(Sampling(RandomOverSampler(), xt_train, yt_train)) #oversampling
data_sampling_t.append(Sampling(SMOTE(n_jobs=-1), xt_train, yt_train)) #oversampling
data_sampling_t.append(Sampling(SMOTEENN(n_jobs=-1), xt_train, yt_train)) #oversampling followed by undersampling
data_sampling_t.append(Sampling(SMOTETomek(n_jobs=-1), xt_train, yt_train)) #oversampling followed by undersampling

ee = EasyEnsemble(random_state=0)  #ensemble sampling
ee.fit(xt_train, yt_train)
xt_resampled_ee, yt_resampled_ee = ee.sample(xt_train, yt_train)
xt_resampled_ee, yt_resampled_ee = xt_resampled_ee.reshape(144960,9), yt_resampled_ee.reshape(144960)
data_sampling_t.append(('EasyEnsemble', xt_resampled_ee, yt_resampled_ee))

bc=BalanceCascade(estimator=LogisticRegression(), n_max_subset=10, random_state=0) #ensemble sampling
bc.fit(xt_train, yt_train)
xt_resampled_bc, yt_resampled_bc = bc.sample(xt_train, yt_train)
xt_resampled_bc, yt_resampled_bc = xt_resampled_bc.reshape(144960,9), yt_resampled_bc.reshape(144960)
data_sampling_t.append(('BalanceCascade', xt_resampled_bc, yt_resampled_bc))

#modeling

#Random Forest
rf_scores_t = []
for s, X, y in data_sampling_t:
    rfc = RandomForestClassifier(max_features='log2', max_depth=3, n_estimators=1000, min_samples_split=2, oob_score=False, n_jobs=4, random_state=1)
#     params = {'max_features': ['log2', 'sqrt'], 'max_depth': [3,5], 'n_estimators': [100,1000],
#               'min_samples_split': [2,3]}
#     rfc = GridSearchCV(rfc, params, scoring='accuracy', cv=3, n_jobs=4)
    X = pd.DataFrame(X)
    rfc.fit(X.values, y)
    y_hat_rfc = rfc.predict(xt_test.values)
    y_hat_rfc_proba = rfc.predict_proba(xt_test.values)[:, 1]
    fpr, tpr, threshold = roc_curve(yt_test, y_hat_rfc_proba)
    rf_scores_t.append((rfc.__class__.__name__,
                     s,
                     precision_score(yt_test, y_hat_rfc),
                     recall_score(yt_test, y_hat_rfc),
                     f1_score(yt_test, y_hat_rfc),
                     accuracy_score(yt_test, y_hat_rfc),
                     auc(fpr, tpr),
                     confusion_matrix(y_test, y_hat_rfc)))
rf_scores_t = pd.DataFrame(rf_scores_t, columns=['model', 'sampling', 'precision', 'recall', 'f1', 'accuracy', 'auc', 'confusion_matrix'])

#XGBoost
xgb_scores_t = []
for s, X, y in data_sampling_t:
    xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                            booster='gbtree', n_jobs=4)
    X = pd.DataFrame(X)
    xgbc.fit(X.values,y)
    y_hat_xgb = xgbc.predict(xt_test.values)
    y_hat_xgb_proba = xgbc.predict_proba(xt_test.values)[:,1]
    fpr, tpr, thresholds = roc_curve(yt_test, y_hat_xgb_proba)
    xgb_scores_t.append((xgbc.__class__.__name__,
                      s,
                      precision_score(yt_test, y_hat_xgb),
                      recall_score(yt_test, y_hat_xgb),
                      f1_score(yt_test, y_hat_xgb),
                      accuracy_score(yt_test, y_hat_xgb),
                      auc(fpr, tpr),
                      confusion_matrix(yt_test, y_hat_xgb)))

xgb_scores_t = pd.DataFrame(xgb_scores_t, columns=['model', 'sampling', 'precision', 'recall', 'f1', 'accuracy', 'auc', 'confusion_matrix'])

##Output scores and feature importance

scores_t = pd.concat([rf_scores_t, xgb_scores_t], axis=0)

feature_importance_rf = rfc.feature_importances_/rfc.feature_importances_.max()
feature_importance_xgb = xgbc.feature_importances_/xgbc.feature_importances_.max()
feature_importance_t = pd.DataFrame({'feature': xt_train.columns.tolist(), 'RandomForest': feature_importance_rf,
                                   'XGBoost': feature_importance_xgb}).sort_values('XGBoost', ascending=False)

# scores_t
# feature_importance_t