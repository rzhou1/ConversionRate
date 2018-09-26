# Conversion rate: Exploring sampling on highly imbalanced data
## This repo will cover data sampling on highly imblanced data and their effect on improving performance of selected machine learning algorithm. It will also briefly introduce what metrics should be selected for evaluating a model with originally imbalanced dataset.

## 1. Background: about the problem and data imbalance
  The data revolution has a lot to do with the fact that now we are able to collect all sorts of data about people who buy something on our site as well as people who don't. This gives us a tremendous opportunity to understand what's working well (and potentially scale it even further) and what's not working well (and fix it).
  
  We have data about users who hit our site: whether they converted or not as well as some of their characteristics such as their country, the marketing channel, their age, and the number of pages visited during that session. The goal is to build a model that predicts conversion rate and, based on the model, come up with ideas to improve revenue.
  
  A dataset is IMBALANCED if its classes are not equally represented. Imbalanced datasets are popular in fraud detection, text classification, customer conversion (like the problem to explore here), etc. One way to addressing the imbalanced dataset is by re-sampling original dataset, namely, under-sampling the majority, or over-sampling the minority, or the mixture of both to achiving approximately balanced classes of representation. 
  
  Due to its imbalanced nature, the typical predictive accuracy for evaluating performance of a machine learning model is not appropriate. Take the conversion problem as an example, it has class of converted ~3% and class of non-converted ~97%. Assigning all tests to non-converted class will give accuracy of 97%, which to most of models by machine learning is considered as an excellent predictive outcome. However, the purpose of building the model is to have a fairly good predicton of the minority class and also allow for some mis-classificaton from the majority class. Here receiver operating characteristics (ROC) will be used as a technique for evaluating the performance of classification, with metric ouputs of area under the curve (AUC), precison, recall and F1-score.

## 2. Data resampling and modeling
  Before re-sampling, the original dataset was cleanned, pre-processed including binning, skewness reduction for numeric features, one-hot encoded (categorical feature transformation), and splitted into train and test datasets. The data resampling (only for train dataset) was achieved by open-source package imblearn. Here we resample the original data by random under-sampling, random over-sampling, synthetic minority over-sampling technique (SMOTE), SMOTEENN (over-sampling followed by under-sampling), and SMOTETomek (over-sampling followed by under-sampling), EasyEnsemble (ensemble sampling) and BalanceCascade (ensemble sampling). The original literatures for these resampling techniques are listed in the end.

  RandomForest (minimizing variance) and XGBoost (minimizing bias) have been chosen for model. Both have been trained with either original dataset or resampled train datasets and with optimized model parameters, and then both are used to predict the ORIGINAL test dataset. 

## 3. Results and Discussion

![roc_conversion](https://user-images.githubusercontent.com/34787111/45992613-b19b0780-c03f-11e8-80d8-04a1d45be8c4.png)
Figure 1. Receiver operating characteristics by RandomForest (left) and XGBoost (right).


## 4. Summary

## 5. Reference for data resampling

CNN - "Addressing the Curse of Imbalanced Training Sets: One-Sided Selection", by Kubat et al., 1997.
One-Sided Selection - "Addressing the Curse of Imbalanced Training Sets: One-Sided Selection", by Kubat et al., 1997.
NCL - "Improving identification of difficult small classes by balancing class distribution", by Laurikkala et al., 2001.
SMOTE - "SMOTE: synthetic minority over-sampling technique", by Chawla et al., 2002.
Borderline SMOTE - "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning", by Han et al., 2005
SVM_SMOTE - "Borderline Over-sampling for Imbalanced Data Classification", Nguyen et al., 2011.
SMOTE + Tomek - "Balancing training data for automated annotation of keywords: a case study", Batista et al., 2003.
SMOTE + ENN - "A study of the behavior of several methods for balancing machine learning training data", Batista et al., 2004.
EasyEnsemble & BalanceCascade - "Exploratory Understanding for Class-Imbalance Learning", by Liu et al., 2009.

