# Conversion rate: Exploring sampling on highly imbalanced data
## This repo will cover data sampling on highly imblanced data and their effect on improving performance of selected machine learning algorithm. It will also briefly introduce what metrics should be selected for evaluating a model with originally imbalanced dataset.

(Step-by-step implementation of this problem shown in the "ConversionRate_sampling.ipynb"; standalone coding shown in the "ConversionRate.py".)

## 1. Background: about the problem and data imbalance
  The data revolution has a lot to do with the fact that now we are able to collect all sorts of data about people who buy something on our site as well as people who don't. This gives us a tremendous opportunity to understand what's working well (and potentially scale it even further) and what's not working well (and fix it).
  
  We have data about users who hit our site: whether they converted or not as well as some of their characteristics such as their country, the marketing channel, their age, and the number of pages visited during that session. The goal is to build a model that predicts conversion rate and, based on the model, come up with ideas to improve revenue.
  
  A dataset is IMBALANCED if its classes are not equally represented. Imbalanced datasets are popular in fraud detection, text classification, customer conversion (like the problem to explore here), etc. One way to addressing the imbalanced dataset is by re-sampling original dataset, namely, under-sampling the majority, or over-sampling the minority, or the mixture of both to achiving approximately balanced classes of representation. 
  
  Due to its imbalanced nature, the typical predictive accuracy for evaluating performance of a machine learning model is not appropriate. Take the conversion problem as an example, it has class of converted ~3% and class of non-converted ~97%. Assigning all tests to non-converted class will give accuracy of 97%, which to most of models by machine learning is considered as an excellent predictive outcome. However, the purpose of building the model is to have a fairly good predicton of the minority class and also allow for some mis-classificaton from the majority class. Here receiver operating characteristics (ROC) will be used as a technique for evaluating the performance of classification, with metric ouputs of area under the curve (AUC), precison, recall and F1-score.

## 2. Data sampling and modeling
  Before sampling, the original dataset was cleanned, pre-processed including binning, skewness reduction for numeric features, one-hot encoded (categorical feature transformation), and splitted into train and test datasets. The data resampling (only for train dataset) was achieved by open-source package imblearn. Here we resample the original data by random under-sampling, random over-sampling, synthetic minority over-sampling technique (SMOTE), SMOTEENN (over-sampling followed by under-sampling), and SMOTETomek (over-sampling followed by under-sampling), EasyEnsemble (ensemble sampling) and BalanceCascade (ensemble sampling). The original literatures for these resampling techniques are listed in the end.

  RandomForest (minimizing variance) and XGBoost (minimizing bias) have been chosen for model. Both have been trained with either original dataset (base) or resampled train datasets and with optimized model parameters, and then both are used to predict the ORIGINAL test dataset. 

## 3. Results and Discussion

![roc_conversion](https://user-images.githubusercontent.com/34787111/45992613-b19b0780-c03f-11e8-80d8-04a1d45be8c4.png)

Figure 1. Receiver operating characteristics by RandomForest (left) and XGBoost (right).

  The ROC shown in Figure 1 were generated from both models with different datasets. All classifiers result in good AUC, suggesting two classes from test dataset well-separated. However, according to metric performance, in general, XGBoost is better than RandomForest regardless of using base or resampled datasets. Comparing base with resampled, as shown in table 1, the latter result in drastic improvement in recall but at a cost of more dramatic decreasing in precision, suggesting a tradeoff of predicting less false negatives but more false positives. This is understandable since positive is a class of under-represented in original dataset but manually resampled to be balanced with negative. Thus, this resampling will enlarge features (leading to predictive positives) that in reality may not necessarily is true positive. Due to more imbalanced precision and recall from resampled models, base model gives the best F1 score. However, this does not negate the usefulness of resampling techniques in such problems. If a business model wants to minimize false negatives (do not want to miss any opportunity to convert a customer) more than minimize false positive, resampling is a favored choice. On the contrary, if minimizing false positive is upheld, then base model (say, RandomForest here) is a choice.
  
  Revisiting the metrics in table 1, base models have much better accuracy than resampling models. However, a number of them has better auc than base models, suggesting inappropriate usage of accuracy for evaluating the model performance. The metric confusion-matrix shows the exact values of true negatives (TN), false positive (FP), false negative (FN) and true positive (TP), respectively. Each classifier results in some values for all 4 categories.
  
  To this end, we have successfully predicted conversion rate from test data and evaluated model performance. We then turn to understand what matter(s) the most regarding conversion rate. The feature importances for two models (both using base dataset) were shown in Table 2. "Total_pages_visited" has been unanimously recoginized as the most important feature by both models. This could be explained by that people are interested in this website tending to visit more pages such as for understanding more details, filling required infos, checking service updates, etc. In reality, this is like a leaked feature, since we cannot say whether more visits lead to conversion or the converted visit the websites more frequent thereafter. "new_user" and "country_China" have also shown high importance to both models. As shown in Figure 2, both shows much lower conversion rate compared to their counterpart features. "age" has stand out in XGBoost but not in RandomForest. In addition, source plays insignificant role here.
  
  What if we removed the feature "total_pages_converted"? Are machine learning algorithms able to give reasonable predictions using the remaining features. We repeated the data sampling and modeling processes as described above. 
  
  
  
  
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

