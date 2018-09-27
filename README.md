# Conversion rate: Exploring resampling on highly imbalanced data
## This repo will cover data resampling on highly imblanced data and their effect on improving performance of selected machine learning algorithm. It will also briefly introduce what metrics should be selected for evaluating a model with originally imbalanced dataset. The bussiness recommendations to improving converstion rate have been provided based on results from machine learning models.

(Step-by-step implementation of this problem shown in the "ConversionRate_sampling.ipynb"; standalone coding shown in the "ConversionRate.py".)

## 1. Background: about the problem and data imbalance
  The data revolution has a lot to do with the fact that now we are able to collect all sorts of data about people who buy something on our site as well as people who don't. This gives us a tremendous opportunity to understand what's working well (and potentially scale it even further) and what's not working well (and fix it).
  
  We have data about users who hit our site: whether they converted or not as well as some of their characteristics such as their country, the marketing channel, their age, and the number of pages visited during that session. The goal is to build a model that predicts conversion rate and, based on the model, come up with ideas to improve revenue.
  
  A dataset is IMBALANCED if its classes are not equally represented. Imbalanced datasets are popular in fraud detection, text classification, customer conversion (like the problem to explore here), etc. One way to addressing the imbalanced dataset is by re-sampling original dataset, namely, under-sampling the majority, or over-sampling the minority, or the mixture of both to achiving approximately balanced classes of representation. 
  
  Due to its imbalanced nature, the typical predictive accuracy for evaluating performance of a machine learning model is not appropriate. Take the conversion problem as an example, it has class of converted ~3% and class of non-converted ~97%. Assigning all tests to non-converted class will give accuracy of 97%, which to most of models by machine learning is considered as an excellent predictive outcome. However, the purpose of building the model is to have a fairly good predicton of the minority class and also allow for some mis-classificaton from the majority class. Here receiver operating characteristics (ROC) will be used as a technique for evaluating the performance of classification, with metric ouputs of area under the curve (AUC), precison, recall and F1-score.

## 2. Data resampling and modeling
  Before resampling, the original dataset was cleanned, pre-processed including binning, skewness reduction for numeric features, one-hot encoded (categorical feature transformation), and splitted into train and test datasets. The data resampling (only for train dataset) was achieved by open-source package imblearn. Here we resample the original data by random under-sampling, random over-sampling, synthetic minority over-sampling technique (SMOTE), SMOTEENN (over-sampling followed by under-sampling), and SMOTETomek (over-sampling followed by under-sampling), EasyEnsemble (ensemble sampling) and BalanceCascade (ensemble sampling). The original literatures for these resampling techniques are listed in the end.

  RandomForest (minimizing variance) and XGBoost (minimizing bias) have been chosen for model. Both have been trained with either original dataset (base) or resampled train datasets and with optimized model parameters, and then both are used to predict the ORIGINAL test dataset. 

## 3. Results and Discussion

![roc_conversion](https://user-images.githubusercontent.com/34787111/45992613-b19b0780-c03f-11e8-80d8-04a1d45be8c4.png)

Figure 1. Receiver operating characteristics by RandomForest (left) and XGBoost (right).

  The ROC shown in Figure 1 were generated from both models with different datasets. All classifiers result in good AUC, suggesting two classes from test dataset well-separated. However, according to metric performance, in general, XGBoost is better than RandomForest regardless of using base or resampled datasets. Comparing base with resampled, as shown in table 1, the latter result in drastic improvement in recall but at a cost of more dramatic decreasing in precision, suggesting a tradeoff of predicting less false negatives but more false positives. This is understandable since positive is a class of under-represented in original dataset but manually resampled to be balanced with negative. Thus, this resampling will enlarge features (leading to predictive positives) that in reality may not necessarily is true positive. Due to more imbalanced precision and recall from resampled models, base model gives the best F1 score. However, this does not negate the usefulness of resampling techniques in such problems. If a business model wants to minimize false negatives (do not want to miss any opportunity to convert a customer) more than minimize false positive, resampling is a favored choice. On the contrary, if minimizing false positive is upheld, then base model (say, RandomForest here) is a choice.

|no.|model|resampling|precision|recall|f1|accuracy|auc|confusion_matrix|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|1|RF|base|0.936237|0.502712|0.654169|0.983462|0.965088|[[91760, 101], [1467, 1483]]|
|2|RF|RandomUnderSampler|0.355634|0.924407|0.513656|0.945534|0.978916|[[86920, 4941], [223, 2727]]|
|3|RF|RandomOverSampler|	0.352812|	0.927119|	0.511119|	0.944817|	0.979179|	[[86844, 5017], [215, 2735]]|
|4|RF|SMOTE|0.355225|	0.924068|	0.513178|	0.945449|	0.978526|	[[86913, 4948], [224, 2726]]|
|5|RF|SMOTEENN|	0.367605|	0.916271|	0.524702|	0.94835|0.977473|	[[87211, 4650], [247, 2703]]|
|6|RF|SMOTETomek|	0.352365|	0.92678|	0.510599|	0.944722|	0.978879|	[[86836, 5025], [216, 2734]]|
|7|RF|EasyEnsemble|	0.352941|	0.927458|	0.511306|	0.944838|	0.978711|	[[86845, 5016], [214, 2736]]|
|8|RF|BalanceCascade|	0.365946|	0.917966|	0.523285|	0.94796|	0.978506|	[[87169, 4692], [242, 2708]]|
|9|XGB|base|	0.842934|	0.689492|	0.758531|	0.986341|	0.985684|	[[91482, 379], [916, 2034]]|
|10|XGB|RandomUnderSampler|	0.351393|	0.932203|	0.510393|	0.944352|	0.985669|	[[86785, 5076], [200, 2750]]|
|11|XGB|RandomOverSampler|	0.340528|	0.935932|	0.499367|	0.94161|0.985703|	[[86514, 5347], [189, 2761]]|
|12|XGB|SMOTE|0.38012|0.921695|	0.53825|	0.950797|	0.985516|	[[87427, 4434], [231, 2719]]|
|13|XGB|SMOTEENN|	0.383811|	0.921017|	0.541829|	0.951535|	0.985393|	[[87499, 4362], [233, 2717]]|
|14|XGB|SMOTETomek|	0.373732|	0.924068|	0.532214|	0.949457|	0.985416|	[[87293, 4568], [224, 2726]]|
|15|XGB|EasyEnsemble|	0.341427|	0.934237|	0.500091|	0.941884|	0.985698|	[[86545, 5316], [194, 2756]]|
|16|XGB|BalanceCascade|	0.353312|	0.931186|	0.512261|	0.944827|	0.985708|	[[86833, 5028], [203, 2747|

Table 1. Prediction metrics from both RandomForest and XGBoost using base and resampled datasets.

  Revisiting the metrics in table 1, base models have much better accuracy than resampling models. However, a number of them has better auc than base models, suggesting inappropriate usage of accuracy for evaluating the model performance. The metric confusion-matrix shows the exact values of true negatives (TN), false positive (FP), false negative (FN) and true positive (TP), respectively. Each classifier results in some values for all 4 categories.
  
  |feature|RandomForest|XGBoost|
  |:---:|:---:|:---:|
  |total_pages_visited|1.0|1.0|
  |new_user|0.229|0.292|
  |country_China|0.180|0.292|
  |age|0.060|0.498|
  |country_US|0.016|0.041|
  |country_UK|0.014|0.003|
  |country_Germany|0.005|0.024|
  |source_Direct|0.0004|0.03|
  |source_Ads|0.00008|0.003|
  |source_Seo|0.00002|0.031|

   Table 2. Feature importances from both RandomForest and XGBoost models using base dataset.
  
  To this end, we have successfully predicted conversion rate from test data and evaluated model performance. We then turn to understand what matter(s) the most regarding conversion rate. The feature importances for two models (both using base dataset) were shown in Table 2. "Total_pages_visited" has been unanimously recoginized as the most important feature by both models. This could be explained by that people are interested in this website tending to visit more pages such as for understanding more details, filling required infos, checking service updates, etc. In reality, this is like a leaked feature, since we cannot say whether more visits lead to conversion or the converted visit the websites more frequent thereafter. "new_user" and "country_China" have also shown high importance to both models. As shown in Figure 2, both shows much lower conversion rate compared to their counterpart features. "age" has stand out in XGBoost but not in RandomForest. In addition, source plays insignificant role here.
  
  ![fig_newuser_china](https://user-images.githubusercontent.com/34787111/46118291-a6201b80-c1ba-11e8-984e-61231479cdfa.png)
  
  Figure 2. Plots of conversion rate vs countries (left) and conversion rate vs new_user(right, '1" means new_user).
    
  What if we removed the feature "total_pages_converted"? Are machine learning algorithms able to give reasonable predictions using the remaining features. We repeated the data sampling and modeling processes as described above. Unfortunately, all the models result in very poor prediction, very low precision and true positive. It is interesting to find that base models result in the best accuracy but zero in true positive. Again, predictive accuracy is not appropriate for evaluating models with imbalanced dataset. Besides, this result suggests that strong feature(s) like "total_pages_visited" here are of utmost importance for correct prediction by machine learning models.
  
## 4. Summary and suggestion

  Data resampling was explored to balancing the original highly-imbalanced dataset. The model results suggest that resampling is able to improving recall significantly (at the cost of precison reduction). 
  
  The feature 'total_pages_visited' has intrinsic correlation with conversion. More pages visiting very likely because people have decided to converting. Promoting more page visits may or may not result in higher conversion rate. An A/B test can be designed to promoting pages visit to a group of web-visitors while the other group without any promotion. The two groups should have almost idential distribution in term of country, age, and new_user population.

  China has much lower conversion rate compared to the remaining countries, though it has generally similar distributions in age and new_user population and source. This is likely western world based websites. Is there a promotion difference b/t western world and China? Or cultural difference b/t China and western world contributing to the conversion? Or translation inaccuracy? Or a strong competitor locally in China? All these should be investigated and there are huge room to improving conversion rate in China.
  
  Young age typically has higher conversion rate. Thus, in the next promotion, targeting young population could result in higher conversion rate. In addition, it is also necessary to ask why the service attracks more young people but not the older? Is it because of service nature or others? Also, there are also large room to improving conversion rate by even leveling off all ages to the young class.

  The new_user has much lower conversion rate. By screening out their characteristics (such as young age, from western world, having several pages visited), targeting new_user (eg. by promotion or sending reminding messages) may result in revisiting by new_users and thus have the conversion.

## 5. Reference for data sampling

SMOTE - "SMOTE: synthetic minority over-sampling technique", by Chawla et al., 2002.
Borderline SMOTE - "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning", by Han et al., 2005.
SVM_SMOTE - "Borderline Over-sampling for Imbalanced Data Classification", Nguyen et al., 2011.
SMOTE + Tomek - "Balancing training data for automated annotation of keywords: a case study", Batista et al., 2003.
SMOTE + ENN - "A study of the behavior of several methods for balancing machine learning training data", Batista et al., 2004.
EasyEnsemble & BalanceCascade - "Exploratory Understanding for Class-Imbalance Learning", by Liu et al., 2009.

