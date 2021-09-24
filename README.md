# News-Popularity-Prediction

Predicting Popularity of various news topics (namely Obama, Microsoft, Economy and Palestine) on different social media platform (Facebook, LinkedIn and GooglePlus).

---

## Table of Content
  * [Abstract](#abstract)
  * [Problem Statement](#problem-statement)
  * [Data Description](#data-description)
  * [Project Outline](#project-outline)
    - 1 [Data Wrangling](#data-wrangling)
    - 2 [Standardization](#standardization)
    - 3 [EDA](#eda)
    - 4 [Text Pre-processing](#text-pre-processing)
    - 5 [Encoding categorical values](#encoding-categorical-values)
    - 6 [Feature Selection](#feature-selection)
    - 7 [Model Fitting](#model-fitting)
    - 8 [Hyper-parameter Tuning](#hyper-parameter-tuning)
    - 9 [Metrics Evaluation](#metrics-evaluation)
    - 10 [Feature Importance - SHAP Implementation](#feature-importance-shap-implementation)
  * [Conclusion](#run)
  * [Reference](#reference)

---


# Abstract
News popularity on various social media platforms depends on multiple features like the topic, source of publication, time-span and sentiment score. Here we are provided with a dataset that contains news items and their respective social feedback on different platforms: Facebook, GooglePlus and LinkedIn.
Based on the previous trend, this data analysis and prediction with machine learning models can help us understand what are the reasons for news popularity on social media and obtain the best regression model.

# Problem Statement
A large data set of news items and their respective social feedback on multiple platforms: Facebook, Google+ and LinkedIn.The collected data relates to a period of 8 months, between November 2015 and July 2016, accounting for about 100,000 news items on four different topics: Economy, Microsoft, Obama and Palestine.

# Data Description
We have 13 dataset which are categorised into 2 types News_Final dataset and Time Span Dataset.
### **1. News_Final Dataset(1 Dataset)**: 
It contains information of news related to 4 topics Microsoft, Obama, Economy and Palestine collected from different sources and their popularity on 3 social media platform namely Facebook, GooglePlus and LinkedIn.

| Feature Name | Type | Description |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|IDLink| (numeric) | Unique identifier of news items|
|Title |(string)|Title of the news item according to the official media sources|
|Headline |(string)|Headline of the news item according to the official media sources|
|Source |(string)|Original news outlet that published the news item|
|Topic |(string)|Query topic used to obtain the items in the official media sources|
|PublishDate| (timestamp)|Date and time of the news items' publication|
|SentimentTitle| (numeric)|Sentiment score of the text in the news items' title|
|SentimentHeadline |(numeric)|Sentiment score of the text in the news items' headline|
|Facebook| (numeric)|Final value of the news items' popularity according to the social media source Facebook|
|GooglePlus| (numeric)|Final value of the news items' popularity according to the social media source Google+|
|LinkedIn |(numeric)|Final value of the news items' popularity according to the social media source LinkedIn|

### **2. Time Span Dataset(12 dataset)**:  
The 12 datasets are permutation of 4 topics and 3 platforms as stated above. Each dataset contains 2 days news popularity divided in 20 minutes interval after it is published i.e. 144 Time span features.

| Feature Name | Type | Description |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|IDLink |(numeric)| Unique identifier of news items|
|TS1 |(numeric)|Level of popularity in time slice 1 (0-20 minutes upon publication)|
|TS2 |(numeric)|Level of popularity in time slice 2 (20-40 minutes upon publication)|
|TS... |(numeric)|Level of popularity in time slice ...|
|...|...|...|
|...|...|...|
|TS144 |(numeric)|Final level of popularity after 2 days (2860-2880) upon publication|

---
# Project Outline

## 1. Data Wrangling
After loading our datasets, we observed that in the news df, the source column contained 279 null values, so we replaced the null values with the source that has published the maximum number of news items. Further, we dropped the news items for which the published date was before Nov 2015 as this is trash data. We also performed the data cleaning by dropping duplicate rows and news items with null values in headlines. We treated the outliers in the dataset by using the 90th percentile method.

## 2. Standardization
We observed that the values of our dependent variables were quite large compared to the values in the independent variables. So to standardize the data we applied StandardScaler for data transformation.
| Data Before Standardization  | Data After Standardization |
|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ![image](https://user-images.githubusercontent.com/43884418/134707732-b0121b71-803c-4e91-be55-e89da3407267.png)|![image](https://user-images.githubusercontent.com/43884418/134707850-c0854391-76c6-48ae-8bf2-51b96ba5ae09.png) |

## 3. EDA
In Exploratory Data Analysis, we categorized the sentimentTitle and sentimentHeadline into positive, neutral and negative sentiment categorical values. Then we observed that the number of sources publishing news items was quite large so to obtain a proper analysis we categorized the sources into 4 types, by grouping them on the number of news items published by each source. Further, we compared the popularity level of news items on the three social media platforms, observed the trend of popularity level based on sources and topics. We also observed the change in popularity level within the two days of publishing by using the time-span dataset.

## 4. Text Pre-processing
For handling the textual data in the news title and headline we used TF-IDF Vectorizer and CountVectorizer.

## 5. Encoding categorical values
We used one-hot encoding for converting the categorical columns such as source types, topics, sentiment category into numerical values so that our model can understand and extract valuable information from these columns.

## 6. Feature Selection
For feature selection, we used algorithms like ExtraTreeRegressor, which provides ordering features on the basis of their gini importance and helps in obtaining features  which are more important compared to others for our model.
Next we obtained correlation between the independent and dependent features to understand their relation.

![image](https://user-images.githubusercontent.com/43884418/134711702-d95ada21-a7ed-4314-a340-8c7118edaf23.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/43884418/134711764-1b859ea1-95bb-417b-be18-c78d344e7c63.png" />
</p>

## 7. Model Fitting
For modeling we tried the various regression algorithms like:
* Decision Tree Regression
* CatBoost
* LightGBM
* GradientBoosting 
* KNN

## 8. Hyper parameter Tuning
Tuning of hyperparameters is necessary for modeling to obtain better accuracy and to avoid overfitting. In our project, we used the
 - GridSearchCV, 
 - RandomizedSearchCV and 
 - HalvingRandomizedSearchCV.

 Performance of different hyperparameter tuning techniques on Google Colaboratory
| GridSearchCV  |  RandomizedSearchCV  | HalvingRandomizedSearchCV |
|----|----|----|
|![image](https://user-images.githubusercontent.com/35359451/134711878-350e3e1a-8e63-4436-89d6-e203a42723c5.png)|![image](https://user-images.githubusercontent.com/35359451/134711905-559bf15c-5a6b-41c6-a7d8-4f9ef7951aab.png)|![image](https://user-images.githubusercontent.com/35359451/134711961-2b362b0d-9d64-40ec-990e-31f3eafb5ba6.png)|

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
 We used plotly to showcase the parameter score of hyperparameters in different algorithms when doing hyperparameter tuning. In the plot below we are showing hyperparameter tuning of CatBoost using HalvingRandomizedSearchCV
![image](https://user-images.githubusercontent.com/35359451/134615300-a0d2a597-4195-4691-bf04-9eff13a2d58c.png)

## 9. Metrics Evaluation
We used some of the metrics valuation techniques like **MSE, RMSE, R2 Score, Adjusted R2, RMLSE**, to obtain the accuracy and error rate of our models before and after hyperparameter tuning.

## 10. Feature Importance - SHAP Implementation
We implemented the SHAP value plot to obtain the importance of independent features in the model prediction to show the positive or negative relationship for each variable with the target.

| Force Plot  | Summary Plot | Summary Plot (Bar Type) |
|----|----|----|
|![image](https://user-images.githubusercontent.com/35359451/134615402-ed7a3c28-c192-47e5-85b8-f356b223477d.png)|![image](https://user-images.githubusercontent.com/35359451/134615469-f41fde4e-f2a6-4f6b-8da2-307cd5379337.png)|![image](https://user-images.githubusercontent.com/35359451/134615492-d517fe35-766e-4099-b3e6-50227061d9ab.png)|
---
# Conclusion
 - Starting from loading the datasets, we covered data wrangling, EDA, feature selection and modeling.
 - The R2 Score obtained for all models revolved around **85% to 92%** for all the three dependent variables.
 - Further, we carried out hyperparameter tuning and obtained the best score and best parameters for all the models and there was not much improvement in the R2   Score.
 - So the accuracy obtained by the best model is **92%**. From here we can also conclude that the TS columns in the time-span dataset have a higher influence on our dependent variables.


---
Here is a glimpse of few graphs we plotted, there are many more in the notebook please have a look.
![image](https://user-images.githubusercontent.com/43884418/134709525-2da3b8df-4946-4abc-af7e-4b65fa37bc07.png)|![image](https://user-images.githubusercontent.com/43884418/134709450-781d758e-6d96-45b9-86bc-27c6ba96283c.png)|![image](https://user-images.githubusercontent.com/43884418/134708698-c6ec806f-f312-422a-9d36-b3473923345f.png)|
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/43884418/134709549-180089bc-fbb8-4b83-a5ef-a7db84a6fa8a.png)|![image](https://user-images.githubusercontent.com/43884418/134708891-14ec0429-303e-4803-8ce8-5083558f401f.png)|![image](https://user-images.githubusercontent.com/43884418/134708790-222596fa-9037-4d22-92d1-d407d30b7319.png)|
![image](https://user-images.githubusercontent.com/43884418/134709027-edb41333-d2f3-4c02-85d4-7c8a679dc698.png)|![image](https://user-images.githubusercontent.com/43884418/134709084-2992a69b-15d4-4d35-ad1f-20bb1ef9a41e.png)|![image](https://user-images.githubusercontent.com/43884418/134709193-269ce187-c35d-4823-b8ee-2a462edd9e58.png)|
![image](https://user-images.githubusercontent.com/43884418/134709318-05cbd219-d8fa-4059-a587-afcea586b2b0.png)|![image](https://user-images.githubusercontent.com/43884418/134710694-c77750bb-2eec-4478-8a98-820d9b6775e6.png)|![image](https://user-images.githubusercontent.com/35359451/134726433-007060f1-b975-42f4-a98a-4897c402e981.png)|

---
# References
 - https://towardsdatascience.com
 - https://www.analyticsvidhya.com
 - https://machinelearningmastery.com


---
---

Meet The Team:

> Bhwesh Gaur: https://github.com/bhweshgaur

> S Sravya Sri : https://github.com/SSravyaSri

> Priyanka Dodeja :  https://github.com/PriyankaDodeja

> Ankit Bansal : https://github.com/ankit986
