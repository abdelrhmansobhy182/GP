import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import IncreaseData
import encoding
import PredictionModel

# read data
data = pd.read_csv('kidney_disease.csv')
#data.drop('id', axis=1, inplace=True)
data.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                    'aanemia', 'class']
data['packed_cell_volume'] = pd.to_numeric(data['packed_cell_volume'], errors='coerce')
data['white_blood_cell_count'] = pd.to_numeric(data['white_blood_cell_count'], errors='coerce')
data['red_blood_cell_count'] = pd.to_numeric(data['red_blood_cell_count'], errors='coerce')

num_cols = list(data.select_dtypes(['int64', 'float64']))
cat_cols = list(data.select_dtypes('object'))
############## Remove Space
for col in cat_cols:
    data[col] = data[col].str.rstrip()
    data[col] = data[col].str.lstrip()
#######################################################
Selected = ['age', 'blood_pressure', 'blood_urea', 'serum_creatinine', 'pus_cell_clumps', 'bacteria', 'hypertension',
            'diabetes_mellitus', 'coronary_artery_disease'
    , 'appetite', 'peda_edema', 'aanemia', 'class', ]

CDataCol = ['age', 'blood_pressure', 'blood_urea', 'serum_creatinine']
CDataCat = ['pus_cell_clumps', 'bacteria', 'hypertension', 'diabetes_mellitus', 'coronary_artery_disease'
    , 'appetite', 'peda_edema', 'aanemia', 'class', ]

test_list = ['specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell', 'blood_glucose_random', 'sodium',
             'potassium',
             'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
CategralTest = ['red_blood_cells', 'pus_cell']
NumaricTest = ['specific_gravity', 'albumin', 'sugar', 'blood_glucose_random', 'sodium', 'potassium',
               'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
## create new data without selected
test_data = data[test_list]

test_data.to_csv("test data.csv", index=False)

data2 = data
data2 = encoding.encode(data2)
data2.to_csv("temp.csv", index=False)

############## mode
for col in CDataCat:
    data[col] = data[col].fillna(data[col].mode())
############## encoding
pre = LabelEncoder()
for i in CDataCat:
    data[i] = pre.fit_transform(data[i])

################# median
for col in CDataCol:
    data[col] = data[col].fillna(data[col].median())



# apply normalization techniques
for column in data.columns:
	data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())




train_data = data[Selected]
train_data.to_csv("train data.csv", index=False)
###################################################

modelNum = linear_model.LinearRegression()
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
modelCat = LogisticRegression()

for i in CategralTest:
    result = PredictionModel.merge(train_data, data[i])
    TestData = PredictionModel.getTestData(result, i)
    modelCat, selected = PredictionModel.trainModel(result, i, True, modelCat)
    data = PredictionModel.predict(TestData, selected, modelCat, data, i)

for i in NumaricTest:
    result = PredictionModel.merge(train_data, data[i])
    TestData = PredictionModel.getTestData(result, i)
    regressor, selected = PredictionModel.trainModel(result, i, False, regressor)
    data = PredictionModel.predict(TestData, selected, regressor, data, i)
    newData = pd.concat([train_data, data[i]], axis=1, join='inner')
    train_data = newData


data.to_csv("AllDetails.csv", index=False)
# data = IncreaseData.increaseData(data)
print(data)

Y = data['class']
X = data.loc[:, data.columns != 'class']
#Feature Selection
#Get the correlation between the features
# corr = data.corr()
# #Top 50% Correlation training features with the Value
# top_feature = corr.index[abs(corr['class']) > 0.5]
# #Correlation plot
# plt.subplots(figsize=(12, 8))
# top_corr = data[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
# top_feature = top_feature.delete(-1)
# X = X[top_feature]

#print(top_feature)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

########################## Random Forest ##########################
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Top 10 Features
feature_scores=pd.DataFrame(rf.feature_importances_,columns=['Score'],index=X_train.columns).sort_values(by='Score',ascending=False)
top10_feature = feature_scores.nlargest(n=10, columns=['Score'])

plt.figure(figsize=(8,14))
font = {'family' : 'monospace',
        'size'   : 8}

matplotlib.rc('font', **font)
g = sns.barplot(x=top10_feature.index, y=top10_feature['Score'])
p = plt.title('Top 10 Features')
p = plt.xlabel('Feature name')
p = plt.ylabel('score')
p = g.set_xticklabels(g.get_xticklabels(), horizontalalignment='right')
plt.show()
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)
########################## Logistic Regression ##########################
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)
########################## Decision Tree ##########################
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)
########################## Support Vector Machine ##########################
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Support Vector Machine Accuracy:", accuracy)
########################## KNeighbors ##########################
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNeighbors Accuracy:", accuracy)
########################## Naive Bayes ##########################
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)