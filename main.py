import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import encoding
import PredictionModel

# read data
data = pd.read_csv('kidney_disease.csv')
data.drop('id', axis=1, inplace=True)
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
    result.to_csv("result.csv", index=False)
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
    # print(train_data)
def prediction_plot(model,X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    y_test = np.array(list(y_test))
    y_pred = np.array(y_pred)
    df_ans = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    print(df_ans)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(classification_report(y_test, y_pred))

    # confusion matrix plot

# confusion matrix plot
def conf(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', color='black')
    ax.set_ylabel('Actual outputs',  color='black')
    ax.xaxis.set(ticks=range(2))
    ax.yaxis.set(ticks=range(2))
    # ax.set_ylim(9.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()


data.to_csv("AllDetails.csv", index=False)
X= data.iloc[:,:-1]
Y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# fit the model
# #KNN
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# prediction_plot(model,X_train, X_test, y_train, y_test)
# conf(y_test, y_pred)

# bagging classifier
# ensemble learning with decision tree
# no. of base classifier
num_trees = 100
base_cls = DecisionTreeClassifier()
seed = 8
model = BaggingClassifier(base_estimator = base_cls,
                          n_estimators = num_trees,
                          random_state = seed)
history = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
prediction_plot(model,X_train, X_test, y_train, y_test)
conf(y_test, y_pred)
