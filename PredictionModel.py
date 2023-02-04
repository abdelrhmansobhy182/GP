from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import math

def merge (data , dataframe):
    result = pd.concat([data, dataframe], axis=1, join='inner')
    return result



###### all data
def trainModel (data , dataframe,classification):
    data.dropna(inplace=True)
    X_Train = data.drop(dataframe, axis=1)
    Y_train = data[dataframe]
    if classification is True:
        model = LogisticRegression()
    else :
        model = LinearRegression()
    model.fit(X_Train, Y_train)
    return  model

# after merge
def getTestData(data , dataframe):
    TestData = data[data[dataframe].isnull()]
    return  TestData

def predict(TestData , model ,data , dataframe):
    X_Test = TestData.iloc[:, 0:13]
    Y_Pred = model.predict(X_Test)
    TestData['prediction'] = Y_Pred
    counter = 0
    for i in range(data.shape[0]):
        try:
            if math.isnan(float(data[dataframe][i])):
                data[dataframe][i] = Y_Pred[counter]
                counter += 1
        except:
            continue
    return data




