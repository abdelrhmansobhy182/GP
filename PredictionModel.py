import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pandas as pd
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, r2_score
import sklearn.metrics as sm


def merge(data, dataframe):
    result = pd.concat([data, dataframe], axis=1, join='inner')
    return result


modelNum = linear_model.LinearRegression()


###### all data
def trainModel(data, dataframe, classification):
    data.dropna(inplace=True)
    X = data.drop(dataframe, axis=1)
    Y = data[dataframe]
    # print(dataframe)

    feature = ""
    if classification is True:
        model = LogisticRegression()
        feature = selectFeatureChi2(X, Y)
        X = data[feature]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)
        model.fit(X_train, y_train)



    else:
        feature = selectFeatureCorr(data, dataframe)
        X = data[feature]
        print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)
        # poly_features = PolynomialFeatures(degree=3)
        # X_train_poly = poly_features.fit_transform(X_train)
        # model = linear_model.LinearRegression()
        # model.fit(X_train_poly, y_train)
        modelNum.fit(X_train, y_train)

    if classification is True:
        y_pred = model.predict(X_test)
        # classificationAccuracy(y_test,y_pred)
        return model, feature
    else:
        # regressionAccuracy(y_test,y_pred)
        print("Accuracy of training dataset:", modelNum.score(X_train, y_train))
        print("Accuracy of test dataset:", modelNum.score(X_test, y_test))
        # print('Coefficient of determination: %.2f' % r2_score(y_test, predict))
        return modelNum, feature
    # print(feature)


# after merge
def getTestData(data, dataframe):
    TestData = data[data[dataframe].isnull()]
    return TestData


def predict(TestData, selected, model, data, dataframe):
    # print(TestData)
    X_Test = TestData[selected]
    # print(X_Test)
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


def selectFeatureChi2(X, Y):
    best_features = SelectKBest(score_func=chi2, k=3)
    fit = best_features.fit(X, Y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    features_scores = pd.concat([df_columns, df_scores], axis=1)
    features_scores.columns = ['Features', 'Score']
    features_scores = features_scores.sort_values(by='Score', ascending=False)
    feature = features_scores['Features'].tolist();
    # print(feature[0:3])
    return feature[0:3]


def selectFeatureCorr(data, Y):
    corr = data.corr()
    # print(data)
    # Top 50% Correlation training features with the Value
    top_feature = corr.index[abs(corr[Y]) > 0.4]
    # Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_corr = data[top_feature].corr()
    top_feature = top_feature.delete(-1)
    # print(len(top_feature))
    # print(top_feature)
    return top_feature


def scale(X):
    # Apply Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X


def classificationAccuracy(y_test, y_pred):
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred, zero_division=1))
    print('Precision:', metrics.precision_score(y_test, y_pred, zero_division=1))
    print("CL Report:", metrics.classification_report(y_test, y_pred, zero_division=1))


def regressionAccuracy(y_test, y_pred):
    print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))
    print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2))
    print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2))
    print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))
    print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))
