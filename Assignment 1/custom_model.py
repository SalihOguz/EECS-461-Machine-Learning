from sklearn.linear_model import LinearRegression
import analysis_and_preprocessing as app
import os

def model_evaluation(X_train, y_train):
    # input: X_train and y_train matrices
    # output: LinearRegression model's coef_ and intercept_ parameters
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    coef_, intercept_ = regressor.coef_, regressor.intercept_
    return coef_, intercept_

def predict(instance, coef_, intercept_):
    # input: instance matrix, coef_ array and intercept_ array
    # ouput: list of predictions for input instances
    regressor = LinearRegression(fit_intercept = True)
    regressor.coef_ = coef_
    regressor.intercept_ = intercept_
    predictions = regressor.predict(instance)
    return predictions

X_train, X_test, y_train, y_test = app.main(os.getcwd(), 0.15, "population")
coef, inter = model_evaluation(X_train, y_train)
predictions = predict(X_test, coef, inter)

import matplotlib.pyplot as plt
plt.scatter(y_test, predictions, color = 'blue')
plt.plot(y_test, y_test, color = 'red')
plt.title('Median House Values')
plt.xlabel('Test Set')
plt.ylabel('Prediction')
plt.show()
