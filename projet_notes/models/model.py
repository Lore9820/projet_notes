# Description: Ce fichier contient les fonctions permettant de créer un modèle de régression linéaire, de prédire les notes et de calculer le RMSE
from controllers.preprocessing import *
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn import linear_model as lm
import numpy as np
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import learning_curve

def my_linear_regression(X:pd.DataFrame, y:pd.Series, formula:str):
    """
    Permet de créer un modèle de régression linéaire
    :param df: Dataframe avec les features (X_train)
    :param df_notes: Dataframe avec le target (y_train)
    :return: Modèle régression linéaire
    """
    X_temp = X.copy()
    X_temp['target'] = y
    model = smf.ols(formula, data=X_temp).fit()
    results = model.summary()
    return model, results

def evaluation_model(model, X_train, y_train, X_test, y_test, param):
    """

    :param model:
    :param X:
    :param y:
    :return:
    """
    reg = model(alpha=param)
    reg.fit(X_train, y_train)
    print("model avec elastic net", reg.score(X_train, y_train))
    print(reg.score(X_test, y_test))

    N, train_score, val_score = learning_curve(reg, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.xlabel('train_sizes')
    plt.legend()
    plt.show()

    return reg


def predict_model(model, X_test):
    """
    Permet de prédire les notes avec un modèle de machine learning
    :param model: Modèle de machine learning
    :param df: Dataframe avec les features
    :return: Prédictions
    """
    return model.predict(X_test)

def rmse_model(pred, y_test):
    """
    Permet de calculer le RMSE d'un modèle de machine learning
    :param pred: Prédictions
    :param y_test: Valeurs réelles
    :return: RMSE
    """
    return np.sqrt(np.mean((pred - y_test)**2))

def save_model(model):
    """
    Permet de sauvegarder un modèle de machine learning
    :param model: Modèle de machine learning
    :param path: Chemin où sauvegarder le modèle
    :return: None
    """
    joblib.dump(model, "../models/model.pkl")
    print("Model opgeslagen als models/model.pkl")

if __name__ == '__main__':
    from models.read_files import *
    logs = get_logs()
    notes = get_notes()
    logs = filter_logs(logs, notes)
    logs = split_columns(logs)
    notes = filter_notes(notes, logs)
    df = creer_df(logs)
    df = df_transformer(df)
    X_train, X_test, y_train, y_test = separation_train_test(df, notes)
    print(X_train.head())
    formula = f"target ~ {' + '.join(X_train.columns)}"
    model, results = my_linear_regression(X_train, y_train, formula=formula)
    #pred = predict_model(model, X_test)
    #rmse_ols = rmse_model(pred, y_test)
    """print(model)
    #print(model.coef_)
    #print(model.intercept_)
    print(results)
    print(model.predict(X_test), y_test)
    #print(rmse_ols)
    #print(model.score(X_train, y_train))
    #print(model.score(X_test, y_test))"""


    """reg = lm.RidgeCV(alphas=np.logspace(-6, 6, 13))
    print(reg.fit(X_train, y_train))
    print(reg.alpha_)
    reg2 = lm.Ridge(alpha=reg.alpha_)
    reg2.fit(X_train, y_train)
    #print(reg2.coef_)
    #print(reg2.intercept_)
    print("model avec ridge", reg2.score(X_train, y_train))
    print(reg2.score(X_test, y_test))"""

    """reg3 = lm.LassoCV(alphas=np.logspace(-6, 6, 13))
    print(reg3.fit(X_train, y_train))
    print(reg3.alpha_)
    reg4 = lm.Lasso(alpha=reg3.alpha_)
    reg4.fit(X_train, y_train)
    #print(reg4.coef_)
    #print(reg4.intercept_)
    print("model avec lasso", reg4.score(X_train, y_train))
    print(reg4.score(X_test, y_test))"""

    """reg5 = LinearSVR(C=0.01, dual=False).fit(X_train, y_train)
    model = SelectFromModel(reg5, prefit=True)
    X_new = model.transform(X_train)
    print(X_new.shape)"""

    """reg6 = lm.LinearRegression().fit(X_train, y_train)
    print("general LM model", reg6.score(X_train, y_train))
    print(reg6.score(X_test, y_test))"""

    """reg7 = lm.ElasticNetCV(alphas=np.logspace(-6, 6, 13))
    print(reg7.fit(X_train, y_train))
    print(reg7.alpha_)
    reg8 = lm.ElasticNet(alpha=reg7.alpha_)
    reg8.fit(X_train, y_train)
    print("model avec elastic net", reg8.score(X_train, y_train))
    print(reg8.score(X_test, y_test))

    evaluation_model(ElasticNet, X_train, y_train, X_test, y_test, reg7.alpha_)"""

    """reg9 = lm.SGDRegressor(max_iter=1000, tol=1e-3)
    reg9.fit(X_train, y_train)
    print("SGDRegressor", reg9.score(X_train, y_train))
    print(reg9.score(X_test, y_test))

    reg10 = NuSVR(C=1.0, nu=0.1)
    reg10.fit(X_train, y_train)
    print("NuSVR", reg10.score(X_train, y_train))
    print(reg10.score(X_test, y_test))

    reg11 = SVR(C=1.0, epsilon=0.2)
    reg11.fit(X_train, y_train)
    print("SVR", reg11.score(X_train, y_train))
    print(reg11.score(X_test, y_test))

    reg12 = LinearSVR(epsilon=0.0, tol=1e-5)
    reg12.fit(X_train, y_train)
    print("LinearSvr", reg12.score(X_train, y_train))
    print(reg12.score(X_test, y_test))"""

    reg13 = AdaBoostRegressor(n_estimators=100, random_state=0, loss='square')
    reg13.fit(X_train, y_train)
    print("Adaboost", reg13.score(X_train, y_train))
    print(reg13.score(X_test, y_test))
    N, train_score, val_score = learning_curve(reg13, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.xlabel('train_sizes')
    plt.legend()
    plt.show()

    save_model(reg13)

    """reg14 = RandomForestRegressor(max_depth=2, random_state=0)
    reg14.fit(X_train, y_train)
    print("randomforest", reg14.score(X_train, y_train))
    print(reg14.score(X_test, y_test))

    reg15 = GradientBoostingRegressor(random_state=0)
    reg15.fit(X_train, y_train)
    print("GBR", reg15.score(X_train, y_train))
    print(reg15.score(X_test, y_test))"""