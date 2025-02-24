# Description: Ce fichier contient les fonctions permettant de créer un modèle de régression linéaire, de prédire les notes et de calculer le RMSE

from controllers.preprocessing import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import logging
import sys
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn import linear_model as lm
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
    model = smf.ols(formula, data=X_temp)
    return model

def entrainer_model(model, X:pd.DataFrame, y:pd.Series):
    """
    Permet d'entrainer un modèle de machine learning
    :param model: Modèle de machine learning
    :param X: Features
    :param y: Target
    :return: Modèle entrainé
    """
    model.fit(X, y)
    return model

def evaluation_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Permet d'évaluer un modèle de machine learning
    :param model: Modèle de machine learning
    :param X_train: jeu d'entrainement features
    :param y_train: jeu d'entrainement target
    :param X_test: jeu de test features
    :param y_test: jeu de test target
    :param model_name:
    :return:
    """
    print(f"Le modèle est: {model}")
    try:
        model.fit(X_train, y_train)
    except:
        logging.error(msg="Erreur de fitting du modèle")
        sys.exit()

    # metrics sur le jeu d'entrainement
    pred_train = model.predict(X_train)
    rmse_train = rmse_model(pred_train, y_train)
    print(f"\tLe RMSE sur le jeu train est: {rmse_train}")
    R2_train = R2_model(pred_train, y_train)
    print(f"\tLe R² sur le jeu train est: {R2_train}")

    # metrics sur le jeu de test
    pred_test = model.predict(X_test)
    rmse_test = rmse_model(pred_test, y_test)
    print(f"\tLe RMSE sur le jeu test est: {rmse_test}")
    R2_test = R2_model(pred_test, y_test)
    print(f"\tLe R² sur le jeu test est: {R2_test}")
    print("\n")

    # courbe d'apprentissage
    try:
        N, train_score, val_score = learning_curve(model, X_train, y_train, cv=4, train_sizes=np.linspace(0.2, 1, 10))
        plt.figure(figsize=(12, 8))
        plt.plot(N, train_score.mean(axis=1), label='train score')
        plt.plot(N, val_score.mean(axis=1), label='validation score')
        plt.xlabel('train_sizes')
        plt.ylabel('score (R²)')
        plt.title(f'Courbe d\'apprentissage - {model_name}')
        plt.legend()
        plt.show()
    except:
        logging.error(msg="Erreur de visualisation de la courbe d'apprentissage")

    # courbe de prédiction
    try:
        plt.figure(figsize=(12, 8))
        plt.scatter(y_test, pred_test, c='red', label='test')
        plt.scatter(y_train, pred_train, c='blue', label='train')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        plt.xlabel('Valeurs réelles de y')
        plt.ylabel('Prédictions')
        plt.title(f'Courbe de prédiction - {model_name}')
        plt.legend()
        plt.show()
    except:
        logging.error(msg="Erreur de visualisation de la courbe de prédiction")

    # courbe de résidus
    try:
        plt.figure(figsize=(12, 8))
        plt.scatter(y_train, y_train - pred_train, c='blue', marker='o', label='train')
        plt.scatter(y_test, y_test - pred_test, c='red', marker='s', label='test')
        plt.axhline(y=0, color='black', lw=2)
        plt.xlabel('y value')
        plt.ylabel('Résidus')
        plt.title(f'Courbe de résidus - {model_name}')
        plt.legend()
        plt.show()
    except:
        logging.error(msg="Erreur de visualisation de la courbe de résidus")

    return rmse_train, R2_train, rmse_test, R2_test

def predict_model(model, X):
    """
    Permet de prédire les notes avec un modèle de machine learning
    :param model: Modèle de machine learning
    :param df: Dataframe avec les features
    :return: Prédictions
    """
    pred = model.predict(X)
    return pred

def rmse_model(pred, y):
    """
    Permet de calculer le RMSE d'un modèle de machine learning
    :param pred: Prédictions
    :param y_test: Valeurs réelles
    :return: RMSE
    """
    return np.sqrt(np.mean((pred - y)**2))

def R2_model(pred, y):
    """
    Permet de calculer le coefficient de déterminant R² d'un modèle de machine learning
    :param pred: Prédictions
    :param y: Target
    :return: R² (score)
    """
    return 1 - (((y - pred)** 2).sum() / ((y - y.mean()) ** 2).sum())

def save_model(model, filename):
    """
    Permet de sauvegarder un modèle de machine learning
    :param model: Modèle de machine learning
    :param path: Chemin où sauvegarder le modèle
    :return: None
    """
    filepath = f"models/{filename}.pkl"
    joblib.dump(model, filepath)

if __name__ == '__main__':
    from models.read_files import *
    from controllers.feature_selection import *
    logs = get_logs("../data/logs.csv")
    notes = get_notes("../data/notes.csv")
    logs = filter_logs(logs, notes)
    logs = split_columns(logs)
    notes = filter_notes(notes, logs)
    df = creer_df(logs)
    df = df_transformer(df)
    #df.columns.to_series().to_csv("models/expected_columns.csv", index=False, header=False)
    X_train, X_test, y_train, y_test = separation_train_test(df, notes)
    print(X_train.head())

    # test de la fonction my_linear_regression
    formula = f"target ~ {' + '.join(X_train.columns)}"
    model = my_linear_regression(X_train, y_train, formula=formula).fit()
    print(model.summary())

    # test de la fonction entrainer_model
    model = entrainer_model(LinearRegression(), X_train, y_train)
    print(model.score(X_train, y_train))

    # test de la fonction evaluation_model
    model = LinearRegression()
    evaluation_model(model, X_train, y_train, X_test, y_test, "Linear Regression")



