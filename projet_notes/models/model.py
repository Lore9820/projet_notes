# Description: Ce fichier contient les fonctions permettant de créer un modèle de régression linéaire, de prédire les notes et de calculer le RMSE
from cProfile import label

from controllers.preprocessing import *
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import joblib
import logging
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    model = smf.ols(formula, data=X_temp)
    return model


def fit_linear_regression_sklearn(X: pd.DataFrame, y: pd.Series):
    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model with the provided data
    model.fit(X, y)

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

    :param model:
    :param X:
    :param y:
    :return:
    """
    print(f"Le modèle est: {model}")
    """try:
        model.fit(X_train, y_train)
    except:
        logging.error(msg="Erreur de fitting du modèle")
        sys.exit()"""

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
        plt.scatter(pred_train, y_train - pred_train, c='blue', marker='o', label='train')
        plt.scatter(pred_test, y_test - pred_test, c='red', marker='s', label='test')
        plt.axhline(y=0, color='black', lw=2)
        plt.xlabel('Prédictions')
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

    # All features
    sel_all = X_train.columns
    # Selected features, Lasso feature selection
    sel_lasso = lasso_feature_selection(X_train, y_train)
    # Selected features, forward feature selection
    sel_forward, summary = forward_feature_selection(X_train, y_train, taux=0.01)

    # Regroupement des features selectionnées
    sels = [sel_all, sel_lasso, sel_forward]

    # Définiton et fit  des modèles linéaires
    models_lm = []
    model_names = ["All Features", "Lasso Features", "Forward Features"]
    for sel, name in zip(sels, model_names):
        model = fit_linear_regression_sklearn(X_train[sel], y_train)
        models_lm.append((name, model))

    # Evaluation des modèles linéaires
    for name, model in models_lm:
        print(f"Evaluating model lm: {name}")
        evaluation_model(model, X_train[model.feature_names_in_],
                         y_train, X_test[model.feature_names_in_],
                         y_test, name)

    # Fit et evaluation d'un modèle ElasticNet sur l'ensemble des features
    model_enet = lm.ElasticNetCV(alphas=np.logspace(-6, 6, 13))
    model_enet.fit(X_train, y_train)
    evaluation_model(model_enet, X_train, y_train,
                     X_test, y_test, "Elastic Net")

    # Fit et evaluation d'un modèle Random Forest sur l'ensemble des features
    model_rf = RandomForestRegressor(max_depth=2, random_state=0)
    model_rf.fit(X_train, y_train)
    evaluation_model(model_rf, X_train, y_train,
                     X_test, y_test, "Random Forest")


    """formula = f"target ~ {' + '.join(X_train.columns)}"
    model, results = my_linear_regression(X_train, y_train, formula=formula)
    #save_model(model)
    pred_train = predict_model(model, X_train)
    rmse_train = rmse_model(pred_train, y_train)
    print(f"Le RMSE de la régression multiple sur le jeu train est: {rmse_train}")
    pred_test = predict_model(model, X_test)
    rmse_test = rmse_model(pred_test, y_test)
    print(f"Le RMSE de la régression multiple sur le jeu test est: {rmse_test}")"""
    """print(model)
    #print(model.coef_)
    #print(model.intercept_)
    print(results)
    print(model.predict(X_test), y_test)
    print(rmse_ols)
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

    """"
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
    """

    """reg14 = RandomForestRegressor(max_depth=2, random_state=0)
    reg14.fit(X_train, y_train)
    print("randomforest", reg14.score(X_train, y_train))
    print(reg14.score(X_test, y_test))

    reg15 = GradientBoostingRegressor(random_state=0)
    reg15.fit(X_train, y_train)
    print("GBR", reg15.score(X_train, y_train))
    print(reg15.score(X_test, y_test))"""


    """
    models = [lm.LinearRegression()]
    formula = f"target ~ {' + '.join(sel)}"
    models.append(my_linear_regression(X_train, y_train, formula=formula).fit())

    # Sans feature selection
    print("Sans feature selection")
    sel = X_train.columns
    for model in models:
        evaluation_model(model, X_train, y_train, X_test, y_test)

    # Avec forward feature selection
    print("Avec forward feature selection")
    sel, result = forward_feature_selection(X_train, y_train, taux=0.01)


    # Avec Lasso feature selection
    print("Avec Lasso feature selection")
    sel = lasso_feature_selection(X_train, y_train)


    #X_train = X_train[sel]
    #X_test = X_test[sel]
    #model = lm.LinearRegression() 

    formula = f"target ~ {' + '.join(sel)}"
    model = my_linear_regression(X_train, y_train, formula=formula).fit()
    evaluation_model(model, X_train, y_train, X_test, y_test)
    """

