from controllers.preprocessing import *
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf

def my_linear_regression(X:pd.DataFrame, y:pd.Series, formula:str):
    """
    Permet de créer un modèle de régression linéaire
    :param df: Dataframe avec les features (X_train)
    :param df_notes: Dataframe avec le target (y_train)
    :return: Modèle régression linéaire
    """
    df_temp = X.copy()
    df_temp['target'] = y
    model = smf.ols(formula, data=df_temp).fit()
    results = model.summary()
    return model, results

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
    '''print(features)
    print(adj_R2)
    model, results = my_linear_regression(X_train, y_train)
    pred = predict_model(model, X_test)
    rmse_ols = rmse_model(pred, y_test)
    print(model)
    print(model.coef_)
    print(model.intercept_)
    print(results)
    print(rmse_ols)
    #print(model.score(X_train, y_train))
    #print(model.score(X_test, y_test))'''