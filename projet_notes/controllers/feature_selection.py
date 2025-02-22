import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.model import my_linear_regression
from sklearn.linear_model import LassoCV

def forward_feature_selection(X: pd.DataFrame, y: pd.Series, taux: float = 0.001):
    """
    Fait une séléction de features utilisant forward feature selection.
    :param:
    X: DataFrame avec les features
    y (pd.Series): Series avec le target

    :return:
    list: les features séléctionnés en ordre de séléction
    dict: Dictionaire avec les features incluses et les R² adjusted
    """
    selected_features = []
    remaining_features = X.columns.tolist()
    adj_R2_dict = {}
    prev_adj_R2 = 0

    while remaining_features:
        best_feature = None
        best_adj_R2 = prev_adj_R2

        for feature in remaining_features:
            # Creer le df temporair avec les features à tester. Le target sera ajouté dans la fonction my_linear_regression
            df_temp = X[selected_features + [feature]].copy()

            # Construire la formule pour la régression linéaire, entrainer le modèle et obtenir le R² ajusté
            formula = f"target ~ {' + '.join(selected_features + [feature])}"
            model, results = my_linear_regression(X=df_temp, y=y, formula=formula)
            adj_R2 = model.rsquared_adj

            # Mettre à jour le meilleur R² ajusté et la meilleure feature
            if adj_R2 > best_adj_R2:
                best_adj_R2 = adj_R2
                best_feature = feature

        # S'arrêter s'il n'y a pas de improvement
        if best_adj_R2 <= (prev_adj_R2+taux):
            break

        # Update les lists et les R² adjusted
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        adj_R2_dict[best_feature] = best_adj_R2
        prev_adj_R2 = best_adj_R2

    return selected_features, adj_R2_dict

def lasso_feature_selection(X: pd.DataFrame, y: pd.Series):
    """
    Fait une séléction de features utilisant Lasso
    :param:
    X: DataFrame avec les features
    y (pd.Series): Series avec le target
    :return: liste avec les features séléctionnés
    """
    lasso = LassoCV(cv=5, max_iter=100000)
    lasso.fit(X, y)
    selected_features = X.columns[(lasso.coef_ != 0)]
    #print(f"Selected features: {selected_features}")

    return selected_features

if __name__ == '__main__':
    from models.read_files import *
    from models.model import separation_train_test
    from controllers.preprocessing import *
    logs = get_logs()
    notes = get_notes()
    logs = filter_logs(logs, notes)
    logs = split_columns(logs)
    notes = filter_notes(notes, logs)
    df = creer_df(logs)
    pd.set_option('display.max_columns', None)
    df = df_transformer(df)
    X_train, X_test, y_train, y_test = separation_train_test(df, notes)
    features, adj_R2 = forward_feature_selection(X_train, y_train, taux=0.01)
    print(len(features))
    print(features)
    print(adj_R2)
    features = lasso_feature_selection(X_train, y_train)
    print(features)
