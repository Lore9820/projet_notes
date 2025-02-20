import pandas as pd
from models.model import my_linear_regression

def forward_feature_selection(X: pd.DataFrame, y: pd.Series):
    """
    Fait une séléction de features utilisant forward feature selection.
    :param:
    X: DataFrame avec les features
    y (pd.Series): Series avec le target

    :returne:
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
            # Maak een tijdelijke DataFrame met geselecteerde features en target
            df_temp = X[selected_features + [feature]].copy()

            # Bouw de formule dynamisch
            formula = f"target ~ {' + '.join(selected_features + [feature])}"
            model, results = my_linear_regression(X=df_temp, y=y, formula=formula)
            adj_R2 = model.rsquared_adj

            if adj_R2 > best_adj_R2:
                best_adj_R2 = adj_R2
                best_feature = feature

        # Stop als er geen verbetering is
        if best_adj_R2 <= prev_adj_R2:
            break

        # Update de lijsten en R²-waarden
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        adj_R2_dict[', '.join(selected_features)] = best_adj_R2
        prev_adj_R2 = best_adj_R2

    return selected_features, adj_R2_dict

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
    features, adj_R2 = forward_feature_selection(X_train, y_train)
    print(features)
    print(adj_R2)