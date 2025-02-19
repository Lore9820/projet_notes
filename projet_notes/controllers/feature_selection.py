import pandas as pd

def forward_feature_selection(X: pd.DataFrame, y: pd.Series):
    """
    Fait une séléction de features utilisant un forward feature selection.
    :param:
    X: DataFrame avec les features
    y (pd.Series): Series avec le target

    :returne:
    list: les features séléctionnés en ordre de séléction
    dict: Dictionaire avec les features incluses et les R² adjusted
    """
    selected_features = []
    remaining_features = X.columns.tolist()  # Kopie van featurelijst
    adj_R2_dict = {}
    prev_adj_R2 = 0

    while remaining_features:
        best_feature = None
        best_adj_R2 = prev_adj_R2

        for feature in remaining_features:
            # Maak een tijdelijke DataFrame met geselecteerde features en target
            df_temp = X[selected_features + [feature]].copy()
            df_temp['target'] = y  # Voeg de target toe

            # Fix feature-namen met Q()
            sanitized_features = df_temp.drop(columns=['target'])

            # Bouw de formule dynamisch
            formula = f"target ~ {' + '.join(sanitized_features)}"
            model = smf.ols(formula, data=df_temp).fit()
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
