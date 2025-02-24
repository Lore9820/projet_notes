# Description: Main file to run the project
from models.model import *
from models.read_files import *
from controllers.preprocessing import *
from controllers.feature_selection import *

# Suppress convergence warnings
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

if __name__ == '__main__':
    #logs_file = input("Saissisez le nom du fichier avec les logs (fichier csv, inclure l'extension) :")
    #notes_file = input("Saissisez le nom du fichier avec les notes (fichier csv, inclure l'extension) :")

    logs = get_logs("data/logs.csv")
    notes = get_notes("data/notes.csv")
    logs = filter_logs(logs, notes)
    logs = split_columns(logs)
    notes = filter_notes(notes, logs)
    df = creer_df(logs)
    df = df_transformer(df)
    # df.columns.to_series().to_csv("models/expected_columns.csv", index=False, header=False)
    X_train, X_test, y_train, y_test = separation_train_test(df, notes)
    print(X_train.head())

    # All features
    select_all = X_train.columns
    # Selected features, Lasso feature selection
    select_lasso = lasso_feature_selection(X_train, y_train)
    # Selected features, forward feature selection
    select_forward, summary = forward_feature_selection(X_train, y_train, taux=0.01)

    # Regroupement des features selectionnées
    selects = [select_all, select_lasso, select_forward]
    select_names = ["All Features", "Lasso Features", "Forward Features"]

    # Définition des modèles à tester
    models = { "Linear_Regression" : LinearRegression(),
               "Random_Forest" : RandomForestRegressor(max_depth=2, random_state=0),
               "AdaBoost" : AdaBoostRegressor(n_estimators=100, random_state=0),}

    # Iterate over each feature selection method
    for select_name, select in zip(select_names, selects):
        # Apply the feature selection method
        X_train_selected = X_train[select]
        X_test_selected = X_test[select]

        # Iterate over each model
        for model_name, model in models.items():
            print(f"Evaluating {model_name} with {select_name}")

            # Fit and evaluate the model
            evaluation_model(model, X_train_selected, y_train, X_test_selected, y_test,
                             f"{model_name} with {select_name}")

    # sauvegarder les modèles pour pouvoir continuer
    LinReg = LinearRegression().fit(X_train[select_lasso], y_train)
    save_model(LinReg, "linear_regression")

    RandFor = RandomForestRegressor(max_depth=2, random_state=0).fit(X_train, y_train)
    save_model(RandFor, "random_forest")

    AdaB = AdaBoostRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
    save_model(AdaB, "ada_boost")

