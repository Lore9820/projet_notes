# Description: Main file to run the project
from models.model import *
from models.read_files import *
from controllers.preprocessing import *
from controllers.feature_selection import *



if __name__ == '__main__':
    #logs_file = input("Saissisez le nom du fichier avec les logs (fichier csv, inclure l'extension) :")
    #notes_file = input("Saissisez le nom du fichier avec les notes (fichier csv, inclure l'extension) :")

    from models.read_files import *
    from controllers.feature_selection import *

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