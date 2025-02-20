import logging
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from controllers.features_creation import *
import re

#Créer le DataFrame pour analyses
def creer_df(df_logs:pd.DataFrame):
    """
    Fonction qui permet de créer le DataFrame qui va servir pour le reste de l'analyse
    :param df_logs: dataframe contenant les logs
    :return: DataFrame avec toutes les features qu'on a créées
    """
    df = nb_actions(df_logs)
    df = df.merge(moyenne_actions_par_jour(df_logs), on="pseudo", how="left")
    df = df.merge(nb_jours_avec_action(df_logs), on="pseudo", how="left")
    df = df.merge(variabilite_activite(df_logs), on="pseudo", how="left")
    df = df.merge(tempsdiff(df_logs), on="pseudo", how="left")
    df = df.merge(constance_activite(df_logs), on="pseudo", how="left")
    df = df.merge(periode_moyen_activite(df_logs), on="pseudo", how="left")
    df = df.merge(pourcentage_nuit(df_logs), on="pseudo", how="left")
    df = df.merge(pourcentage_matin(df_logs), on="pseudo", how="left")
    df = df.merge(pourcentage_aprem(df_logs), on="pseudo", how="left")
    df = df.merge(pourcentage_soir(df_logs), on="pseudo", how="left")
    df = df.merge(semaine_vs_weekend(df_logs), on="pseudo", how="left")
    df = df.merge(nb_contexte_gen(df_logs), on="pseudo", how="left")
    df = df.merge(nb_specifications(df_logs), on='pseudo', how='left')
    df = df.merge(nb_composant(df_logs), on="pseudo", how="left")
    df = df.merge(nb_chaque_contexte(df_logs), on="pseudo", how="left")
    df = df.merge(top_contexte(df_logs), on="pseudo", how="left")
    df = df.merge(nb_chaque_composant(df_logs), on="pseudo", how="left")
    df = df.merge(top_composant(df_logs), on="pseudo", how="left")
    df = df.merge(nb_evenement(df_logs), on="pseudo", how="left")
    df = df.merge(nb_chaque_evenement(df_logs), on="pseudo", how="left")
    df = df.merge(top_evenement(df_logs), on="pseudo", how="left")
    return df

def save_dataframe(df:pd.DataFrame, filename:str):
    """
    Ecrire le df créé dans un fichier csv
    :param df: DataFrame à écrire
    :param filename: nombre du fichier (sans extension)
    """
    df.to_csv(f"{filename}.csv", index=False)
    print(f"DataFrame opgeslagen als {filename}.csv")

# Train-set split
def separation_train_test(df:pd.DataFrame, df_notes:pd.DataFrame):
    """
    Préparation basique des dataframes pour la suite de l'analyse
    :param df: dataframe contenant les features
    :param df_notes: dataframe contenant les notes
    :return: un dataframe X_train avec les features et 80% des observations, un dataframe X_test avec les features et 20% des observations,
    un vecteur y_train avec les notes de 80% des observations et un vecteur y_test avec les notes de 20% des observations
    """
    df_all = df.merge(df_notes, on="pseudo") #S'assurer que les deux df sont au même ordre
    y = df_all["note"]
    X = df_all.drop(["note", "pseudo"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

# Transformation du DataFrame logs
def enlever_correlations_complets(df:DataFrame):
    """
    Enlève les features qui on une corrélation de 1 (deuxième encontré est enlèvé)
    :param df: Dataframe avec les features
    :return: Dataframe sans doublons
    """
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
    columns_to_drop = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) == 1:  # If correlation is exactly 1
                colname = corr_matrix.columns[i]  # Get the column name
                columns_to_drop.append(colname)
    df_cleaned = df.drop(columns=columns_to_drop)
    #print(f"Columns dropped: {columns_to_drop}")
    #print(df.shape)
    return df_cleaned

def encodage(df:pd.DataFrame):
    """
    Fonction qui permet d'encoder les variables catégorielles
    :param df: Dataframes avec les features
    :return: Dateframe avec seulement des variables numériques
    """
    encoder = OneHotEncoder(sparse_output=False)
    categorical_cols = df.select_dtypes(include='object').columns
    encoded_array = encoder.fit_transform(df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols)
    df_encode = df.drop(columns=categorical_cols).join(df_encoded)

    return df_encode

def scaling(df:pd.DataFrame):
    """
    Permet de standardiser un dataframe antérieurement encodé. MinMax est utilisé pour garder les colonnes binaires
    :param df: Dataframe avec seulement valeurs numériques
    :return: Dataframe avec colonnes scaled
    """
    df_scaled = df.copy()
    cols_to_scale = df_scaled.columns.difference(['pseudo'])
    scaler = MinMaxScaler()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    return df_scaled

def reduction_dimensions(df:pd.DataFrame):
    """

    :param df:
    :return:
    """
    pass

def df_transformer(df:pd.DataFrame):
    """
    Permet de transformer le dataframe et le mettre en bon format pour utiliser dans les modèles de machine learning
    :param df: Dataframe avec les features
    :return: Dataframe transformé sans variables corrélées à 100%, toutes les variables numériques (encodées) et scalées
    """
    df = enlever_correlations_complets(df)
    df = encodage(df)
    df = scaling(df)

    #Remplacer les espaces et les tirets par des underscores
    for col in df.columns:
        new_col = re.sub(r'\W+', '_', col)
        df = df.rename(columns={col: new_col})

    return df

if __name__ == '__main__':
    import models.read_files as modele

    logs = modele.get_logs()
    notes = modele.get_notes()
    logs = modele.filter_logs(logs, notes)
    logs = modele.split_columns(logs)
    notes = modele.filter_notes(notes, logs)
    print(logs.head(10))
    print(logs.shape)
    print(notes.shape)

    df = creer_df(logs)
    print(df.head(10))
    print(df.shape)
    print(df.columns)

    X_train, X_test, y_train, y_test = separation_train_test(df, notes)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    pd.set_option('display.max_columns', None)
    print(X_train.head())

    """X_train_encode = encodage(X_train)
    print(X_train_encode.shape)
    print(X_train_encode.head())
    """

    #save_dataframe(df, "df_complet")

    #df_cleaned = df_transformer(X_train)
    #print(df_cleaned.head(10))
    #print(df_cleaned.shape)