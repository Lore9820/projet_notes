# Description: Module qui permet de lire les fichiers csv avec les logs et les notes, de les filtrer et de les transformer pour l'analyse
import logging
import sys
import pandas as pd

DEFAULT_LOGS = "data/logs.csv"
DEFAULT_NOTES = "data/notes.csv"

def get_logs(filename = DEFAULT_LOGS):
    '''
    Reads in datafile with logs
    :param filename: file in format csv
    :return: DataFrame logs (if file charged)
    '''
    try:
        logs = pd.read_csv(filename, parse_dates=['heure'])
        return logs
    except:
        logging.error(msg=f"Erreur de chargement du dataset : {filename}")
        sys.exit()

def get_notes(filename = DEFAULT_NOTES):
    '''
    Reads in datafile with notes
    :param filename: file in format csv
    :return: DataFrame notes (if file charged)
    '''
    try:
        notes = pd.read_csv(filename)
        notes['note'] = notes['note'].replace('-', 0).astype(int)
        return notes
    except:
        logging.error(msg=f"Erreur de chargement du dataset : {filename}")
        sys.exit()

def filter_logs(df_logs, df_notes):
    '''
    Fonction qui permet d'enlever les lignes dans le DataFrame logs qui n'ont pas de pseudo correspondant dans le DataFrame notes
    :param df_logs: le fichier avec les logs
    :param df_notes: le fichier avec les notes
    :return: le DataFrame logs filtré (lignes sans correspondance enlevé)
    '''
    logs = df_logs[df_logs["pseudo"].isin(df_notes["pseudo"])]
    return logs

def filter_notes(df_notes, df_logs):
    '''
    Fonction qui permet d'enlever les lignes dans le DataFrame notes qui n'ont pas de pseudo correspondant dans le DataFrame logs
    :param df_notes: le fichier avec les notes
    :param df_logs: le fichier avec les logs
    :return: le DataFrame notes filtré (lignes sans correspondance enlevé)
    '''
    notes = df_notes[df_notes["pseudo"].isin(df_logs["pseudo"])]
    return notes

def split_columns(df_logs):
    '''
    Fonction qui permet de séparer le colonne contexte en contexte générale et spécifications,
    et la colonne heure (datetime) en jour et heures
    :param df_logs: le DataFrame avec les logs
    :return: DataFrame logs avec colonnes séparé
    '''
    df_logs[["contexte_general", "specification"]] = df_logs["contexte"].str.split(": ", n=1, expand=True)
    df_logs["jour"] = df_logs["heure"].dt.date
    df_logs["heures"] = df_logs["heure"].dt.time
    df_logs = df_logs.drop(["contexte"], axis=1)
    return df_logs

if __name__ == "__main__":
    print(get_logs().info())
    print(get_logs())

    print(get_notes().info())
    print(get_notes())

    print(filter_logs(get_logs(), get_notes()).info())
    print(filter_notes(get_notes(), get_logs()).shape)

    print(split_columns(get_logs()))