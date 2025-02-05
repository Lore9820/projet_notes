import logging
import sys
import pandas as pd

DEFAULT_LOGS = "logs.csv"
DEFAULT_NOTES = "notes.csv"

def get_logs(filename = DEFAULT_LOGS):
    '''
    Reads in datafile with logs
    :param filename: file in format csv
    :return: logs (if file charged)
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
    :return: notes (if file charged)
    '''
    try:
        notes = pd.read_csv(filename)
        return notes
    except:
        logging.error(msg=f"Erreur de chargement du dataset : {filename}")
        sys.exit()

def filter_logs(file_logs, file_notes):
    '''
    Fonction qui permet d'enlever les lignes dans le DataFrame logs qui n'ont pas de pseudo correspondant dans le DataFrame notes
    :param file_logs: le fichier avec les logs
    :param file_notes: le fichier avec les notes
    :return: le fichier logs filtré (lignes sans correspondance enlevé)
    '''
    logs = file_logs[file_logs["pseudo"].isin(file_notes["pseudo"])]
    return logs

if __name__ == "__main__":
    print(get_logs().info())
    print(get_logs())

    print(get_notes().info())
    print(get_notes())

    print(filter_logs(get_logs(), get_notes()).info())