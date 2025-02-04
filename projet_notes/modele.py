import logging
import sys
import pandas as pd

DEFAULT_LOGS = "logs.csv"

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

if __name__ == "__main__":
    print(get_data().info())
    print(get_data())