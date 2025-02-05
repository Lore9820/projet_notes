import logging
import pandas as pd

def nb_actions(df_logs:pd.DataFrame):
    '''
    Calculer le nombre d'actions réalisées par chaque personne
    :param data: dataset anonymisé avec les logs
    :return: DataFrame pseudo -> nb
    '''
    res = df_logs.groupby('pseudo').size().reset_index(name='nb')
    return res

def moyenne_actions_par_jour(df_logs:pd.DataFrame):
    '''
    Calcule le nombre d'actions par jour
    :param df_logs: DataFrame anonymisé avec les logs
    :return:
    '''
    res = df_logs.groupby(['pseudo', 'jour']).size().reset_index(name='nb')
    res = res.groupby('pseudo')['nb'].mean().reset_index(name='moyenne_nb')
    return res

def max_actions_par_jour(df_logs:pd.DataFrame):
    '''
    Calcule le nombre d'actions par jour
    :param df_logs: DataFrame anonymisé avec les logs
    :return:
    '''
    res = df_logs.groupby(['pseudo', 'jour']).size().reset_index(name='nb')
    res = res.groupby('pseudo')['nb'].max().reset_index(name='max_nb')
    return res

def nb_jours_avec_action(df_logs:pd.DataFrame):
    """
    Calcule le nombre de jours uniques où chaque pseudo a effectué une action.

    :param df_logs: DataFrame contenant les logs
    :return: DataFrame avec le nombre de jours avec action par pseudo
    """
    res = df_logs.groupby('pseudo')['jour'].nunique().reset_index(name='nb_jours_avec_action')
    return res

if __name__ == '__main__':
    import modele

    #test1
    logs = modele.get_logs()
    notes = modele.get_notes()
    logs = modele.filter_logs(logs, notes)
    logs = modele.split_columns(logs)

    print(nb_actions(logs))
    print(moyenne_actions_par_jour(logs))
    print(max_actions_par_jour(logs))
    print(nb_jours_avec_action(logs))
