# Description: Ce fichier contient les fonctions pour créer les features à partir des logs
import pandas as pd

#features général
def nb_actions(df_logs:pd.DataFrame):
    """
    Calculer le nombre d'actions réalisées par chaque personne
    :param df_logs: dataset anonymisé avec les logs
    :return: DataFrame pseudo -> nb
    """
    res = df_logs.groupby('pseudo').size().reset_index(name='nb_actions')
    return res

#features concernant les jours
def moyenne_actions_par_jour(df_logs:pd.DataFrame):
    """
    Calcule le nombre d'actions par jour
    :param df_logs: DataFrame anonymisé avec les logs
    :return:
    """
    res = df_logs.groupby(['pseudo', 'jour']).size().reset_index(name='nb')
    res = res.groupby('pseudo')['nb'].mean().reset_index(name='moyenne_nb_actions')
    return res

def max_actions_par_jour(df_logs:pd.DataFrame):
    """
    Calcule le nombre d'actions par jour
    :param df_logs: DataFrame anonymisé avec les logs
    :return: DataFrame pseudo -> max
    """
    res = df_logs.groupby(['pseudo', 'jour']).size().reset_index(name='nb')
    res = res.groupby('pseudo')['nb'].max().reset_index(name='max_nb_actions')
    return res

def variabilite_activite(df_logs:pd.DataFrame):
    """
    Calcule la variabilité du nombre d'actions par jour, utilisant l'écart-type
    :param df_logs: DataFrame anonymisé avec les logs
    :return: DataFrame pseudo -> var
    """
    res = df_logs.groupby(['pseudo', 'jour']).size().reset_index(name='nb')
    res = res.groupby('pseudo')['nb'].std().reset_index(name='std_actions_par_jour')
    return res

def nb_jours_avec_action(df_logs:pd.DataFrame):
    """
    Calcule le nombre de jours uniques où chaque pseudo a effectué une action.
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame avec le nombre de jours avec action par pseudo
    """
    res = df_logs.groupby('pseudo')['jour'].nunique().reset_index(name='nb_jours_avec_action')
    return res

def tempsdiff(df_logs:pd.DataFrame):
    """
    Calcule le nombre de jours entre le premier jour et le dernier jour avec une action
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame pseudo -> tempsdiff
    """
    res = df_logs.groupby('pseudo')['jour'].agg(lambda x: (x.max() - x.min()).days).reset_index(name='tempsdiff_jours')
    return res

def constance_activite(df_logs:pd.DataFrame):
    """
    Calcule constance d'activite (le nombre de jour avec activité divisé par le nombre de jours entre le premier jour et le dernier jour avec activité)
    :param df_logs: DataFrame contenant les logs
    :return: dataframe pseudo -> constance
    """
    nb_jours_avec_activite = nb_jours_avec_action(df_logs)
    difftemps = tempsdiff(df_logs)
    res = nb_jours_avec_activite.merge(difftemps, on='pseudo')
    res['constance_activite'] = res['nb_jours_avec_action'] / res['tempsdiff_jours'].replace(0, 1)
    return res[['pseudo', 'constance_activite']]

def semaine_vs_weekend(df_logs:pd.DataFrame):
    """
    Calcule le pourcentage d'activité pendant le weekend
    :param df_logs: Dataframe contenant les logs
    :return: Dataframe pseudo -> pourcentage d'activite pendant le weekend
    """
    df_logs['jour_semaine'] = pd.to_datetime(df_logs['jour']).dt.weekday  # lundi=0, dimanche=6
    df_logs['is_weekend'] = df_logs['jour_semaine'] >= 5
    res = df_logs.groupby('pseudo')['is_weekend'].mean().reset_index(name='pct_weekend')
    return res

#features avec heures
def periode_moyen_activite(df_logs:pd.DataFrame):
    """
    Calcule la période moyenne d'activite par jour en minutes
    :param df_logs: dataframe contenant les logs
    :return: dataframe pseudo -> période moyenne d'activté par jour en minutes
    """
    periode = df_logs.groupby(['pseudo', 'jour'])['heure'].agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
    res = periode.groupby('pseudo').mean().reset_index(name='activite_moyenne_par_jour_min')
    return res

def pourcentage_nuit(df_logs:pd.DataFrame):
    """
    Calcule la pourcentage d'activité pendant la nuit (entre 22h le soir et 7h du matin)
    :param df_logs: dataframe contenant les logs
    :return: DataFrame pseudo -> pourcentage d'activité la nuit
    """
    df_logs['heure_seule'] = df_logs["heure"].dt.hour
    df_logs['pourcentage_nuit'] = (df_logs['heure_seule'] < 7) | (df_logs['heure_seule'] >= 22)
    res = df_logs.groupby('pseudo')['pourcentage_nuit'].mean().reset_index(name='pourcentage_activite_nuit')
    return res

def pourcentage_matin(df_logs:pd.DataFrame):
    """
    Calcule la pourcentage d'activité le matin (entre 7h et 13h)
    :param df_logs: dataframe contenant les logs
    :return: DataFrame pseudo -> pourcentage d'activité le matin
    """
    df_logs['heure_seule'] = df_logs["heure"].dt.hour
    df_logs['pourcentage_matin'] = (df_logs['heure_seule'] >= 7) & (df_logs['heure_seule'] < 13)
    res = df_logs.groupby('pseudo')['pourcentage_matin'].mean().reset_index(name='pourcentage_activite_matin')
    return res

def pourcentage_aprem(df_logs:pd.DataFrame):
    """
    Calcule la pourcentage d'activité l'après-midi (entre 13h et 18h)
    :param df_logs: dataframe contenant les logs
    :return: DataFrame pseudo -> pourcentage d'activité l'après-midi
    """
    df_logs['heure_seule'] = df_logs["heure"].dt.hour
    df_logs['pourcentage_aprem'] = (df_logs['heure_seule'] >= 13) & (df_logs['heure_seule'] < 18)
    res = df_logs.groupby('pseudo')['pourcentage_aprem'].mean().reset_index(name='pourcentage_activite_aprem')
    return res

def pourcentage_soir(df_logs:pd.DataFrame):
    """
    Calcule la pourcentage d'activité le soir (entre 18h et 22h)
    :param df_logs: dataframe contenant les logs
    :return: DataFrame pseudo -> pourcentage d'activité le soir
    """
    df_logs['heure_seule'] = df_logs["heure"].dt.hour
    df_logs['pourcentage_soir'] = (df_logs['heure_seule'] >= 18) & (df_logs['heure_seule'] < 22)
    res = df_logs.groupby('pseudo')['pourcentage_soir'].mean().reset_index(name='pourcentage_activite_soir')
    return res

#features utilisant le composant
def nb_composant(df_logs:pd.DataFrame):
    """
    Calcule le nombre de composants différentes
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame pseudo -> nombre de composants
    """
    res = df_logs.groupby('pseudo')['composant'].nunique().reset_index(name='nb_composant')
    return res

def nb_chaque_composant(df_logs:pd.DataFrame):
    """
    Calcule le nombre d'activités de chaque composant
    :param df_logs: dataframe contenant les logs
    :return: dataframe avec pseudo et le nombre d'activités pour chaque composant
    """
    res = pd.crosstab(df_logs['pseudo'], df_logs['composant']).reset_index()
    res.columns = ['pseudo'] + [f"composant_{col}" for col in res.columns if col != 'pseudo']
    res = res.fillna(0)
    return res

def top_composant(df_logs:pd.DataFrame):
    """
    Calcule le composant le plus utilisé par pseudo
    :param df_logs: dateframe contenant les logs
    :return: dataframe pseudo -> le composant le plus utilisé
    """
    top_composant = df_logs.groupby(['pseudo', 'composant']).size().reset_index(name='count')
    res = top_composant.loc[top_composant.groupby('pseudo')['count'].idxmax(), ['pseudo', 'composant']]
    res = res.rename(columns={'composant': 'top_composant'})
    return res

#features utilisant le contexte générale
def nb_contexte_gen(df_logs:pd.DataFrame):
    """
    Calcule le nombre de contexte générale différentes
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame pseudo -> nombre de contextes
    """
    res = df_logs.groupby('pseudo')['contexte_general'].nunique().reset_index(name='nb_contexte')
    return res

def nb_specifications(df_logs:pd.DataFrame):
    """
    Calcule le nombre de specifications de contexte différentes
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame pseudo -> nombre de specifications de contexte différentes
    """
    res = df_logs.groupby('pseudo')['specification'].nunique().reset_index(name='nb_specification')
    return res

def nb_chaque_contexte(df_logs:pd.DataFrame):
    """
    Calcule le nombre d'activités de chaque pseudo pour chaque contexte général.
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame avec pseudo et le nombre d'activités pour chaque contexte
    """
    res = pd.crosstab(df_logs['pseudo'], df_logs['contexte_general']).reset_index()
    res.columns = ['pseudo'] + [f"contexte_{col}" for col in res.columns if col != 'pseudo']
    res = res.fillna(0)
    return res

def top_contexte(df_logs:pd.DataFrame):
    """
    Calcule le contexte le plus utilisé par pseudo
    :param df_logs: dateframe contenant les logs
    :return: dataframe pseudo -> le contextes le plus utilisé
    """
    top_contexte = df_logs.groupby(['pseudo', 'contexte_general']).size().reset_index(name='count')
    res = top_contexte.loc[top_contexte.groupby('pseudo')['count'].idxmax(), ['pseudo', 'contexte_general']]
    res = res.rename(columns={'contexte_general': 'top_contexte'})
    return res

#features utilisant l'evenement
def nb_evenement(df_logs:pd.DataFrame):
    """
    Calcule le nombre de evenements différentes
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame pseudo -> nombre de evenements différents
    """
    res = df_logs.groupby('pseudo')['evenement'].nunique().reset_index(name='nb_evenement')
    return res

def nb_chaque_evenement(df_logs:pd.DataFrame):
    """
    Calcule le nombre d'activités de chaque pseudo pour chaque evenement.
    :param df_logs: DataFrame contenant les logs
    :return: DataFrame avec pseudo et le nombre d'activités pour chaque evenement
    """
    res = pd.crosstab(df_logs['pseudo'], df_logs['evenement']).reset_index()
    res.columns = ['pseudo'] + [f"evenement_{col}" for col in res.columns if col != 'pseudo']
    res = res.fillna(0)
    return res

def top_evenement(df_logs:pd.DataFrame):
    """
    Calcule l'evenement le plus utilisé par pseudo
    :param df_logs: dateframe contenant les logs
    :return: dataframe pseudo -> l'evenement le plus utilisé
    """
    top_evenement = df_logs.groupby(['pseudo', 'evenement']).size().reset_index(name='count')
    res = top_evenement.loc[top_evenement.groupby('pseudo')['count'].idxmax(), ['pseudo', 'evenement']]
    res = res.rename(columns={'evenement': 'top_evenement'})
    return res
