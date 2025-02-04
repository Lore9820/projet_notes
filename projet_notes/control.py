import logging
import pandas as pd

def nb_actions(data:pd.DataFrame):
    '''
    Calculer le nombre d'actions réalisées par chaque apprenant
    :param data: dataset anonymisé
    :return: DataFrame pseudo -> nb
    '''
    res = data.groupby('pseudo').size().reset_index(name='nb')
    res = res.sort_values(by='nb', ascending=False)
    return res

if __name__ == '__main__':
    import modele

    #test1
    logs = modele.get_logs()
    pseudo_count = nb_actions(logs)
    print(pseudo_count)