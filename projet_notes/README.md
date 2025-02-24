# Prédiction des Notes d'Apprenants
Ce projet permet de prédire les notes des apprenants à partir de leurs données de logs d'activité en utilisant des modèles de machine learning.

## Structure du Projet

- **models/** : Contient les fichiers liés aux modèles de machine learning.
- **controllers/** : Contient les fichiers liés aux prétraitements des données.
- **views/** : Contient les fichiers liés à l'interface graphique.

## Fonctionnalités

- **Prétraitements des données** :
    - Chargement des données et transformation en dataframe utilisable pour un modèle de machine learning (création de features)
    - Transformation des données en données utilisables pour un modèle de machine learning (encodage, normalisation, etc.)
    - Sélection des données pertinentes (sélection des features pertinentes utilisant forward feature selection et lasso selection)
- **Entraînement du modèle de machine learning** :
    - Entraînement de plusiers modèles sur les données d'entraînement (Régression linéaire, AdaBoost, RandomForest) et évaluation de leurs performances utilisant les métriques RMSE et R2 et plusieurs graphiques
- **Prédiction des notes** :
    - Utilisation d'un modèle de machine learning pour prédire les notes des apprenants à partir des logs d'activité
- **Interface graphique** :
    - Interface graphique permettant de charger les données, de faire les prétraitements et de faire la prédiction


## Installation

1. Cloner le projet :
    ```bash
    git clone https://github.com/Lore9820/projet_notes.git
    ```
2. Naviguer dans le répertoire du projet :
    ```bash
    cd projet_notes
    ```
3. Installer les dépendances :
    ```bash
    pip install pandas numpy scikit-learn joblib
    ```
4. Exécuter le projet :
    ```bash
    python views/view.py
    ```
5. Dans l'interface graphique :
   - Cliquez sur "Choisir un fichier de logs" pour sélectionner votre fichier CSV
   - Cliquez sur "Lancer la prédiction" pour obtenir la prédiction de la note

## Format des Données

Le fichier CSV d'entrée doit contenir les logs d'activité des apprenants avec les colonnes suivantes :
- timestamp : horodatage de l'activité
- evenement : type d'événement
- composant : composant concerné
- contexte : contexte de l'activité
!!! A finir encore !!! (vérifier que pseudo ne doit pas être présent et que autres colonnes sont ignorées)

## Modèle de Prédiction

Le projet utilise un modèle AdaBoost pour la prédiction des notes. Le modèle est entraîné sur les données historiques des apprenants et leurs notes finales.

## Auteurs

Lore Goethals

