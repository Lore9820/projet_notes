import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import numpy as np
import joblib

from models.read_files import *
from models.model import *
from controllers.preprocessing import *
from controllers.feature_selection import *
from controllers.features_creation import *

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Prédiction du note de l\'apprenant')
        self.root.geometry('600x400')
        
        # Création des widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Titre
        title_label = ttk.Label(main_frame, 
                              text='Prédiction du note de l\'apprenant',
                              font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Bouton pour choisir le fichier
        self.file_button = ttk.Button(main_frame, 
                                    text="Choisir un fichier de logs",
                                    command=self.load_file)
        self.file_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Label pour afficher le nom du fichier choisi
        self.file_label = ttk.Label(main_frame, text="Aucun fichier sélectionné")
        self.file_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Bouton pour lancer la prédiction
        self.predict_button = ttk.Button(main_frame, 
                                       text="Lancer la prédiction",
                                       command=self.make_prediction,
                                       state='disabled')
        self.predict_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Label pour afficher le résultat
        self.result_label = ttk.Label(main_frame, 
                                    text="",
                                    font=('Helvetica', 12))
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Fichiers CSV", "*.csv")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"Fichier sélectionné: {file_path.split('/')[-1]}")
            self.predict_button.config(state='normal')
            
    def make_prediction(self):
        try:
            # Charger les données
            logs = get_logs(self.file_path)
            
            # Transformer les données
            logs = split_columns(logs)
            df = creer_df(logs)
            df = df_transformer(df)
            print(df.head(5))
            
            # Charger le modèle et faire la prédiction
            adaboost = joblib.load('../models/ada_boost.pkl')
            prediction = adaboost.predict(df)[0]
            
            # Afficher le résultat
            self.result_label.config(
                text=f"La prédiction utilisant le modèle AdaBoost est: {prediction:.2f}",
                foreground='green'
            )
            
        except Exception as e:
            self.result_label.config(
                text=f"Erreur: {str(e)}",
                foreground='red'
            )

def main():
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()