import modele
import control
import view


if __name__ == '__main__':
    logs_file = input("Saissisez le nom du fichier avec les logs (fichier csv, inclure l'extension) :")
    notes_file = input("Saissisez le nom du fichier avec les notes (fichier csv, inclure l'extension) :")

    logs = modele.get_logs(logs_file)
    notes = modele.get_notes(notes_file)
    logs = modele.filter_logs(logs, notes)
    logs = modele.split_columns(logs)
    notes = modele.filter_notes(notes, logs)

    df = control.creer_df(logs)

