
import sys
import os

# Ajouter le répertoire parent au chemin pour importer le module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ModeleIAAutomatique

def analyser_mon_dataset(chemin_donnees, colonne_cible):
    """
    Utilise le module IA Automatique pour analyser votre propre jeu de données
    
    Paramètres:
    - chemin_donnees: Chemin vers votre fichier CSV ou Excel
    - colonne_cible: Nom de la colonne cible à prédire
    """
    # Créer une instance du modèle
    modele = ModeleIAAutomatique("mon_analyse")
    
    # Charger et préparer les données
    print(f"Chargement des données depuis {chemin_donnees}...")
    X_train, X_test, y_train, y_test, features = modele.preparer_donnees(
        chemin_donnees, colonne_cible)
    
    if X_train is None:
        print("Erreur lors de la préparation des données. Vérifiez le chemin et le format du fichier.")
        return
    
    print(f"Variables utilisées comme features: {', '.join(features)}")
    
    # Entraîner les modèles
    reponse = input("\nSouhaitez-vous entraîner un modèle Random Forest? (o/n): ")
    if reponse.lower() == 'o':
        n_estimators = int(input("Nombre d'arbres (par exemple 100): ") or "100")
        print(f"\nEntraînement du modèle Random Forest avec {n_estimators} arbres...")
        modele.entrainer_modele_foret(X_train, y_train, n_estimators=n_estimators)
        modele.evaluer_modele(X_test, y_test)
    
    reponse = input("\nSouhaitez-vous entraîner un modèle TensorFlow? (o/n): ")
    if reponse.lower() == 'o':
        epochs = int(input("Nombre d'époques (par exemple 10): ") or "10")
        print(f"\nEntraînement du modèle TensorFlow sur {epochs} époques...")
        modele.entrainer_modele_tensorflow(X_train, y_train, epochs=epochs)
        modele.evaluer_modele(X_test, y_test)
    
    # Sauvegarder le modèle
    reponse = input("\nSouhaitez-vous sauvegarder le modèle? (o/n): ")
    if reponse.lower() == 'o':
        modele.sauvegarder_modele()
        print("Modèle sauvegardé!")

if __name__ == "__main__":
    # Exemple d'utilisation
    print("=== UTILISATION DU MODULE IA AUTOMATIQUE AVEC VOS DONNÉES ===")
    print("Ce script vous permet d'analyser votre propre jeu de données.")
    print("Assurez-vous que votre fichier est au format CSV ou Excel et qu'il contient une colonne cible.")
    
    # Soit utiliser les données d'exemple, soit vos propres données
    choix = input("\nSouhaitez-vous utiliser les données d'exemple? (o/n): ")
    
    if choix.lower() == 'o':
        # Vérifier si les données d'exemple existent, sinon les créer
        if not os.path.exists('data/donnees_exemple.csv'):
            print("Génération des données d'exemple...")
            from main import main
            main()
        analyser_mon_dataset('data/donnees_exemple.csv', 'target')
    else:
        chemin = input("Entrez le chemin vers votre fichier de données: ")
        cible = input("Entrez le nom de la colonne cible: ")
        analyser_mon_dataset(chemin, cible)
