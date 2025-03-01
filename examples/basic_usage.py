
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ModeleIAAutomatique

def example_workflow():
    """Exemple d'utilisation du ModeleIAAutomatique avec des données d'exemple"""
    
    # Créer une instance du modèle
    modele = ModeleIAAutomatique("modele_exemple")
    
    # Charger et préparer les données (utiliser les données d'exemple générées par main.py)
    print("Chargement et préparation des données...")
    X_train, X_test, y_train, y_test, features = modele.preparer_donnees(
        'data/donnees_exemple.csv', 'target')
    
    if X_train is not None:
        # Entraîner un modèle Random Forest
        print("\nEntraînement d'un modèle Random Forest...")
        modele.entrainer_modele_foret(X_train, y_train, n_estimators=100)
        
        # Évaluer le modèle
        print("\nÉvaluation du modèle Random Forest:")
        modele.evaluer_modele(X_test, y_test)
        
        # Sauvegarder le modèle
        print("\nSauvegarde du modèle...")
        modele.sauvegarder_modele()
        
        # Exemple d'utilisation du modèle pour des prédictions
        print("\nExemple de prédiction:")
        # Utiliser les 5 premiers exemples du jeu de test
        predictions = modele.predire(X_test[:5])
        print(f"Prédictions: {predictions}")
        print(f"Valeurs réelles: {y_test[:5].values}")
        
        print("\nEntraînement d'un modèle TensorFlow...")
        # Entraîner un modèle TensorFlow (avec moins d'époques pour l'exemple)
        history = modele.entrainer_modele_tensorflow(X_train, y_train, epochs=5)
        
        # Évaluer le modèle TensorFlow
        print("\nÉvaluation du modèle TensorFlow:")
        modele.evaluer_modele(X_test, y_test)
        
        print("\nExemple terminé avec succès!")
    else:
        print("Impossible de continuer en raison d'erreurs lors de la préparation des données.")

if __name__ == "__main__":
    # Vérifier si le dossier examples existe, sinon le créer
    if not os.path.exists('examples'):
        os.makedirs('examples')
        
    # Vérifier si les données d'exemple existent, sinon exécuter main()
    if not os.path.exists('data/donnees_exemple.csv'):
        print("Génération des données d'exemple...")
        from main import main
        main()
    
    # Exécuter l'exemple
    example_workflow()
