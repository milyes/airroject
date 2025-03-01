
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Ajouter le répertoire parent au chemin pour importer le module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ModeleIAAutomatique

def generer_donnees_moons():
    """Génère un jeu de données de classification en forme de lunes"""
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    df['target'] = y
    
    # Sauvegarder les données
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/donnees_moons.csv', index=False)
    print("Données 'moons' générées et sauvegardées dans data/donnees_moons.csv")
    return df

def visualiser_donnees(df):
    """Visualise les données générées"""
    plt.figure(figsize=(10, 6))
    plt.scatter(df['feature_1'], df['feature_2'], c=df['target'], cmap='viridis', 
                edgecolors='k', alpha=0.7)
    plt.title('Jeu de données "moons"')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Classe')
    plt.savefig('data/visualisation_moons.png')
    print("Visualisation sauvegardée dans data/visualisation_moons.png")

def comparer_modeles():
    """Compare les performances des modèles Random Forest et TensorFlow"""
    # Génération des données
    df = generer_donnees_moons()
    visualiser_donnees(df)
    
    # Créer une instance du modèle
    modele = ModeleIAAutomatique("comparaison_modeles")
    
    # Préparer les données
    print("\nPréparation des données...")
    X_train, X_test, y_train, y_test, features = modele.preparer_donnees(
        'data/donnees_moons.csv', 'target')
    
    if X_train is not None:
        # Entraîner et évaluer le modèle Random Forest
        print("\n--- MODÈLE RANDOM FOREST ---")
        modele.entrainer_modele_foret(X_train, y_train, n_estimators=100)
        rf_accuracy = modele.evaluer_modele(X_test, y_test)
        
        # Entraîner et évaluer le modèle TensorFlow
        print("\n--- MODÈLE TENSORFLOW ---")
        history = modele.entrainer_modele_tensorflow(X_train, y_train, epochs=15)
        tf_evaluation = modele.evaluer_modele(X_test, y_test)
        
        # Sauvegarder les modèles
        modele.sauvegarder_modele()
        
        # Test de prédiction
        print("\n--- TEST DE PRÉDICTION ---")
        # Créer quelques exemples de test
        exemples_test = pd.DataFrame({
            'feature_1': [0.5, -0.5, 0, 1.5],
            'feature_2': [0.5, 0.5, 0, 0]
        })
        
        # Faire des prédictions
        predictions = modele.predire(exemples_test)
        print(f"Données de test:\n{exemples_test}")
        print(f"Prédictions: {predictions}")
        
        print("\nComparaison terminée avec succès!")
    else:
        print("Impossible de continuer en raison d'erreurs lors de la préparation des données.")

if __name__ == "__main__":
    comparer_modeles()
