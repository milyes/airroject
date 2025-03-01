
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import os

class ModeleIAAutomatique:
    def __init__(self, nom="modele_ia"):
        self.nom = nom
        self.modele = None
        self.scaler = StandardScaler()
        print(f"Module IA Automatique '{self.nom}' initialisé")
    
    def preparer_donnees(self, chemin_donnees, cible, test_size=0.2):
        """Charge et prépare les données pour l'entraînement"""
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(chemin_donnees):
                print(f"Erreur: Le fichier {chemin_donnees} n'existe pas")
                return None
            
            # Déterminer le type de fichier
            extension = os.path.splitext(chemin_donnees)[1].lower()
            
            if extension == '.csv':
                donnees = pd.read_csv(chemin_donnees)
            elif extension == '.xlsx' or extension == '.xls':
                donnees = pd.read_excel(chemin_donnees)
            else:
                print(f"Format de fichier non pris en charge: {extension}")
                return None
            
            if cible not in donnees.columns:
                print(f"Erreur: La colonne cible '{cible}' n'existe pas dans les données")
                return None
            
            # Séparer features et cible
            X = donnees.drop(cible, axis=1)
            y = donnees[cible]
            
            # Gérer les valeurs manquantes
            X = X.fillna(X.mean())
            
            # Normaliser les données
            X_scaled = self.scaler.fit_transform(X)
            
            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            print(f"Données préparées: {X_train.shape[0]} exemples d'entraînement, {X_test.shape[0]} exemples de test")
            return X_train, X_test, y_train, y_test, X.columns
        
        except Exception as e:
            print(f"Erreur lors de la préparation des données: {str(e)}")
            return None
    
    def entrainer_modele_foret(self, X_train, y_train, n_estimators=100):
        """Entraîne un modèle Random Forest"""
        try:
            modele = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            modele.fit(X_train, y_train)
            self.modele = modele
            print(f"Modèle de forêt aléatoire entraîné avec {n_estimators} arbres")
            return True
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle: {str(e)}")
            return False
    
    def entrainer_modele_tensorflow(self, X_train, y_train, epochs=10):
        """Entraîne un modèle simple TensorFlow"""
        try:
            # Déterminer si classification ou régression
            if len(np.unique(y_train)) <= 10:  # Classification avec peu de classes
                units_output = len(np.unique(y_train))
                activation_output = 'softmax'
                loss = 'sparse_categorical_crossentropy'
            else:  # Régression ou classification avec beaucoup de classes
                units_output = 1
                activation_output = 'linear'
                loss = 'mse'
            
            # Créer un modèle séquentiel simple
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(units_output, activation=activation_output)
            ])
            
            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            
            # Entraîner le modèle
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                validation_split=0.2,
                verbose=1
            )
            
            self.modele = model
            print(f"Modèle TensorFlow entraîné sur {epochs} époques")
            return history
        except Exception as e:
            print(f"Erreur lors de l'entraînement du modèle TensorFlow: {str(e)}")
            return None
    
    def evaluer_modele(self, X_test, y_test):
        """Évalue les performances du modèle"""
        if self.modele is None:
            print("Erreur: Aucun modèle n'a été entraîné")
            return None
        
        try:
            # Vérifier si c'est un modèle sklearn ou tensorflow
            if isinstance(self.modele, tf.keras.Model):
                # Modèle TensorFlow
                evaluation = self.modele.evaluate(X_test, y_test, verbose=0)
                print(f"Performances du modèle TensorFlow - Loss: {evaluation[0]:.4f}, Accuracy: {evaluation[1]:.4f}")
                return evaluation
            else:
                # Modèle sklearn
                y_pred = self.modele.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Précision du modèle: {accuracy:.4f}")
                print("\nRapport de classification:")
                print(classification_report(y_test, y_pred))
                return accuracy
        except Exception as e:
            print(f"Erreur lors de l'évaluation du modèle: {str(e)}")
            return None
    
    def sauvegarder_modele(self):
        """Sauvegarde le modèle entraîné"""
        if self.modele is None:
            print("Erreur: Aucun modèle n'a été entraîné")
            return False
        
        try:
            # Créer le dossier models s'il n'existe pas
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # Sauvegarder selon le type de modèle
            if isinstance(self.modele, tf.keras.Model):
                # Modèle TensorFlow
                self.modele.save(f"models/{self.nom}_tf")
                print(f"Modèle TensorFlow sauvegardé dans models/{self.nom}_tf")
            else:
                # Modèle sklearn
                import joblib
                joblib.dump(self.modele, f"models/{self.nom}.joblib")
                # Sauvegarder aussi le scaler
                joblib.dump(self.scaler, f"models/{self.nom}_scaler.joblib")
                print(f"Modèle sauvegardé dans models/{self.nom}.joblib")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            return False
    
    def charger_modele(self, modele_path):
        """Charge un modèle précédemment sauvegardé"""
        try:
            if os.path.isdir(modele_path):
                # C'est probablement un modèle TensorFlow
                self.modele = tf.keras.models.load_model(modele_path)
                print(f"Modèle TensorFlow chargé depuis {modele_path}")
            else:
                # C'est probablement un modèle sklearn
                import joblib
                self.modele = joblib.load(modele_path)
                
                # Essayer de charger le scaler associé
                scaler_path = modele_path.replace('.joblib', '_scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                
                print(f"Modèle chargé depuis {modele_path}")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            return False
    
    def predire(self, donnees):
        """Fait des prédictions avec le modèle entraîné"""
        if self.modele is None:
            print("Erreur: Aucun modèle n'a été entraîné ou chargé")
            return None
        
        try:
            # Prétraiter les données
            if isinstance(donnees, pd.DataFrame):
                # Utiliser le scaler pour normaliser
                donnees_scaled = self.scaler.transform(donnees)
            elif isinstance(donnees, np.ndarray):
                donnees_scaled = self.scaler.transform(donnees)
            else:
                print("Format de données non pris en charge. Utilisez pandas DataFrame ou numpy array")
                return None
            
            # Faire la prédiction
            predictions = self.modele.predict(donnees_scaled)
            return predictions
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            return None

def main():
    print("Module IA Automatique initialisé!")
    print("Pour utiliser ce module, créez une instance de la classe ModeleIAAutomatique")
    print("Exemple:")
    print("  modele = ModeleIAAutomatique('mon_modele')")
    print("  X_train, X_test, y_train, y_test, _ = modele.preparer_donnees('chemin/vers/donnees.csv', 'colonne_cible')")
    print("  modele.entrainer_modele_foret(X_train, y_train)")
    print("  modele.evaluer_modele(X_test, y_test)")
    print("  modele.sauvegarder_modele()")
    
    # Exemple de données synthétiques si aucune donnée réelle n'est disponible
    print("\nCréation d'un exemple avec données synthétiques...")
    from sklearn.datasets import make_classification
    
    # Générer des données synthétiques
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    
    # Convertir en DataFrame pour simulation
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Sauvegarder les données synthétiques
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/donnees_exemple.csv', index=False)
    
    print("Données d'exemple créées dans data/donnees_exemple.csv")
    print("\nVous pouvez maintenant tester le module avec:")
    print("  modele = ModeleIAAutomatique('test_model')")
    print("  X_train, X_test, y_train, y_test, _ = modele.preparer_donnees('data/donnees_exemple.csv', 'target')")
    print("  modele.entrainer_modele_foret(X_train, y_train)")
    print("  modele.evaluer_modele(X_test, y_test)")

if __name__ == "__main__":
    main()
