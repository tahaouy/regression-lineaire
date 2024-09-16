
# Prédiction de la Performance des Étudiants

Je m'exerce à utiliser des modèles de régression linéaire pour prédire l’indice de performance des étudiants 

## Installation

Pour exécuter ce projet, vous aurez besoin d’installer les bibliothèques suivantes :


``pip install pandas numpy scikit-learn``


## Utilisation

1.  **Charger les données**  :
    
    -   Assurez-vous que le fichier  `Student_Performance.csv`  est dans le même répertoire que votre script Python.
    -   Charger les données et effectuer le prétraitement nécessaire.
2.  **Prétraitement des données**  :
    
    -   Remplacer les valeurs catégorielles par des valeurs numériques.
    -   Supprimer les doublons et les valeurs manquantes.
    -   Diviser les données en ensembles d’entraînement et de test.
3.  **Modèles de régression linéaire**  :
    
    -   Trois modèles sont testés :
        1.  Utilisant les variables  `Hours_Studied`,  `Sleep_Hours`,  `Extracurricular_Activities`.
        2.  Utilisant les variables  `Sleep_Hours`,  `Sample_Question_Papers_Practiced`,  `Previous_Scores`.
        3.  Utilisant toutes les variables disponibles.
4.  **Évaluation des modèles**  :
    
    -   Calculer l’erreur absolue moyenne (MAE), le coefficient de détermination (R²) et l’erreur quadratique moyenne (RMSE) pour chaque modèle.
    -   Comparer les performances des modèles.
