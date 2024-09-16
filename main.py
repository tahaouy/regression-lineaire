from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

np.random.seed(43)
data = pd.read_csv('Student_Performance.csv')
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].replace({'No': 0, 'Yes': 1})
data.drop_duplicates(inplace=True)
data = data.dropna(axis=0)
msk = np.random.rand(len(data)) < 0.8
train = data[msk]
test = data[~msk]

#premier cas avec les variables Hours_Studied, Sleep_Hours, Extracurricular_Activities
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['Hours_Studied', 'Sleep_Hours', 'Extracurricular_Activities']]) 
train_y = np.asanyarray(train[['Performance_Index']])
test_x = np.asanyarray(test[['Hours_Studied', 'Sleep_Hours', 'Extracurricular_Activities']])
test_y = np.asanyarray(test[['Performance_Index']])
regr.fit(train_x, train_y)
test_y_ = regr.predict(test_x)
mae = np.mean(np.absolute(test_y_ - test_y))
r_squared = r2_score(test_y, test_y_)
rmse = np.sqrt(mean_squared_error(test_y, test_y_))
print("Erreur absolue: %.2f" % mae)
print("R²: %.2f" % r_squared)
print("RMSE: %.2f" % rmse)
#on remarque que le modèle est vraiment pas bon car ca donne des mauvais resultats R²: 0.14 qui doit etre proche de 1 et l'ecart entre la prediction et la valeur reel est tres eleve RMSE: 17.75 / Erreur absolue: 15.39

print("-------------------------------------------------")

#deuxieme cas avec les variables : Sleep_Hours, Sample_Question_Papers_Practiced, Previous_Scores
rain_x = np.asanyarray(train[['Sleep_Hours', 'Sample_Question_Papers_Practiced', 'Previous_Scores']])
train_y = np.asanyarray(train[['Performance_Index']])
test_x = np.asanyarray(test[['Sleep_Hours', 'Sample_Question_Papers_Practiced', 'Previous_Scores']])
test_y = np.asanyarray(test[['Performance_Index']])
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)
test_y_ = regr.predict(test_x)
mae = np.mean(np.absolute(test_y_ - test_y))
r_squared = r2_score(test_y, test_y_)
rmse = np.sqrt(mean_squared_error(test_y, test_y_))
print("Erreur absolue: %.2f" % mae)
print("R²: %.2f" % r_squared)
print("RMSE: %.2f" % rmse)
#ce modèle est pire que le premier car la valeur de R² est de -9.29 et l'ecart entre la prediction et la valeur reel est tres eleve RMSE: 61.28 / Erreur absolue: 60.59

print("-------------------------------------------------")

#troisieme cas avec toutes les variables
train_x = np.asanyarray(train[['Hours_Studied', 'Sleep_Hours', 'Extracurricular_Activities', 'Sample_Question_Papers_Practiced', 'Previous_Scores']])
train_y = np.asanyarray(train[['Performance_Index']])
test_x = np.asanyarray(test[['Hours_Studied', 'Sleep_Hours', 'Extracurricular_Activities', 'Sample_Question_Papers_Practiced', 'Previous_Scores']])
test_y = np.asanyarray(test[['Performance_Index']])
regr.fit(train_x, train_y)
test_y_ = regr.predict(test_x)
mae = np.mean(np.absolute(test_y_ - test_y))
r_squared = r2_score(test_y, test_y_)
rmse = np.sqrt(mean_squared_error(test_y, test_y_))
print("Erreur absolue: %.2f" % mae)
print("R²: %.2f" % r_squared)
print("RMSE: %.2f" % rmse)
#la valeur de R² est de 0.99 et l'ecart entre la prediction et la valeur reel est tres faible RMSE: 1.99 / Erreur absolue: 1.58 ce modèle est meilleur que le premier ainsi que le deuxieme et est beaucoup plus fiable

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Le troisième modèle qui utilise toutes les variables disponibles est de loin le plus fiable et précis pour prédire l’indice de performance des étudiants. Les autres avec moins de variables ne parviennent pas à capturer suffisamment d’informations pour faire des prédictions précises. Cela montre l’importance d’inclure un ensemble complet de variables pertinentes pour obtenir des résultats de prédiction plus précis mais ca ne veut pas toujours dire que plus il y'en a mieux c'est; qualité >>> quantité.
