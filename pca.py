# Bibliotecas generales
import pandas as pd # cargar y manipular datos
import sklearn
import matplotlib.pyplot as plt # graficas

# descomposition = reduccion de dimensionalidad
from sklearn.decomposition import KernelPCA # reduccion de dimensiones
# from sklearn.decomposition import IncrementalPCA # reduccion de dimensiones

#  clasificador lineal
from sklearn.linear_model import LogisticRegression # modelo de clasificación

# Escalador que todos esten en la misma escala de cero a uno
from sklearn.preprocessing import StandardScaler # normalizar datos
# Partir datos en pruebas y entrenamiento
from sklearn.model_selection import train_test_split # divide  el dataset en entrenamiento t test

 
if __name__ == "__main__":
  dt_heart = pd.read_csv('./data/heart.csv')

  # print(dt_heart.head(5))

  # borrar target
  dt_features = dt_heart.drop(['target'], axis=1)
  # asignar aqui
  dt_target = dt_heart['target']

  # Sobreescribir con los valores ya transformados
  dt_features = StandardScaler().fit_transform(dt_features)

  # partir el conjunto de entrenamiento
  # test_size = tamaño del conjunto de entrenamiento
  X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
  
  kpca = KernelPCA(n_components=4, kernel='poly')
  kpca.fit(X_train)

  dt_train = kpca.transform(X_train)
  dt_test = kpca.transform(X_test)

  logistic = LogisticRegression(solver='lbfgs')
  logistic.fit(dt_train, y_train)
  print('SCORE KPCA: ', logistic.score(dt_test, y_test))


  # print(X_train.shape)
  # print(y_train.shape)

  # pca = PCA(n_components=4)
  # pca.fit(X_train)

  # ipca = IncrementalPCA(n_components=3, batch_size=20)
  # ipca.fit(X_train) 

  # # Nos muestra la informacion valiosa
  # plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
  # plt.show()

  # # clasificacion con regresion logistica
  # logistic = LogisticRegression(solver='lbfgs')

  # dt_train = pca.transform(X_train)
  # dt_test = pca.transform(X_test)

  # logistic.fit(dt_train, y_train)
  # print('SCORE PCA: ', logistic.score(dt_test, y_test))

  # dt_train = ipca.transform(X_train)
  # dt_test = ipca.transform(X_test)
  # logistic.fit(dt_train, y_train)
  # print('SCORE IPCA: ', logistic.score(dt_test, y_test))
