import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Recabar la Height y Weight de 50 amigos
df = pd.read_csv('datos-amigos.csv')

# Etiquetar los datos según el género
df['Class'] = ['male', 'female'] * 25

# Dividir los datos en conjunto de entrenamiento y prueba
df_train = df[:40]
df_test = df[40:]

# Especificar los datos y etiquetas de entrenamiento
X_train = df_train[['Height', 'Weight']]
y_train = df_train['Class']

# Visualizar gráfica de dispersión de entrenamiento
ax_train = plot.axes()
ax_train.scatter(df_train.loc[df_train['Class'] == 'male', 'Height'],
                 df_train.loc[df_train['Class'] == 'male', 'Weight'],
                 c="blue",
                 label="male")
ax_train.scatter(df_train.loc[df_train['Class'] == 'female', 'Height'],
                 df_train.loc[df_train['Class'] == 'female', 'Weight'],
                 c="pink",
                 label="female")
plot.xlabel("Height")
plot.ylabel("Weight")
ax_train.legend()

# Especificar los datos y etiquetas de prueba
X_test = df_test[['Height', 'Weight']]
y_test = df_test['Class']

# Calcular el valor de k
k = int(np.sqrt(X_train.shape[0]))

if k % 2 == 0:  # Hacer que k sea impar para evitar empates
    k += 1

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

# Probar con cada muestra de prueba
for i in range(len(X_test)):
    dfp = pd.DataFrame()
    dfp['Height'] = [X_test.iloc[i]['Height']]
    dfp['Weight'] = [X_test.iloc[i]['Weight']]

    ax_test = plot.axes()
    ax_test.scatter(df_train.loc[df_train['Class'] == 'male', 'Height'],
                    df_train.loc[df_train['Class'] == 'male', 'Weight'],
                    c="blue",
                    label="male")
    ax_test.scatter(df_train.loc[df_train['Class'] == 'female', 'Height'],
                    df_train.loc[df_train['Class'] == 'female', 'Weight'],
                    c="pink",
                    label="female")
    ax_test.scatter(dfp['Height'],
                    dfp['Weight'],
                    c="black")
    plot.xlabel("Height")
    plot.ylabel("Weight")
    ax_test.legend()

    prediccion = knn.predict(dfp)
    print('\nCon los datos:')
    print(dfp)
    print('La Class predicha es:')
    print(prediccion)

    plot.show()
    