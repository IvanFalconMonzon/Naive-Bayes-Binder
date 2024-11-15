# Importamos las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


############################################
# 1. Preparación de datos #
############################################


# Cargamos el dataset; reemplazamos '?' por NaN para indicar valores faltantes
vote_data = pd.read_csv("house-votes-84.data", header=None, na_values='?')


# Asignamos nombres de columnas
column_names = ["NAME", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16"]
vote_data.columns = column_names


# Convertimos 'NAME' en valores numéricos: 'democrat' como 1 y 'republican' como 0
vote_data["NAME"] = vote_data["NAME"].apply(lambda x: 1 if x == 'democrat' else 0)


# Reemplazamos 'y' con 1, 'n' con 0 en las columnas de votación, y NaN con -1 para valores faltantes
vote_data.replace({'y': 1, 'n': 0}, inplace=True)
vote_data.fillna(-1, inplace=True)


##############################################
# 2. Creación de datos de entrenamiento/test #
##############################################


# Dividimos los datos en conjunto de entrenamiento y prueba (85% y 15%)
train_data, test_data = train_test_split(vote_data, test_size=0.15, random_state=42, stratify=vote_data["NAME"])


# Separamos las características (X) y el objetivo (y) en ambos conjuntos
X_train = train_data.drop(columns="NAME")
y_train = train_data["NAME"]
X_test = test_data.drop(columns="NAME")
y_test = test_data["NAME"]


##########################################
# 3. Entrenamiento del modelo #
##########################################


# Entrenamos un modelo Naive Bayes con datos numéricos usando GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)


# Realizamos predicciones en el conjunto de prueba
y_pred = nb_classifier.predict(X_test)


# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Republican", "Democrat"]).plot()
plt.title("Matriz de Confusión")
plt.show()


# Mostramos instancias donde el modelo se equivocó
print("Errores de predicción:")
print(test_data[y_test != y_pred])


##########################################
# 4. Curvas ROC y Precision-Recall #
##########################################
# Probabilidades de predicción
y_pred_proba = nb_classifier.predict_proba(X_test)[:, 1]  # Probabilidad de 'democrat'

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (área = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Tasa de Falsos Positivos")
plt.ylabel("Tasa de Verdaderos Positivos")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.show()

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, color="blue", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.show()
