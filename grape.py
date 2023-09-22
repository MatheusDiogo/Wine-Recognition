# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Carregar o conjunto de dados Wine do scikit-learn
wine = load_wine()
data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
target = pd.Series(data=wine.target, name='target')

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

# Padronizar as features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Realizar previsões
y_pred = knn.predict(X_test)

# Matriz de Confusão
# Calcular e exibir a matriz de confusão
confusion = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão')
plt.show()

from tabulate import tabulate
from sklearn.metrics import classification_report

# Calculando a precisão, recall e F1-score
classification_rep = classification_report(y_test, y_pred, target_names=wine.target_names, output_dict=True)

# Convertendo o relatório de classificação em uma tabela
table = []
for class_name in wine.target_names:
    metrics = classification_rep[class_name]
    table.append([class_name, metrics['precision'], metrics['recall'], metrics['f1-score']])

# Adicionando uma linha de cabeçalho à tabela
table.insert(0, ['Classe', 'Precisão', 'Recall', 'F1-Score'])

# Imprimindo a tabela
print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

# Curva ROC
y_score = knn.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(wine.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc[0]))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Curva de Aprendizagem
train_sizes, train_scores, test_scores = learning_curve(knn, X_train, y_train, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.title('Curva de Aprendizagem')
plt.xlabel('Tamanho do Conjunto de Treinamento')
plt.ylabel('Pontuação')
plt.grid()

plt.plot(train_sizes, train_scores_mean, 'o-', label='Pontuação no Treinamento')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Pontuação no Teste')
plt.legend(loc='best')
plt.show()

# Gráfico da Fronteira de Decisão (usando as duas primeiras características)
from mlxtend.plotting import plot_decision_regions

# Selecionar as duas primeiras características do conjunto de teste
X_test_2d = X_test[:, [0, 10]]

# Treinar o modelo KNN nas duas primeiras características
knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train[:, [0, 10]], y_train)

# Plotar a fronteira de decisão
plot_decision_regions(X_test_2d, y_test.values, clf=knn_2d, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Fronteira de Decisão (duas primeiras características)')
plt.show()