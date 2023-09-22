import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Carregando o conjunto de dados
wine = load_wine()
x=wine.data
y=wine.target

# Definindo Parametros folds e seed
num_folds = 5
seed = 42
np.random.seed(seed)

# Crie um objeto KFold para dividir os dados em folds
kf = KFold(n_splits=num_folds)

# Crie um modelo de Floresta Aleatória
model = RandomForestClassifier(n_estimators=50, max_depth=1, min_samples_leaf=2, random_state=seed)

# Normalizar os dados
scaler = StandardScaler()
x_normalized = scaler.fit_transform(x)

# Escolher duas características específicas
feature1_index = 0  # Índice da primeira característica
feature2_index = 11  # Índice da segunda característica

# Características: 
# 1) Alcohol - Álcool
# 2) Malic acid - Ácido málico
# 3) Ash - Cinzas
# 4) Alcalinity of ash - Alcalinidade das cinzas
# 5) Magnesium - Magnésio
# 6) Total phenols - Fenóis totais
# 7) Flavanoids - Flavonóides
# 8) Nonflavanoid phenols - Fenóis não flavonóides
# 9) Proanthocyanins - Proantocianinas
# 10) Color intensity - Intensidade de cor
# 11) Hue - Matiz
# 12) OD280/OD315 of diluted wines - OD280/OD315 de vinhos diluídos
# 13) Proline - Prolina

# Criando um DataFrame com os dados
wine_df = pd.DataFrame(data=x, columns=wine.feature_names)
# Adicionando a coluna de rótulos 'target' ao DataFrame
wine_df['target'] = y

#wine_df = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])

# Exibindo as primeiras linhas do DataFrame
print(wine_df.head(10))

# Dividir os dados em conjuntos de treinamento e teste (por exemplo, 80% treinamento, 20% teste)
x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=seed, stratify=y)

model.fit(x_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(x_test)

from sklearn.tree import plot_tree

# Plotando uma árvore de decisão individual da Floresta Aleatória
plt.figure(figsize=(15, 10))
plot_tree(model.estimators_[0], feature_names=wine.feature_names, class_names=[str(i) for i in wine.target_names], filled=True, rounded=True, fontsize=6)
plt.show()

# Calcular a precisão
accuracy = accuracy_score(y_test, y_pred)
print("Precisao no Conjunto de Teste:", accuracy)

import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# Plotar um gráfico de dispersão das duas características com cores representando as classes
plt.figure(figsize=(10, 6))
plt.scatter(wine_df[wine.feature_names[feature1_index]], wine_df[wine.feature_names[feature2_index]], c=wine_df['target'], cmap=plt.cm.Paired)
plt.xlabel(wine.feature_names[feature1_index])
plt.ylabel(wine.feature_names[feature2_index])
plt.title('Gráfico de Dispersão das Duas Características')
plt.show()

from mlxtend.plotting import plot_decision_regions

# Plotando as regiões de decisão
plt.figure(figsize=(10, 6))

# Criar um modelo de árvore de decisão que aceita apenas duas características
model_RandomForest = RandomForestClassifier(n_estimators=50, max_depth=1, min_samples_leaf=2, random_state=seed)

# Ajustar o modelo aos dados de treinamento com apenas as duas características selecionadas
model_RandomForest.fit(x_train[:, [feature1_index, feature2_index]], y_train)

# Use apenas as duas características selecionadas
x_test_selected = x_test[:, [feature1_index, feature2_index]]

# # Especifique os rótulos das classes para a legenda
# scatter_kwargs = {'s': 80, 'edgecolor': None, 'alpha': 0.7, 'label': 'Classe'}

plot_decision_regions(x_test_selected, y_test, clf=model_RandomForest)
plt.xlabel(wine.feature_names[feature1_index])
plt.ylabel(wine.feature_names[feature2_index])
plt.title('Fronteiras de Decisão')

# Adicionando a legenda
plt.legend(title="Wines")
plt.show()

from sklearn.model_selection import learning_curve

# Calculando as curvas de aprendizado
train_sizes, train_scores, test_scores = learning_curve(model, x_normalized, y, cv=5, scoring='accuracy', n_jobs=-1)

# Calculando as médias e desvios padrão das pontuações em treinamento e teste
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plotando as curvas de aprendizado
plt.figure(figsize=(10, 6))
plt.title("Curva de Aprendizado (Learning Curve)")
plt.xlabel("Tamanho do Conjunto de Treinamento")
plt.ylabel("Precisão")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treinamento")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Teste")

plt.legend(loc="best")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Binarize as classes (uma classe versus todas as outras)
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# Calculando as probabilidades previstas (necessárias para a Curva ROC) usando os dados de teste
y_prob = model.predict_proba(x_test)

# Calculando a Curva ROC e a pontuação AUC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotando as Curvas ROC
plt.figure(figsize=(8, 6))
colors = ['darkorange', 'navy', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Curva ROC Classe {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo (FPR)')
plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
plt.title('Curvas ROC por Classe')
plt.legend(loc='lower right')
plt.show()