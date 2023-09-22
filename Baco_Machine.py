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
model = RandomForestClassifier(n_estimators=50, max_depth=2, min_samples_leaf=2, random_state=seed)

# Definir o espaço de busca de hiperparâmetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

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
x_train0, x_test, y_train0, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=seed, stratify=y)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

# Criar um objeto GridSearchCV para otimizar os hiperparâmetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf)
grid_search.fit(x_train0, y_train0)

# Obter os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_
print("Melhores Hiperparâmetros:", best_params)

# Obter o modelo treinado com os melhores hiperparâmetros
model = grid_search.best_estimator_

# Realize a validação cruzada
scores = cross_val_score(model, x_train0, y_train0, cv=kf)

# Exiba as pontuações de desempenho em cada fold
print("Pontuacoes de Desempenho em cada Fold:", scores)

# Calcule e exiba a média das pontuações
print("Media das Pontuacoes:", np.mean(scores))

# Escolha a divisão com melhor desempenho médio
best_split = scores.argmax()

# Obtenha os índices dos dados para a melhor divisão
best_train_index, best_test_index = list(kf.split(x_normalized))[best_split]

# Divida novamente os dados apenas para treinamento usando a melhor divisão
x_train, y_train = x_normalized[best_train_index], y[best_train_index]

unique_classes_train = np.unique(y_train)
unique_classes_test = np.unique(y_test)

print("Classes únicas em y_train:", unique_classes_train)
print("Classes únicas em y_test:", unique_classes_test)

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
model_RandomForest = RandomForestClassifier(random_state=seed)

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

# Teste com novo conjunto de dados
new_wine_data = np.array([[13.24, 2.59], [12.64, 3.36], [14.06, 2.15]])  # Exemplo de novos dados

# Criar um novo StandardScaler apenas para as 2 características
scaler_new = StandardScaler()
scaler_new.fit(x[:, [feature1_index, feature2_index]])  # Ajuste apenas para as 2 características originais

# Normalizar os novos dados usando o novo scaler
new_x_normalized = scaler_new.transform(new_wine_data)

# Fazer previsões para as novas amostras
predicted_classes = model.predict(new_x_normalized)

# Obter as probabilidades associadas a cada classe
class_probabilities = model.predict_proba(new_x_normalized)

# Exibir a classe prevista e as probabilidades
for i, (predicted_class, probabilities) in enumerate(zip(predicted_classes, class_probabilities)):
    print(f"Amostra {i + 1}:")
    print(f"Classe prevista: {predicted_class} ({wine.target_names[predicted_class]})")
    for class_idx, prob in enumerate(probabilities):
        print(f"Probabilidade da Classe {class_idx} ({wine.target_names[class_idx]}): {prob:.2f}")
    print()

# Plotar um gráfico de dispersão das duas características com cores representando as classes
plt.figure(figsize=(10, 6))
plt.scatter(new_x_normalized[:, 0], new_x_normalized[:, 1], c=predicted_classes, cmap=plt.cm.Paired)
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])  # Usar o índice 1 para a segunda característica
plt.title('Gráfico de Dispersão das Duas Características para Novos Dados')
plt.show()

# Plotar as regiões de decisão
plt.figure(figsize=(10, 6))

# Converter os novos dados normalizados em um array NumPy
new_x_normalized_np = new_x_normalized

# Plotar as fronteiras de decisão com base nas duas características
plot_decision_regions(new_x_normalized_np, predicted_classes, clf=model)
plt.xlabel(wine.feature_names[0])
plt.ylabel(wine.feature_names[1])
plt.title('Fronteiras de Decisão para Novos Dados')
plt.legend(title="Wines")
plt.show()
