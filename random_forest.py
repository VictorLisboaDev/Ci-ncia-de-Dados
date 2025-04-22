import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ------------------ 1. Carregamento e Pré-processamento dos Dados ------------------
# Supondo que o arquivo CSV se chame 'dados_pacientes.csv'
df = pd.read_csv('dados_pacientes.csv')

# Verificando valores ausentes
df.fillna(df.median(), inplace=True)  # Substitui valores ausentes pela mediana

# Separando features (X) e variável alvo (y)
X = df.drop(columns=['condicao_medica'])  # Supondo que essa seja a variável alvo
y = df['condicao_medica']

# Normalização (se necessário)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ------------------ 2. Divisão do Conjunto de Dados ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------ 3. Treinamento do Modelo Random Forest ------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# ------------------ 4. Avaliação do Modelo ------------------
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba))

# ------------------ 5. Ajuste de Hiperparâmetros ------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Melhores parâmetros:", grid_search.best_params_)

# Treinando o modelo com os melhores parâmetros
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Avaliação do modelo otimizado
y_pred_best = best_rf.predict(X_test)
print("Acurácia após ajuste de hiperparâmetros:", accuracy_score(y_test, y_pred_best))

# ------------------ 6. Validação Cruzada ------------------
scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print("Validação Cruzada - Acurácia Média:", np.mean(scores))