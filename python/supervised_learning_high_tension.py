# Preparação dos dados
import os

import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_theme()


## Importar dataset teste

data_path = './data/' if os.path.exists(
    './data/') else 'https://raw.githubusercontent.com/kreativermario/Projeto-DECD/master/data/'

test_path = data_path + 'treated/prepared/numeric/no-dates/dataset-numeric-high-tensions-test-no-dates.csv'

test_df = pd.read_csv(test_path)

## Importar dataset treino
train_path = data_path + 'treated/prepared/numeric/no-dates/dataset-numeric-high-tensions-train-no-dates.csv'

train_df = pd.read_csv(train_path)

# Define regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0, solver='auto'),
    'Lasso': Lasso(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
    'k-NN': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'SVM': SVR(kernel='linear', max_iter=10000),
    'MLP': MLPRegressor(hidden_layer_sizes=(8,4), max_iter=20000),
}

# Define hyperparameters distributions for each regressor
param_dist = {
    'Linear Regression': {},
    'Ridge': {'alpha': uniform(0.1, 10)},
    'Lasso': {'alpha': uniform(0.1, 10)},
    'ElasticNet': {'alpha': uniform(0.1, 10), 'l1_ratio': uniform(0.1, 0.9)},
    'k-NN': {'n_neighbors': randint(3, 15)},
    'Decision Tree': {'max_depth': [None] + list(range(1, 31))},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None] + list(range(1, 11))},
    'SVM': {'C': uniform(0.1, 10)},
    'MLP': {'hidden_layer_sizes': [(8, 4), (16, 8), (32, 16)], 'max_iter': randint(10000, 20000)},
}

best_params = {}
best_scores = {}
evaluation_metrics = {}

for name, regressor in regressors.items():
    # Perform randomized search for each regressor
    random_search = RandomizedSearchCV(regressor, param_distributions=param_dist[name], n_iter=10, cv=3, scoring='r2', random_state=42)
    random_search.fit(train_df.drop(columns=['energia_ativa_alta_tensao_kwh']), train_df['energia_ativa_alta_tensao_kwh'])

    # Get best parameters and score
    best_params[name] = random_search.best_params_
    best_score = random_search.best_score_

    # Store best parameters and score
    best_scores[name] = best_score

    # Evaluate best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(test_df.drop(columns=['energia_ativa_alta_tensao_kwh']))
    mse = mean_squared_error(test_df['energia_ativa_alta_tensao_kwh'], y_pred)
    mae = mean_absolute_error(test_df['energia_ativa_alta_tensao_kwh'], y_pred)
    r2 = r2_score(test_df['energia_ativa_alta_tensao_kwh'], y_pred)

    # Store evaluation metrics
    evaluation_metrics[name] = {'R^2': r2, 'MSE': mse, 'MAE': mae}

    print(f'{name}:')
    print(f'  Best parameters: {best_params[name]}')
    print(f'  Best cross-validation score (R^2): {best_score}')
    print(f'  Mean Squared Error (MSE): {mse}')
    print(f'  Mean Absolute Error (MAE): {mae}')
    print(f'  R^2 (coefficient of determination): {r2}')
    print()

# Save best parameters, scores, and evaluation metrics to a file
with open('best_hyperparameters_and_scores_randomized_high_tensions.txt', 'w') as f:
    for name, params in best_params.items():
        f.write(f'Algorithm: {name}\n')
        f.write(f'Parameters: {params}\n')
        f.write(f'Best cross-validation score (R^2): {best_scores[name]}\n')
        f.write(f'R^2 (coefficient of determination): {evaluation_metrics[name]["R^2"]}\n')
        f.write(f'Mean Squared Error (MSE): {evaluation_metrics[name]["MSE"]}\n')
        f.write(f'Mean Absolute Error (MAE): {evaluation_metrics[name]["MAE"]}\n\n')
