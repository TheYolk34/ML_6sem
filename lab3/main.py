import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Загрузка данных
train_df = pd.read_csv("lab3/dataset/train.csv", encoding="latin1")
test_df = pd.read_csv("lab3/dataset/test.csv", encoding="latin1")
test_salaries_df = pd.read_csv("lab3/dataset/test_salaries.csv", encoding="latin1")

# Обработка пропусков
train_df["Pr/St"] = train_df["Pr/St"].fillna(train_df["Pr/St"].mode()[0])
test_df["Pr/St"] = test_df["Pr/St"].fillna(test_df["Pr/St"].mode()[0])

# Удаление строк, где DftYr, DftRd, Ovrl пустые
train_df = train_df.dropna(subset=["DftYr", "DftRd", "Ovrl"], how="all")
test_df = test_df.dropna(subset=["DftYr", "DftRd", "Ovrl"], how="all")
test_salaries_df = test_salaries_df.loc[test_df.index]

# Заполнение пропусков в числовых колонках средним значением
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.drop("Salary", errors="ignore")

# Заполняем пропуски средним только в числовых столбцах
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
test_numeric_cols = numeric_cols.intersection(test_df.columns)  # Числовые столбцы в test_df
test_df[test_numeric_cols] = test_df[test_numeric_cols].fillna(test_df[test_numeric_cols].mean())

# Удаление столбцов с некорректными строковыми значениями (например, датами)
for col in train_df.columns:
    if train_df[col].dtype == "object":
        try:
            train_df[col] = train_df[col].astype(float)
            test_df[col] = test_df[col].astype(float)
        except ValueError:
            train_df = train_df.drop(columns=[col])
            test_df = test_df.drop(columns=[col])

# Разделение признаков и целевой переменной
X_train = train_df.drop(columns=["Salary"])
y_train = train_df["Salary"]
X_test = test_df
y_test = test_salaries_df["Salary"]

# Масштабирование данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели KNN с K=5
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Оценка модели
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mae}")
print(f"MSE: {mse}")

# Подбор гиперпараметров
param_grid = {"n_neighbors": range(1, 30)}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring="neg_mean_absolute_error")
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_["n_neighbors"]
print(f"Лучшее значение K: {best_k}")

# Обучение модели с лучшим K
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)

# Оценка оптимальной модели
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
print(f"MAE (оптимальная модель): {mae_best}")
print(f"MSE (оптимальная модель): {mse_best}")

# Вывод предсказанных и реальных значений
print("\nСравнение предсказанных и реальных зарплат (первые 10 примеров):")
for i in range(100):
    print(f"Игрок {i+1}: Предсказано: ${y_pred_best[i]:,.2f}, Реально: ${y_test.iloc[i]:,.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="Реальные зарплаты", color='blue', alpha=0.6)
plt.scatter(range(len(y_pred_best)), y_pred_best, label="Предсказанные зарплаты", color='red', alpha=0.6)
plt.xlabel("Игроки")
plt.ylabel("Зарплата")
plt.legend()
plt.title("Сравнение предсказанных и реальных зарплат игроков")
plt.show()