import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('lab1/vgsales.csv')

# Преобразование года в числовой формат (удалим строки с некорректными значениями)
df = df[df['Year'].notna()]
df['Year'] = df['Year'].astype(int)

# Основные характеристики датасета
print("\nОсновные характеристики датасета:")
print("\nОбщая информация:")
df.info()
print("\nСтатистические характеристики:")
print(df.describe())
print("\nПропуски в данных:")
print(df.isnull().sum())

# Визуальное исследование датасета

# 1. Гистограмма глобальных продаж
plt.figure(figsize=(10, 6))
sns.histplot(df['Global_Sales'], bins=50, kde=True)
plt.title('Распределение глобальных продаж видеоигр')
plt.xlabel('Глобальные продажи (млн копий)')
plt.ylabel('Частота')
plt.xlim(0, 20)  # Ограничиваем для читаемости
plt.savefig('global_sales_hist.png')
plt.show()

# 2. Количество игр по жанрам
plt.figure(figsize=(12, 8))
genre_counts = df['Genre'].value_counts()
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title('Количество игр по жанрам')
plt.xlabel('Количество игр')
plt.ylabel('Жанр')
plt.savefig('genre_counts_bar.png')
plt.show()

# 3. Суммарные продажи по платформам (топ-10)
plt.figure(figsize=(12, 8))
platform_sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)[:10]
sns.barplot(x=platform_sales.values, y=platform_sales.index)
plt.title('Топ-10 платформ по суммарным глобальным продажам')
plt.xlabel('Глобальные продажи (млн копий)')
plt.ylabel('Платформа')
plt.savefig('platform_sales_bar.png')
plt.show()

# 4. Динамика продаж по годам
plt.figure(figsize=(12, 8))
yearly_sales = df.groupby('Year')['Global_Sales'].sum()
sns.lineplot(x=yearly_sales.index, y=yearly_sales.values)
plt.title('Глобальные продажи по годам')
plt.xlabel('Год')
plt.ylabel('Глобальные продажи (млн копий)')
plt.savefig('yearly_sales_line.png')
plt.show()

# Анализ корреляции признаков
# Выбираем только числовые признаки для корреляции
numeric_df = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']]

# Построим корреляционную матрицу
correlation_matrix = numeric_df.corr()
print("\nКорреляционная матрица:")
print(correlation_matrix)

# Тепловая карта корреляции
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Тепловая карта корреляции продаж')
plt.savefig('correlation_heatmap.png')
plt.show()