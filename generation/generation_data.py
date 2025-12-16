import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Установим seed для воспроизводимости результатов генерации случайных чисел
np.random.seed(42)

# Параметры набора данных
n_samples = 1000
n_features = 5

# Генерируем признаки (X) - матрица 1000x5 со случайными значениями (нормальное распределение)
X = np.random.randn(n_samples, n_features)

# Генерируем целевую переменную (y)
# Для реалистичности создадим линейную зависимость с небольшим шумом
# y = w1*x1 + w2*x2 + ... + шум
true_weights = np.random.rand(n_features)
noise = np.random.randn(n_samples) * 0.5  # Добавляем немного шума
y = np.dot(X, true_weights) + noise


# Создаем имена колонок: feature_1, feature_2, ..., feature_5
feature_cols = [f'feature_{i+1}' for i in range(n_features)]

# Создаем DataFrame
df = pd.DataFrame(X, columns=feature_cols)
df['target'] = y

print("Первые 5 строк исходных данных:")
print(df.head())
print("-" * 30)

# Инициализируем скейлер
scaler = MinMaxScaler()

# Нормализуем только колонки с признаками (features), целевую переменную (target) не трогаем
# Согласно заданию, нормализация выполняется до разделения на выборки
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("Первые 5 строк после нормализации:")
print(df.head())
print("-" * 30)

# Делим весь DataFrame на train и test (80% / 20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Вывод размеров полученных выборок
print(f"Размер обучающей выборки (строк, колонок): {train_df.shape}")
print(f"Размер тестовой выборки (строк, колонок): {test_df.shape}")

# Сохранение в файлы (без индекса, чтобы не создавать лишнюю колонку Unnamed: 0)
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("\nФайлы 'train_data.csv' и 'test_data.csv' успешно сохранены.")