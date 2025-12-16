import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Установим seed для воспроизводимости
np.random.seed(42)

# Параметры генерации
n_samples = 1000      # Количество точек
n_features = 4        # Количество признаков (размерность пространства)
n_clusters = 4        # Количество реальных кластеров (центров)
cluster_std = 1.0     # Разброс точек внутри кластера (чем больше, тем сложнее задача)

print(f"Генерация данных: {n_samples} примеров, {n_features} признаков, {n_clusters} кластеров...")

# Генерируем данные с помощью make_blobs
# X - координаты точек, y - номер кластера (0, 1, 2...)
X, y = make_blobs(n_samples=n_samples, 
                  n_features=n_features, 
                  centers=n_clusters, 
                  cluster_std=cluster_std, 
                  random_state=42)

# Создаем имена колонок
feature_cols = [f'feature_{i+1}' for i in range(n_features)]

# Собираем в DataFrame
df = pd.DataFrame(X, columns=feature_cols)

# Добавляем колонку с правильным ответом (cluster_id).
# В задаче кластеризации мы обычно не знаем эти метки при обучении,
# но они нужны нам для проверки качества (метрика ARI).
df['cluster_id'] = y

print("\nПервые 5 строк исходных данных:")
print(df.head())
print("-" * 30)

# Нормализация данных (MinMax Scaling)
# Для алгоритмов на основе расстояний (KMeans, RBF) это критически важно.
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print("Первые 5 строк после нормализации (0..1):")
print(df.head())
print("-" * 30)

# Сохранение в CSV
output_file = 'clustering_data.csv'
df.to_csv(output_file, index=False)

print(f"\nФайл '{output_file}' успешно сохранен.")
print(f"Размер: {df.shape}")
print("\nВАЖНО: При загрузке в программу:")
print("1. Загрузите этот файл как 'Единый датасет'.")
print("2. Выберите тип задачи 'clustering'.")
print("3. (Опционально) Выберите колонку 'cluster_id' как целевую, чтобы посчитать метрику ARI.")
print("   Если не выберете колонку, программа посчитает только Silhouette Score.")