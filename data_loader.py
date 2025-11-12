import numpy as np
import pandas as pd
import os

def load_dataset(path, delimiter=','):
    """
    Загружает данные из CSV или .npy.
    Возвращает полный DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    if path.endswith('.csv'):
        try:
            df = pd.read_csv(path, delimiter=delimiter)
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV: {e}")
    elif path.endswith('.npy'):
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.ndim == 2:
                # Для .npy файлов у нас нет названий колонок, генерируем их
                df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
            else:
                raise ValueError("Формат .npy должен быть 2D массивом")
        except Exception as e:
            raise ValueError(f"Ошибка чтения .npy: {e}")
    else:
        raise ValueError("Поддерживаются только .csv и .npy")

    if df.shape[1] < 2:
        raise ValueError("В данных должно быть минимум 2 столбца (признаки + целевая)")
        
    return df

def split_data(df, target_cols):
    """
    Разделяет DataFrame на X (признаки) и y (целевые переменные).
    target_cols - список названий целевых столбцов.
    """
    if not target_cols:
        raise ValueError("Необходимо выбрать хотя бы один целевой столбец.")
        
    y = df[target_cols].values
    X = df.drop(columns=target_cols).values

    # Убедимся, что типы данных корректны
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float) # Целевая переменная тоже может быть float
    
    return X, y