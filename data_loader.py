import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
        
    # Проверяем, что все целевые столбцы существуют в DataFrame
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Следующие целевые столбцы не найдены: {', '.join(missing_cols)}")

    y = df[target_cols].values
    X = df.drop(columns=target_cols).values

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float) 
    
    return X, y

def load_and_split_dataset(path, target_cols, delimiter=',', test_size=0.2, random_state=42):
    """
    Загружает данные, разделяет на X и y, а затем на обучающую и тестовую выборки.
    Возвращает: X_train, X_test, y_train, y_test
    """
    # 1. Сначала загружаем полный датафрейм
    df = load_dataset(path, delimiter)
    
    # 2. Разделяем на признаки и цели
    X, y = split_data(df, target_cols)

    # 3. Разделяем на обучение и тест (без стратификации, чтобы работало и для регрессии)
    # Для более умного разделения можно использовать функцию из utils.py
    try:
        # Попытка стратификации для классификации
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Если стратификация не удалась (например, в регрессии), делим обычно
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    return X_train, X_test, y_train, y_test, df.columns