import numpy as np
import pandas as pd
import os

def load_dataset(path, target_col=-1, delimiter=','):
    """
    Загружает данные из CSV или .npy.
    Возвращает X (признаки), y (целевая переменная).
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
                df = pd.DataFrame(data)
            else:
                raise ValueError("Формат .npy должен быть 2D массивом")
        except Exception as e:
            raise ValueError(f"Ошибка чтения .npy: {e}")
    else:
        raise ValueError("Поддерживаются только .csv и .npy")

    if df.shape[1] < 2:
        raise ValueError("В данных должно быть минимум 2 столбца (признаки + целевая)")

    if target_col == -1:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        X = df.drop(columns=[df.columns[target_col]]).values
        y = df.iloc[:, target_col].values

    X = np.array(X, dtype=float)
    y = np.array(y).ravel()  # ← УБРАЛИ .reshape(-1, 1) → y теперь 1D!
    return X, y