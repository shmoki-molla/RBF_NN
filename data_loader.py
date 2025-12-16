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
            # Читаем CSV. skipinitialspace помогает убрать пробелы после запятых
            df = pd.read_csv(path, delimiter=delimiter, skipinitialspace=True)
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
        raise ValueError("В данных должно быть минимум 2 столбца")
        
    return df

def split_data(df, target_cols):
    """
    Разделяет DataFrame на X (признаки) и y (целевые переменные).
    Автоматически удаляет нечисловые колонки из X, чтобы избежать ошибок.
    """
    if target_cols:
        missing_cols = [col for col in target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Целевые столбцы не найдены: {', '.join(missing_cols)}")
        
        y = df[target_cols].values
        # Удаляем целевые колонки из признаков
        X_df = df.drop(columns=target_cols)
    else:
        # Если цель не указана (кластеризация без учителя), то всё - признаки
        y = None
        X_df = df

    # --- ИСПРАВЛЕНИЕ: Оставляем только числовые колонки для X ---
    # Это предотвращает краш "could not convert string to float"
    X_numeric = X_df.select_dtypes(include=[np.number])
    
    if X_numeric.shape[1] < X_df.shape[1]:
        dropped = set(X_df.columns) - set(X_numeric.columns)
        print(f"ВНИМАНИЕ: Исключены нечисловые колонки из признаков: {dropped}")

    if X_numeric.empty:
        raise ValueError("В признаках (X) не осталось числовых данных!")

    X = X_numeric.values
    X = np.nan_to_num(X) # Заменяем NaN на 0, чтобы не было ошибок вычислений

    return X, y