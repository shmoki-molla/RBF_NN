import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def standardize(X, mean=None, std=None):
    if mean is None or std is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1  # защита от деления на 0
    return (X - mean) / std, mean, std

def compute_metrics(y_true, y_pred, task):
    y_true = np.array(y_true).ravel().astype(int)  # ← 1D, int

    if task == 'classification':
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
        return {'accuracy': acc, 'f1_macro': f1}

    elif task == 'regression':
        y_pred = y_pred.ravel()
        mse = mean_squared_error(y_true, y_pred)
        return {'MSE': mse, 'RMSE': np.sqrt(mse)}
    return {}

def split_data_stratified(X, y, test_size=0.2, random_state=42):
    """
    Разделяет данные с стратификацией для классификации
    и обычным разделением для регрессии
    """
    if len(np.unique(y)) > 10:  # Если много уникальных значений - вероятно регрессия
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)