import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score

def standardize(X, mean=None, std=None):
    """Стандартизация данных с защитой от деления на 0"""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1.0, std)  # защита от деления на 0
    
    return (X - mean) / std, mean, std

def compute_metrics(y_true, y_pred, task):
    """Вычисление метрик в зависимости от типа задачи"""
    y_true = np.array(y_true)
    
    if task == 'classification':
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Многоклассовая классификация
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            # Бинарная классификация
            y_pred_labels = (y_pred.ravel() > 0.5).astype(int)
        
        y_true_flat = y_true.ravel().astype(int)
        acc = accuracy_score(y_true_flat, y_pred_labels)
        f1 = f1_score(y_true_flat, y_pred_labels, average='weighted', zero_division=0)
        return {'accuracy': acc, 'f1_score': f1}

    elif task == 'regression':
        y_pred_flat = y_pred.ravel()
        y_true_flat = y_true.ravel()
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_flat, y_pred_flat)
        return {'MSE': mse, 'RMSE': rmse, 'R2_score': r2}
    
    return {}