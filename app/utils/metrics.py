import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

def standardize(X, mean=None, std=None):
    if mean is None or std is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1
    return (X - mean) / std, mean, std

def compute_metrics(y_true, y_pred, task, X=None):
    """
    Вычисляет метрики в зависимости от задачи.
    """
    # --- КЛАСТЕРИЗАЦИЯ ---
    if task == 'clustering':
        metrics = {}
        if X is not None and len(np.unique(y_pred)) > 1:
            try:
                sil = silhouette_score(X, y_pred)
                metrics['Silhouette'] = sil
            except Exception:
                metrics['Silhouette'] = -1.0
        
        if y_true is not None:
            # Обработка меток
            if y_true.dtype == object or y_true.dtype.kind in 'US':
                le = LabelEncoder()
                y_true = le.fit_transform(y_true.ravel())
            else:
                y_true = y_true.ravel().astype(int)
                
            ari = adjusted_rand_score(y_true, y_pred)
            metrics['ARI (Ground Truth match)'] = ari
        return metrics

    # --- ОСТАЛЬНЫЕ ЗАДАЧИ ---
    y_true = np.array(y_true).ravel()
    
    if task == 'classification':
        if y_true.dtype == object or y_true.dtype.kind in 'US':
            le = LabelEncoder()
            y_true = le.fit_transform(y_true).astype(int)
        else:
            y_true = y_true.astype(int)
            
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