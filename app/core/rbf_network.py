import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer

# Импорт из нашего пакета utils
from app.utils.metrics import standardize

class RBFNetwork:
    def __init__(self, n_centers=10, spread='auto', regularization=1e-6):
        self.n_centers = n_centers
        self.spread = spread
        self.regularization = regularization

        self.centers = None
        self.width = None
        self.weights = None
        self.task = None
        self.X_mean, self.X_std = None, None
        self.y_mean, self.y_std = None, None
        self.label_binarizer = None

    def _design_matrix(self, X):
        """Создает матрицу активаций RBF"""
        Phi = np.zeros((X.shape[0], self.n_centers))
        for i, center in enumerate(self.centers):
            distances = np.sum((X - center) ** 2, axis=1)
            Phi[:, i] = np.exp(-distances / (2 * self.width ** 2))
        return np.hstack([Phi, np.ones((X.shape[0], 1))])  # bias

    def fit(self, X, y=None, task='classification', center_method='kmeans'):
        self.task = task
        X = np.array(X, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Нормализация признаков
        X_norm, self.X_mean, self.X_std = standardize(X)
        
        # Выбор центров
        if center_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_centers, random_state=42, n_init=10)
            kmeans.fit(X_norm)
            self.centers = kmeans.cluster_centers_
        elif center_method == 'random':
            idx = np.random.choice(len(X_norm), self.n_centers, replace=False)
            self.centers = X_norm[idx]
        else:
            raise ValueError("center_method: 'kmeans' или 'random'")

        # Вычисление ширины (Spread)
        if self.spread == 'auto':
            if len(self.centers) > 1:
                dists = []
                for i, center in enumerate(self.centers):
                    other_centers = np.delete(self.centers, i, axis=0)
                    min_dists = np.min(np.sum((other_centers - center) ** 2, axis=1))
                    dists.append(min_dists)
                self.width = np.sqrt(np.median(dists)) / 2
            else:
                self.width = 1.0
        else:
            self.width = float(self.spread)

        if self.task == 'clustering':
            return

        # Для обучения с учителем
        if y is None:
            raise ValueError("Для regression/classification необходим вектор y.")
            
        y = np.array(y)
        
        if task == 'regression':
            y_norm, self.y_mean, self.y_std = standardize(y)
            y_target = y_norm
        else:  # classification
            self.label_binarizer = LabelBinarizer()
            y_target = self.label_binarizer.fit_transform(y)
            if y_target.shape[1] == 1:
                y_target = np.hstack([1 - y_target, y_target])
            self.y_mean, self.y_std = None, None

        Phi = self._design_matrix(X_norm)
        ridge = Ridge(alpha=self.regularization, fit_intercept=False)
        ridge.fit(Phi, y_target)
        self.weights = ridge.coef_.T if ridge.coef_.ndim > 1 else ridge.coef_.reshape(-1, 1)

    def predict(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if self.task == 'clustering':
            distances = np.zeros((X_norm.shape[0], self.n_centers))
            for i, center in enumerate(self.centers):
                distances[:, i] = np.sum((X_norm - center) ** 2, axis=1)
            return np.argmin(distances, axis=1)

        Phi = self._design_matrix(X_norm)
        output = Phi @ self.weights
        
        if self.task == 'classification':
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            return probabilities
        else:  # regression
            if self.y_mean is not None and self.y_std is not None:
                output = output * self.y_std + self.y_mean
            return output.ravel() if output.shape[1] == 1 else output