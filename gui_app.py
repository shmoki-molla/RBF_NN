import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
from data_loader import load_dataset, load_and_split_dataset
from rbf_network import RBFNetwork
from utils import compute_metrics

class RBFGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RBF Нейросеть")
        self.root.geometry("700x600")

        self.model = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.create_widgets()

    def create_widgets(self):
        # --- Загрузка данных ---
        frame_data = tk.LabelFrame(self.root, text="1. Загрузка данных", padx=10, pady=10)
        frame_data.pack(fill="x", padx=10, pady=5)

        # Существующие кнопки
        tk.Button(frame_data, text="Обучающая выборка", command=self.load_train).grid(row=0, column=0, padx=5, pady=5)
        self.train_label = tk.Label(frame_data, text="Не загружено", fg="red")
        self.train_label.grid(row=0, column=1, sticky="w")

        tk.Button(frame_data, text="Тестовая выборка", command=self.load_test).grid(row=1, column=0, padx=5, pady=5)
        self.test_label = tk.Label(frame_data, text="Не загружено", fg="red")
        self.test_label.grid(row=1, column=1, sticky="w")

        # Новая кнопка для загрузки одного файла
        tk.Button(frame_data, text="Загрузить и разделить данные (80/20)", 
                 command=self.load_and_split, bg="lightblue").grid(row=2, column=0, padx=5, pady=5)
        self.split_label = tk.Label(frame_data, text="Не загружено", fg="red")
        self.split_label.grid(row=2, column=1, sticky="w")

        # --- Параметры ---
        frame_params = tk.LabelFrame(self.root, text="2. Параметры модели", padx=10, pady=10)
        frame_params.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_params, text="Тип задачи:").grid(row=0, column=0, sticky="w")
        self.task_var = tk.StringVar(value="classification")
        ttk.Combobox(frame_params, textvariable=self.task_var, values=["classification", "regression"], state="readonly").grid(row=0, column=1, padx=5)

        tk.Label(frame_params, text="Центры:").grid(row=1, column=0, sticky="w")
        self.centers_var = tk.StringVar(value="4")
        tk.Entry(frame_params, textvariable=self.centers_var, width=10).grid(row=1, column=1, sticky="w", padx=5)

        tk.Label(frame_params, text="Метод центров:").grid(row=2, column=0, sticky="w")
        self.method_var = tk.StringVar(value="kmeans")
        ttk.Combobox(frame_params, textvariable=self.method_var, values=["kmeans", "random"], state="readonly").grid(row=2, column=1, padx=5, sticky="w")

        tk.Label(frame_params, text="Ширина (spread):").grid(row=3, column=0, sticky="w")
        self.spread_var = tk.StringVar(value="auto")
        ttk.Combobox(frame_params, textvariable=self.spread_var, values=["auto", "0.5", "1.0", "2.0"], state="readonly").grid(row=3, column=1, padx=5, sticky="w")

        tk.Button(frame_params, text="Обучить модель", command=self.train_model, bg="lightgreen").grid(row=4, column=0, columnspan=2, pady=10)

        # --- Результаты ---
        frame_results = tk.LabelFrame(self.root, text="3. Результаты", padx=10, pady=10)
        frame_results.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_text = tk.Text(frame_results, height=10, state="disabled")
        self.result_text.pack(fill="both", expand=True)

        tk.Button(frame_results, text="Тестировать", command=self.test_model).pack(side="left", padx=5)
        tk.Button(frame_results, text="Предсказать", command=self.predict_single).pack(side="left", padx=5)

    def load_train(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("NumPy files", "*.npy")])
        if path:
            try:
                X, y = load_dataset(path)
                self.X_train, self.y_train = X, y
                self.train_label.config(text=f"{os.path.basename(path)} ({X.shape[0]} примеров)", fg="green")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def load_test(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("NumPy files", "*.npy")])
        if path:
            try:
                X, y = load_dataset(path)
                self.X_test, self.y_test = X, y
                self.test_label.config(text=f"{os.path.basename(path)} ({X.shape[0]} примеров)", fg="green")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def load_and_split(self):
        """Новый метод: загружает один файл и автоматически разделяет данные"""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("NumPy files", "*.npy")])
        if path:
            try:
                X_train, X_test, y_train, y_test = load_and_split_dataset(path)
                self.X_train, self.y_train = X_train, y_train
                self.X_test, self.y_test = X_test, y_test
                
                filename = os.path.basename(path)
                self.train_label.config(text=f"{filename} train ({X_train.shape[0]} примеров)", fg="green")
                self.test_label.config(text=f"{filename} test ({X_test.shape[0]} примеров)", fg="green")
                self.split_label.config(text="Разделено успешно", fg="green")
                
                self.append_result(f"Данные загружены и разделены:\n")
                self.append_result(f"  Обучающая выборка: {X_train.shape[0]} примеров\n")
                self.append_result(f"  Тестовая выборка: {X_test.shape[0]} примеров\n")
                
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def train_model(self):
        if self.X_train is None:
            messagebox.showwarning("Ошибка", "Загрузите обучающую выборку")
            return

        try:
            n_centers = int(self.centers_var.get())
            if n_centers < 1 or n_centers > 200:
                raise ValueError("Центры: 1–200")
        except:
            messagebox.showerror("Ошибка", "Неверное число центров")
            return

        spread = self.spread_var.get()
        spread = 'auto' if spread == 'auto' else float(spread)

        self.append_result("Обучение...\n")
        self.model = RBFNetwork(n_centers=n_centers, spread=spread)
        try:
            self.model.fit(self.X_train, self.y_train, task=self.task_var.get(), center_method=self.method_var.get())
            self.append_result("Обучение завершено!\n")
        except Exception as e:
            self.append_result(f"Ошибка: {e}\n")

    def test_model(self):
        if not self.model or self.X_test is None:
            messagebox.showwarning("Ошибка", "Обучите модель и загрузите тестовую выборку")
            return

        self.append_result("Тестирование...\n")
        try:
            y_pred = self.model.predict(self.X_test)
            metrics = compute_metrics(self.y_test, y_pred, self.model.task)
            self.append_result("Результаты:\n")
            for k, v in metrics.items():
                self.append_result(f"  {k}: {v:.4f}\n")
        except Exception as e:
            self.append_result(f"Ошибка: {e}\n")

    def predict_single(self):
        if not self.model:
            messagebox.showwarning("Ошибка", "Обучите модель")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Предсказание")
        dialog.geometry("300x150")

        tk.Label(dialog, text="Признаки (через пробел):").pack(pady=5)
        entry = tk.Entry(dialog, width=40)
        entry.pack(pady=5)

        def run():
            try:
                values = list(map(float, entry.get().strip().split()))
                if len(values) != self.X_train.shape[1]:
                    raise ValueError(f"Ожидалось {self.X_train.shape[1]} признаков")
                pred = self.model.predict([values])[0]
                if self.model.task == 'classification':
                    cls = int(np.argmax(pred))
                    probs = ", ".join(f"{p:.3f}" for p in pred)
                    result = f"Класс: {cls}\nВероятности: [{probs}]"
                else:
                    result = f"Значение: {pred[0]:.4f}"
                messagebox.showinfo("Предсказание", result)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
            dialog.destroy()

        tk.Button(dialog, text="Предсказать", command=run).pack(pady=10)

    def append_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text)
        self.result_text.see("end")
        self.result_text.config(state="disabled")