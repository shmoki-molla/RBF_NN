import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Убедитесь, что эти файлы находятся в той же директории
from data_loader import load_dataset, split_data
from rbf_network import RBFNetwork
from utils import compute_metrics

class RBFGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RBF Нейросеть")
        self.root.geometry("700x800")

        self.model = None
        self.train_df, self.test_df = None, None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.column_names = None

        self.create_widgets()

    def create_widgets(self):
        # --- 1. Загрузка данных (с тремя кнопками) ---
        frame_data = tk.LabelFrame(self.root, text="1. Загрузка данных", padx=10, pady=10)
        frame_data.pack(fill="x", padx=10, pady=5)

        # Кнопка 1: Загрузить обучающую выборку
        tk.Button(frame_data, text="Обучающая выборка", command=self.load_train_file).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.train_label = tk.Label(frame_data, text="Не загружено", fg="red")
        self.train_label.grid(row=0, column=1, sticky="w")

        # Кнопка 2: Загрузить тестовую выборку
        tk.Button(frame_data, text="Тестовая выборка", command=self.load_test_file).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.test_label = tk.Label(frame_data, text="Не загружено", fg="red")
        self.test_label.grid(row=1, column=1, sticky="w")

        # Кнопка 3: Загрузить единый датасет для разделения
        tk.Button(frame_data, text="Единый датасет (для разделения 80/20)", command=self.load_single_file_for_splitting, bg="lightblue").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.single_file_label = tk.Label(frame_data, text="Не загружено", fg="red")
        self.single_file_label.grid(row=2, column=1, sticky="w")

        # --- 2. Выбор целевой переменной ---
        self.frame_targets = tk.LabelFrame(self.root, text="2. Выбор целевой(ых) переменной(ых)", padx=10, pady=10)
        self.frame_targets.pack(fill="x", padx=10, pady=5)
        
        targets_info_label = tk.Label(self.frame_targets, text="Выберите столбцы из списка (Ctrl/Shift для выбора нескольких).")
        targets_info_label.pack(anchor="w")

        listbox_frame = tk.Frame(self.frame_targets)
        listbox_frame.pack(fill="x", expand=True, pady=5)

        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")

        self.target_listbox = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, exportselection=False, height=5)
        self.target_listbox.pack(side="left", fill="x", expand=True)
        self.target_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.target_listbox.yview)

        tk.Button(self.frame_targets, text="Подготовить данные к обучению", command=self._prepare_data_from_selection, bg="orange").pack(pady=5)
        
        # ... (остальная часть create_widgets остается без изменений) ...

        # --- 3. Параметры модели ---
        frame_params = tk.LabelFrame(self.root, text="3. Параметры модели", padx=10, pady=10)
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

        # --- 4. Результаты ---
        frame_results = tk.LabelFrame(self.root, text="4. Результаты", padx=10, pady=10)
        frame_results.pack(fill="both", expand=True, padx=10, pady=5)

        self.result_text = tk.Text(frame_results, height=10, state="disabled")
        self.result_text.pack(fill="both", expand=True)

        tk.Button(frame_results, text="Тестировать", command=self.test_model).pack(side="left", padx=5)
        tk.Button(frame_results, text="Предсказать", command=self.predict_single).pack(side="left", padx=5)


    def _reset_data_state(self):
        """Сбрасывает все переменные, связанные с данными."""
        self.train_df, self.test_df = None, None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.target_listbox.delete(0, tk.END)
        self.train_label.config(text="Не загружено", fg="red")
        self.test_label.config(text="Не загружено", fg="red")
        self.single_file_label.config(text="Не загружено", fg="red")
        self.model = None

    def update_target_listbox(self, df):
        """Обновляет список колонок для выбора целевой переменной."""
        self.target_listbox.delete(0, tk.END)
        if df is not None:
            self.column_names = df.columns
            for col_name in self.column_names:
                self.target_listbox.insert(tk.END, col_name)

    def load_train_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV and NumPy files", "*.csv *.npy")])
        if not path: return
        try:
            # Сбрасываем предыдущие загрузки
            self._reset_data_state()
            self.train_df = load_dataset(path)
            self.train_label.config(text=f"{os.path.basename(path)} ({self.train_df.shape[0]} строк)", fg="green")
            self.single_file_label.config(text="Используется отдельный файл для обучения", fg="gray")
            self.update_target_listbox(self.train_df)
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def load_test_file(self):
        if self.train_df is None:
            messagebox.showwarning("Внимание", "Сначала загрузите обучающую выборку.")
            return
        path = filedialog.askopenfilename(filetypes=[("CSV and NumPy files", "*.csv *.npy")])
        if not path: return
        try:
            self.test_df = load_dataset(path)
            # Проверка на совпадение колонок
            if set(self.train_df.columns) != set(self.test_df.columns):
                self.test_df = None
                raise ValueError("Колонки в обучающем и тестовом файлах не совпадают!")
            self.test_label.config(text=f"{os.path.basename(path)} ({self.test_df.shape[0]} строк)", fg="green")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def load_single_file_for_splitting(self):
        path = filedialog.askopenfilename(filetypes=[("CSV and NumPy files", "*.csv *.npy")])
        if not path: return
        try:
            # Сбрасываем всё, так как это новый сценарий
            self._reset_data_state()
            self.train_df = load_dataset(path) # Загружаем всё в train_df
            self.test_df = None # Явно указываем, что тестового файла нет
            
            filename = os.path.basename(path)
            self.single_file_label.config(text=f"{filename} ({self.train_df.shape[0]} строк)", fg="green")
            self.train_label.config(text="Будет создана из единого файла", fg="blue")
            self.test_label.config(text="Будет создана из единого файла", fg="blue")
            self.update_target_listbox(self.train_df)
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def _prepare_data_from_selection(self):
        """Главная функция подготовки данных после выбора целевых столбцов."""
        selected_indices = self.target_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Ошибка", "Выберите хотя бы один целевой столбец!")
            return
        if self.train_df is None:
            messagebox.showerror("Ошибка", "Сначала загрузите данные!")
            return

        selected_targets = [self.target_listbox.get(i) for i in selected_indices]
        
        try:
            # Сценарий 1: Загружен один файл, который нужно разделить
            if self.test_df is None:
                X, y = split_data(self.train_df, selected_targets)
                
                try: # Пытаемся разделить стратифицированно
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                except ValueError: # Если не вышло (например, регрессия), делим обычно
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                self.append_result(f"Единый датасет разделен на:\n")
                self.append_result(f"  Обучение: {self.X_train.shape[0]} примеров\n")
                self.append_result(f"  Тест:      {self.X_test.shape[0]} примеров\n")

            # Сценарий 2: Загружены оба файла (train и test)
            else:
                self.X_train, self.y_train = split_data(self.train_df, selected_targets)
                self.X_test, self.y_test = split_data(self.test_df, selected_targets)
                self.append_result(f"Данные из файлов подготовлены:\n")
                self.append_result(f"  Обучение: {self.X_train.shape[0]} примеров\n")
                self.append_result(f"  Тест:      {self.X_test.shape[0]} примеров\n")
            
            messagebox.showinfo("Успех", "Данные успешно подготовлены к обучению!")

        except Exception as e:
            messagebox.showerror("Ошибка подготовки данных", str(e))

    # Методы train_model, test_model, predict_single и append_result остаются без изменений.
    # Их код можно скопировать из предыдущего ответа.
    
    def train_model(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showwarning("Ошибка", "Данные не подготовлены. Загрузите данные, выберите целевые столбцы и нажмите 'Подготовить данные'.")
            return

        task_type = self.task_var.get()
        # if task_type == 'classification':
        #     if not np.all(np.equal(np.mod(self.y_train, 1), 0)):
        #         messagebox.showerror("Ошибка данных", "Для классификации целевые значения должны быть целыми числами.")
        #         return

        try:
            n_centers = int(self.centers_var.get())
            if n_centers < 1 or n_centers > self.X_train.shape[0]:
                raise ValueError(f"Число центров должно быть от 1 до {self.X_train.shape[0]}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверное число центров: {e}")
            return

        spread_str = self.spread_var.get()
        spread = 'auto' if spread_str == 'auto' else float(spread_str)

        self.append_result("Обучение...\n")
        self.model = RBFNetwork(n_centers=n_centers, spread=spread)
        try:
            self.model.fit(self.X_train, self.y_train, task=self.task_var.get(), center_method=self.method_var.get())
            self.append_result("Обучение завершено!\n")
        except Exception as e:
            self.append_result(f"Ошибка при обучении: {e}\n")
            messagebox.showerror("Ошибка обучения", str(e))

    def test_model(self):
        if not self.model:
            messagebox.showwarning("Ошибка", "Сначала обучите модель.")
            return
        if self.X_test is None or self.y_test is None:
            messagebox.showwarning("Ошибка", "Тестовая выборка не подготовлена.")
            return
            
        self.append_result("Тестирование...\n")
        try:
            y_pred = self.model.predict(self.X_test)
            metrics = compute_metrics(self.y_test, y_pred, self.model.task)
            self.append_result("Результаты на тестовой выборке:\n")
            for k, v in metrics.items():
                self.append_result(f"  {k}: {v:.4f}\n")
        except Exception as e:
            self.append_result(f"Ошибка при тестировании: {e}\n")
            messagebox.showerror("Ошибка тестирования", str(e))

    def predict_single(self):
        if not self.model:
            messagebox.showwarning("Ошибка", "Обучите модель")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Предсказание для одного примера")
        dialog.geometry("400x150")
        
        num_features = self.model.X_mean.shape[0]

        tk.Label(dialog, text=f"Введите {num_features} признаков через пробел:").pack(pady=10)
        entry = tk.Entry(dialog, width=50)
        entry.pack(pady=10)

        def run_prediction():
            try:
                values_str = entry.get().strip().split()
                if len(values_str) != num_features:
                    raise ValueError(f"Ожидалось {num_features} признаков, а введено {len(values_str)}")
                
                values = np.array([float(v) for v in values_str]).reshape(1, -1)
                pred = self.model.predict(values)

                if self.model.task == 'classification':
                    pred_probs = pred[0]
                    cls = np.argmax(pred_probs)
                    probs_str = ", ".join([f"{p:.3f}" for p in pred_probs])
                    result = f"Предсказанный класс: {cls}\nВероятности: [{probs_str}]"
                else: # regression
                    pred_values = pred.flatten()
                    result_str = ", ".join([f"{val:.4f}" for val in pred_values])
                    result = f"Предсказанные значения: [{result_str}]"
                
                messagebox.showinfo("Результат предсказания", result)
                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Ошибка ввода", str(e), parent=dialog)

        tk.Button(dialog, text="Предсказать", command=run_prediction).pack(pady=10)


    def append_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text)
        self.result_text.see("end")
        self.result_text.config(state="disabled")