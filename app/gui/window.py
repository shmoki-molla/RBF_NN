import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import os
import pickle
import numpy as np

# Импорты из внутренней структуры приложения
from app.utils.data_loader import load_dataset, split_data
from app.core.rbf_network import RBFNetwork
from app.utils.metrics import compute_metrics
from app.config import DEFAULT_CENTERS, DEFAULT_METHOD, DEFAULT_SPREAD

class RBFGUIApp:
    def __init__(self, root):
        self.root = root
        
        # Данные состояния
        self.model = None
        self.train_df, self.test_df = None, None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.column_names = None

        # Основной контейнер
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=BOTH, expand=YES)

        # Создание UI
        self.create_header(main_frame)
        self.create_data_section(main_frame)
        
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=X, pady=10)
        
        self.create_params_section(middle_frame)
        self.create_target_section(middle_frame)
        
        self.create_controls_section(main_frame)
        self.create_results_section(main_frame)

    def create_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=X, pady=(0, 15))
        
        title = ttk.Label(header_frame, text="RBF Network Modeling", font=("Helvetica", 20, "bold"), bootstyle=PRIMARY)
        title.pack(side=LEFT)
        
        subtitle = ttk.Label(header_frame, text="Classification • Regression • Clustering", font=("Helvetica", 10))
        subtitle.pack(side=LEFT, padx=10, pady=(10, 0))

    def create_data_section(self, parent):
        frame = ttk.Labelframe(parent, text=" 1. Источники данных ", padding=15, bootstyle=INFO)
        frame.pack(fill=X, pady=5)
        frame.columnconfigure(1, weight=1)

        ttk.Button(frame, text="📂 Train Data", bootstyle="info-outline", command=self.load_train_file, width=20).grid(row=0, column=0, sticky="w", pady=2)
        self.train_label = ttk.Label(frame, text="Файл не выбран", bootstyle=SECONDARY)
        self.train_label.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Button(frame, text="📂 Test Data", bootstyle="info-outline", command=self.load_test_file, width=20).grid(row=1, column=0, sticky="w", pady=2)
        self.test_label = ttk.Label(frame, text="Файл не выбран", bootstyle=SECONDARY)
        self.test_label.grid(row=1, column=1, sticky="w", padx=10)

        ttk.Button(frame, text="📄 Single File (Split)", bootstyle="secondary-outline", command=self.load_single_file_for_splitting, width=20).grid(row=2, column=0, sticky="w", pady=2)
        self.single_file_label = ttk.Label(frame, text="Файл не выбран", bootstyle=SECONDARY)
        self.single_file_label.grid(row=2, column=1, sticky="w", padx=10)

    def create_params_section(self, parent):
        frame = ttk.Labelframe(parent, text=" 2. Конфигурация ", padding=15, bootstyle=PRIMARY)
        frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))

        ttk.Label(frame, text="Тип задачи:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.task_var = tk.StringVar(value="classification")
        self.task_combo = ttk.Combobox(frame, textvariable=self.task_var, values=["classification", "regression", "clustering"], state="readonly", bootstyle=PRIMARY)
        self.task_combo.pack(fill=X, pady=(5, 10))
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_changed)

        row1 = ttk.Frame(frame)
        row1.pack(fill=X, pady=5)
        ttk.Label(row1, text="Число RBF центров:").pack(side=LEFT)
        self.centers_var = tk.StringVar(value=str(DEFAULT_CENTERS))
        ttk.Entry(row1, textvariable=self.centers_var, width=8).pack(side=RIGHT)

        row2 = ttk.Frame(frame)
        row2.pack(fill=X, pady=5)
        ttk.Label(row2, text="Метод центров:").pack(side=LEFT)
        self.method_var = tk.StringVar(value=DEFAULT_METHOD)
        ttk.Combobox(row2, textvariable=self.method_var, values=["kmeans", "random"], state="readonly", width=12).pack(side=RIGHT)

        row3 = ttk.Frame(frame)
        row3.pack(fill=X, pady=5)
        ttk.Label(row3, text="Ширина (Spread):").pack(side=LEFT)
        self.spread_var = tk.StringVar(value=DEFAULT_SPREAD)
        ttk.Combobox(row3, textvariable=self.spread_var, values=["auto", "0.5", "1.0", "2.0"], state="readonly", width=12).pack(side=RIGHT)

    def create_target_section(self, parent):
        frame = ttk.Labelframe(parent, text=" 3. Целевые переменные ", padding=15, bootstyle=WARNING)
        frame.pack(side=LEFT, fill=BOTH, expand=YES)

        self.targets_info_label = ttk.Label(frame, text="Выберите столбцы (Y):", bootstyle=SECONDARY)
        self.targets_info_label.pack(anchor="w", pady=(0, 5))

        list_container = ttk.Frame(frame)
        list_container.pack(fill=BOTH, expand=YES)
        
        scrollbar = ttk.Scrollbar(list_container, bootstyle="round")
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.target_listbox = tk.Listbox(list_container, selectmode=tk.EXTENDED, height=6, exportselection=False, 
                                         font=("Consolas", 10), relief="flat", borderwidth=1)
        self.target_listbox.pack(side=LEFT, fill=BOTH, expand=YES)
        self.target_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.target_listbox.yview)

        ttk.Button(frame, text="⚡ Подготовить данные", bootstyle="warning", command=self._prepare_data_from_selection).pack(fill=X, pady=(10, 0))

    def create_controls_section(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=X, pady=10)

        self.btn_train = ttk.Button(frame, text="▶ ЗАПУСТИТЬ ОБУЧЕНИЕ", command=self.train_model, bootstyle="success", width=30)
        self.btn_train.pack(side=LEFT, fill=X, expand=YES, padx=(0, 5))

        ttk.Button(frame, text="💾 Сохранить", command=self.save_model, bootstyle="secondary-outline").pack(side=LEFT, padx=5)
        ttk.Button(frame, text="📂 Загрузить модель", command=self.load_model, bootstyle="secondary-outline").pack(side=LEFT, padx=5)

    def create_results_section(self, parent):
        frame = ttk.Labelframe(parent, text=" Результаты и Тестирование ", padding=15)
        frame.pack(fill=BOTH, expand=YES, pady=5)

        btn_panel = ttk.Frame(frame)
        btn_panel.pack(fill=X, pady=(0, 10))
        
        ttk.Button(btn_panel, text="📊 Тест на выборке", command=self.test_model, bootstyle="info").pack(side=LEFT, padx=(0, 5))
        ttk.Button(btn_panel, text="🎲 Предсказать одно значение", command=self.predict_single, bootstyle="info-outline").pack(side=LEFT)

        self.result_text = ScrolledText(frame, height=10, state="disabled", font=("Consolas", 9))
        self.result_text.pack(fill=BOTH, expand=YES)

    # --- ЛОГИКА ---

    def _reset_data_state(self):
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.model = None
        self.append_result("--- Сброс данных ---\n")

    def _on_task_changed(self, event):
        self._reset_data_state()
        task = self.task_var.get()
        if task == 'clustering':
            self.targets_info_label.config(text="Цель не обязательна (Кластеризация)")
        else:
            self.targets_info_label.config(text="Выберите столбцы для Y (Target):")

    def update_target_listbox(self, df):
        self.target_listbox.delete(0, tk.END)
        if df is not None:
            self.column_names = df.columns
            for col_name in self.column_names:
                self.target_listbox.insert(tk.END, col_name)

    def load_train_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/NPY", "*.csv *.npy")])
        if not path: return
        try:
            self.train_df = load_dataset(path)
            self._reset_data_state()
            self.train_label.config(text=f"{os.path.basename(path)}", bootstyle="success")
            self.update_target_listbox(self.train_df)
        except Exception as e: messagebox.showerror("Ошибка", str(e))

    def load_test_file(self):
        if self.train_df is None: return messagebox.showwarning("", "Сначала загрузите train")
        path = filedialog.askopenfilename(filetypes=[("CSV/NPY", "*.csv *.npy")])
        if not path: return
        try:
            self.test_df = load_dataset(path)
            if set(self.train_df.columns) != set(self.test_df.columns): 
                 raise ValueError("Колонки не совпадают")
            self.test_label.config(text=f"{os.path.basename(path)}", bootstyle="success")
        except Exception as e: messagebox.showerror("Ошибка", str(e))

    def load_single_file_for_splitting(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/NPY", "*.csv *.npy")])
        if not path: return
        try:
            self.train_df = load_dataset(path)
            self.test_df = None
            self._reset_data_state()
            self.single_file_label.config(text=f"{os.path.basename(path)}", bootstyle="success")
            self.train_label.config(text="Общий файл", bootstyle="secondary")
            self.test_label.config(text="Авто-сплит", bootstyle="secondary")
            self.update_target_listbox(self.train_df)
        except Exception as e: messagebox.showerror("Ошибка", str(e))

    def _prepare_data_from_selection(self):
        task = self.task_var.get()
        selected_indices = self.target_listbox.curselection()
        selected_targets = [self.target_listbox.get(i) for i in selected_indices]

        if not selected_targets and task != 'clustering':
            messagebox.showerror("Внимание", "Выберите целевую переменную для обучения с учителем!")
            return
        
        if self.train_df is None:
            messagebox.showerror("Ошибка", "Данные не загружены")
            return

        try:
            def get_X_y_safe(df, targets):
                return split_data(df, targets)

            if self.test_df is None:
                X, y = get_X_y_safe(self.train_df, selected_targets)
                if y is None: # Clustering or no target
                     from sklearn.model_selection import train_test_split
                     self.X_train, self.X_test = train_test_split(X, test_size=0.2, random_state=42)
                     self.y_train, self.y_test = None, None
                else:
                    from sklearn.model_selection import train_test_split
                    try:
                        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=y
                        )
                    except ValueError:
                         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
            else:
                self.X_train, self.y_train = get_X_y_safe(self.train_df, selected_targets)
                self.X_test, self.y_test = get_X_y_safe(self.test_df, selected_targets)

            dims = self.X_train.shape[1] if self.X_train is not None else 0
            self.append_result(f"✔ Данные готовы: Train={self.X_train.shape[0]}, Test={self.X_test.shape[0]}, Features={dims}\n")
            messagebox.showinfo("Готово", "Данные успешно подготовлены.")

        except Exception as e:
            messagebox.showerror("Ошибка подготовки", str(e))

    def train_model(self):
        if self.X_train is None:
            messagebox.showwarning("Внимание", "Сначала нажмите 'Подготовить данные'")
            return
        
        task = self.task_var.get()
        if task != 'clustering' and self.y_train is None:
             messagebox.showerror("Ошибка", "Нет целевой переменной.")
             return

        try:
            n_centers = int(self.centers_var.get())
            spread_str = self.spread_var.get()
            spread = 'auto' if spread_str == 'auto' else float(spread_str)

            self.append_result(f"⏳ Обучение модели ({task}, centers={n_centers})...\n")
            self.root.update()
            
            self.model = RBFNetwork(n_centers=n_centers, spread=spread)
            self.model.fit(self.X_train, self.y_train, task=task, center_method=self.method_var.get())
            
            self.append_result("✔ Обучение завершено успешно!\n")
        except Exception as e:
            self.append_result(f"❌ ОШИБКА: {e}\n")
            messagebox.showerror("Error", str(e))

    def test_model(self):
        if not self.model: return messagebox.showwarning("", "Модель не обучена")
        if self.X_test is None: return messagebox.showwarning("", "Нет тестовых данных")
        
        self.append_result("🔎 Тестирование...\n")
        try:
            y_pred = self.model.predict(self.X_test)
            metrics = compute_metrics(self.y_test, y_pred, self.model.task, X=self.X_test)
            
            self.append_result("-" * 30 + "\n")
            for k, v in metrics.items():
                self.append_result(f"  {k}: {v:.4f}\n")
            self.append_result("-" * 30 + "\n")
                
            if self.model.task == 'clustering':
                counts = np.bincount(y_pred.astype(int))
                self.append_result(f"  Размеры кластеров: {counts}\n")
                
        except Exception as e:
            self.append_result(f"Ошибка теста: {e}\n")

    def predict_single(self):
        if not self.model: return messagebox.showwarning("", "Нет модели")
        if self.model.X_mean is None: return messagebox.showwarning("", "Модель пуста")

        n_features = self.model.X_mean.shape[0]
        
        d = ttk.Toplevel(self.root)
        d.title("Ручной ввод")
        d.geometry("400x200")
        
        ttk.Label(d, text=f"Введите {n_features} чисел (через пробел):", bootstyle=INFO).pack(pady=10)
        entry = ttk.Entry(d, width=40)
        entry.pack(pady=5)
        
        def run():
            try:
                raw = entry.get().replace(',', ' ')
                vals = np.array([float(x) for x in raw.split()])
                if vals.size != n_features: raise ValueError(f"Нужно {n_features} чисел")
                
                vals = vals.reshape(1, -1)
                res = self.model.predict(vals)
                
                if self.model.task == 'classification':
                    cls = np.argmax(res[0])
                    probs = np.round(res[0], 3)
                    msg = f"Класс: {cls}\nВероятности: {probs}"
                elif self.model.task == 'clustering':
                    msg = f"Кластер: {res[0]}"
                else:
                    msg = f"Значение: {res[0]:.4f}"
                
                messagebox.showinfo("Результат", msg)
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
            
        ttk.Button(d, text="Рассчитать", command=run, bootstyle="success").pack(pady=10)

    def save_model(self):
        if not self.model: return
        f = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")])
        if f:
            with open(f, 'wb') as file: pickle.dump(self.model, file)
            self.append_result(f"💾 Сохранено в {os.path.basename(f)}\n")

    def load_model(self):
        f = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl")])
        if f:
            try:
                with open(f, 'rb') as file: self.model = pickle.load(file)
                self.task_var.set(self.model.task)
                self.centers_var.set(self.model.n_centers)
                if hasattr(self.model, 'spread'): self.spread_var.set(self.model.spread)
                self.append_result(f"📂 Загружена модель: {self.model.task}\n")
                self._on_task_changed(None)
            except Exception as e: messagebox.showerror("Ошибка", str(e))

    def append_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text)
        self.result_text.see("end")
        self.result_text.config(state="disabled")