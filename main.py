# --- START OF FILE main.py ---
import ttkbootstrap as ttk
from gui_app import RBFGUIApp

if __name__ == "__main__":
    # Темы на выбор: 'cosmo', 'flatly', 'journal', 'superhero' (темная), 'darkly' (темная)
    app_window = ttk.Window(themename="cosmo") 
    app = RBFGUIApp(app_window)
    app_window.mainloop()