import ttkbootstrap as ttk
from app.gui.window import RBFGUIApp
from app.config import THEME, APP_TITLE, WINDOW_SIZE

if __name__ == "__main__":
    app_window = ttk.Window(themename=THEME)
    app_window.title(APP_TITLE)
    app_window.geometry(WINDOW_SIZE)
    
    app = RBFGUIApp(app_window)
    app_window.mainloop()