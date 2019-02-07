import numpy as np
import tkinter as tk

from brain import ResultsApp


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1250x800+40+40")
    root.title("Numerical ground state of rotating Bose-Einstein Condensates : Result Presentation")
    app = ResultsApp(root)
    root.mainloop()
