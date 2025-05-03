# profile_app_snakeviz.py
import cProfile
from main import Application
from tkinter import Tk

def run_app():
    root = Tk()
    app = Application(master=root)
    root.mainloop()

# Cria o profiler
pr = cProfile.Profile()

# Executa a aplicação com o profiler ativado
pr.enable()
run_app()
pr.disable()

# Salva em formato .prof (compatível com SnakeViz)
pr.dump_stats("profile_output.prof")