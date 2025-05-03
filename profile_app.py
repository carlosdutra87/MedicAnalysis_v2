# profile_app_snakeviz.py
import cProfile
import pstats
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

# Salva o perfil em formato .prof (para SnakeViz)
pr.dump_stats("profile_output.prof")

# Também salva em formato .txt legível
with open("profile_output.txt", "w") as f:
    ps = pstats.Stats(pr, stream=f)
    ps.sort_stats("cumulative")  # ou 'time', 'calls', etc.
    ps.print_stats()
