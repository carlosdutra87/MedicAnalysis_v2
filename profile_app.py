# profile_app_snakeviz.py
import cProfile
import pstats
from main import Application
from tkinter import Tk

def run_app():
    root = Tk()
    app = Application(master=root)
    root.mainloop()

pr = cProfile.Profile()

pr.enable()
run_app()
pr.disable()

pr.dump_stats("profile_output.prof")

with open("profile_output.txt", "w") as f:
    ps = pstats.Stats(pr, stream=f)
    ps.sort_stats("cumulative") 
    ps.print_stats()
