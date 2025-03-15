# graph_viewer.py
# Contains the GraphViewer class, responsible for displaying the graph of the selected ROI. 
# The user can select the color channel to display and compare the ROIs.
import tkinter as tk
from tkinter import messagebox, ttk, Toplevel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt 
import csv
import ast  
from tkinter import Checkbutton

class GraphViewer(tk.Frame):
    def __init__(self, master=None):
        # Initialize the GraphViewer class with the given parameters.
        super().__init__(master)

        # Data structures
        self.ROI_data = {}
        self.selected_ROI_index = 1
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.selected_color = ''
        self.id_color = -1

        self.image_container = tk.Frame(self)
        self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a matplotlib figure and axis
        self.figure, self.axis = Figure(figsize=(4, 2), tight_layout=True), None
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.color_mode = 'grayscale'
        self.is_color_changed = False
    
        self.compare_button = ttk.Button(self, text="Compare ROIs", command=self.compare_ROIs)
        self.compare_button.pack(side=tk.TOP, padx=10, pady=10)

        # Checkbuttons for displaying red and green channels
        self.display_red_var = tk.BooleanVar()
        self.display_red_var.set(False)  
        self.display_green_var = tk.BooleanVar()
        self.display_green_var.set(False) 

        self.display_red_checkbox = tk.Checkbutton(self, text="Display Red Channel", variable=self.display_red_var,
                                                    command=lambda: self.update_displayed_ROI(self.selected_ROI_index))
        self.display_red_checkbox.pack(side=tk.TOP, padx=10, pady=5)

        self.display_green_checkbox = tk.Checkbutton(self, text="Display Green Channel", variable=self.display_green_var,
                                                    command=lambda: self.update_displayed_ROI(self.selected_ROI_index))
        self.display_green_checkbox.pack(side=tk.TOP, padx=15, pady=5)

    def clean_graph(self): # Clear the graph for new data
        self.axis.clear()
        self.canvas.draw()

    def disable_functionalities_pre_load(self): # Disable functionalities before loading the image
        self.compare_button.config(state="disabled")

    def enable_functionalities_post_load(self): # Enable functionalities after loading the image
        self.compare_button.config(state="normal")

    def compare_ROIs(self): # Function for comparing ROIs
        if len(self.ROI_data) < 2:
            messagebox.showinfo("Error", "There must be at least two ROIs available for comparison.")
            return

        compare_window = Toplevel(self) # Create a new window for comparing ROIs
        compare_window.title("Compare ROIs")
        compare_window.geometry("600x400")

        fig, ax = plt.subplots() 

        def plot_comparison(selected_ROIs): # Plot the comparison of the selected ROIs
            ax.clear() 
            liste=list(range(0, len(self.ROI_data)))
            print(liste)
            
            for ROI_index, i in zip(selected_ROIs, liste):
                if ROI_index in self.ROI_data:

                    y_data_green = self.ROI_data[ROI_index]['means'][0][1]
                    y_data_red=self.ROI_data[ROI_index]['means'][1][1]
                    x_data = self.ROI_data[ROI_index]['means'][0][0]
                    x_data = [float(x) for x in x_data if isinstance(x, (int, float))]
                    y_data_green = [float(y) for y in y_data_green if isinstance(y, (int, float))]
                    y_data_red = [float(y) for y in y_data_red if isinstance(y, (int, float))]
                    
                    if green_vars[i].get():
                        ax.plot(x_data, y_data_green, label=f'ROI {ROI_index} GREEN')
                    if red_vars[i].get():
                        ax.plot(x_data, y_data_red, label=f'ROI {ROI_index} RED')

            compare_window.destroy() 
            ax.set_xlabel('Index')
            ax.set_ylabel('Mean Intensity')
            ax.set_title('ROI Comparison')
            ax.legend()
            plt.show()

        def add_combobox(): # Add a combobox for selecting an ROI to compare 
            if len(comboboxes) < len(self.ROI_data):
                combobox = ttk.Combobox(compare_window, values=list(self.ROI_data.keys()), state="readonly")
                combobox.grid(row=len(comboboxes), column=0, padx=5, pady=5)
                comboboxes.append(combobox)

                green_var = tk.BooleanVar()
                green_checkbox = tk.Checkbutton(compare_window, text="Green", variable=green_var)
                green_checkbox.grid(row=len(comboboxes)-1, column=1, padx=5, pady=5)
                green_vars.append(green_var)

                red_var = tk.BooleanVar()
                red_checkbox = tk.Checkbutton(compare_window, text="Red", variable=red_var)
                red_checkbox.grid(row=len(comboboxes)-1, column=2, padx=5, pady=5)
                red_vars.append(red_var)
            else:
                messagebox.showinfo("Error", "You have already added all available ROIs. Add more ROIs to be able to compare more.")

        
        comboboxes = []
        green_vars = []
        red_vars = []
        add_combobox()
        add_combobox()

        compare_button = ttk.Button(compare_window, text="Compare", command=lambda: plot_comparison([int(combobox.get()) for combobox in comboboxes if combobox.get()]))
        compare_button.grid(row=len(comboboxes), column=4, padx=5, pady=5)

        add_button = ttk.Button(compare_window, text="+", command=add_combobox)
        add_button.grid(row=0, column=3, padx=5, pady=5)

    def toggle_dataset(self, index): # Toggle the visibility of the dataset corresponding to the given index
        if self.axis is not None:
            lines = self.axis.lines
            if index < len(lines):
                line = lines[index]
                line.set_visible(self.datasets_visibility[index].get())  # Get visibility from checkbox variable
                self.canvas.draw()

    def set_ROI_data(self, ROI_dict): # Set the ROI data 
        self.ROI_data = ROI_dict
    
    def update_displayed_ROI(self, ROI_index): # Update the displayed ROI from the selection
        self.selected_ROI_index = ROI_index
        display_red = self.display_red_var.get()
        display_green = self.display_green_var.get()
        if self.axis !=None :
            self.axis.clear()
            self.canvas.draw()
        
        if display_green:
            self.process_to_graph(channel=0)
            
        if display_red:
            self.process_to_graph(channel=1)

    def process_to_graph(self, channel=0): # Process the data to be displayed on the graph
        if self.axis is None:
            self.axis = self.figure.add_subplot(111)

        if not self.ROI_data:
            print("Error: No ROI data available.")
            return

        if self.selected_ROI_index not in self.ROI_data:
            print(f"Error: ROI {self.selected_ROI_index} not found in ROI data.")
            return

        try:
            y_data = self.ROI_data[self.selected_ROI_index]['means'][channel][1]
            x_data = self.ROI_data[self.selected_ROI_index]['means'][channel][0]
            x_data = [float(x) for x in x_data]
            y_data = [float(y) for y in y_data]

            if channel == 0:
                selected_color = 'green'
            else:
                selected_color = 'red'

            self.axis.plot(x_data, y_data, color=selected_color, linestyle='-', label=f'ROI {self.selected_ROI_index}')

            self.axis.set_xlabel('Index')
            self.axis.set_ylabel('Mean Intensity')
            self.axis.set_title('Graph Viewer')

            self.axis.set_xlim([0, len(x_data)]) 
            self.axis.set_ylim([1000, 1500]) 

            self.canvas.draw()
        except Exception as e:
            print(f"Error: Unable to process data: {e}")