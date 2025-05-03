#data_viewer.py
# File responsible for the data viewer container, which displays the data from the segments in a treeview widget. It also allows the user to load and save data as CSV files.
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, StringVar, Toplevel
import csv
import os
import matplotlib.pyplot as plt 
import math

class DataViewer(ttk.Frame):
    def __init__(self, master, GUI=None):
        # Initialize the DataViewer class with the given parameters.
        super().__init__(master)
        self.GUI = GUI
        self.data_viewer = None

        # ROI Data variables
        self.ROI_data = {}
        self.selected_ROI_index = 0
        self.listes_graphs = []
        self.page_size = 50 
        self.current_page = {}
        self.total_pages = {} 

        tree_frame = ttk.Frame(self)
        tree_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.tree = ttk.Treeview(tree_frame, columns=('Index', 'Means Green','Means Red'), show='headings', selectmode='browse') # Treeview widget to display the data from the segments
        self.tree.heading('#1', text='Index')
        self.tree.column('#1', width=50)  

        self.tree.heading('#2', text='Means Green')
        self.tree.column('#2', width=100)  

        self.tree.heading('#3', text='Means Red')
        self.tree.column('#3', width=100)  
        self.tree.pack(side=tk.LEFT)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')

        self.tree.configure(yscroll=scrollbar.set)

        button_container = ttk.Frame(self)
        button_container.pack(side=tk.TOP, pady=10)

        self.selected_ROI_var = StringVar() # Variable to store the selected ROI
        self.ROI_combobox = ttk.Combobox(button_container, textvariable=self.selected_ROI_var, state="disabled") # Combobox to display the ROIs
        self.ROI_combobox.pack(side=tk.LEFT, padx=10)
        self.ROI_combobox.bind("<<ComboboxSelected>>", self.handle_combobox_selection)

        self.prev_button = ttk.Button(button_container, text="Previous", command=self.show_previous_page) # Button to show the previous page of the data
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = ttk.Button(button_container, text="Next", command=self.show_next_page) # Button to show the next page of the data
        self.next_button.pack(side=tk.LEFT, padx=10)

        csv_buttons_frame = ttk.Frame(self)
        csv_buttons_frame.pack(side=tk.TOP, pady=10)

        self.load_button = ttk.Button(csv_buttons_frame, text="Load CSV", command=self.load_csv) # Button to load a CSV file
        self.load_button.pack(side=tk.LEFT, padx=10)

        self.save_button = ttk.Button(csv_buttons_frame, text="Save as CSV", command=self.save_as_csv) # Button to save the data as a CSV file
        self.save_button.pack(side=tk.LEFT, padx=10)
    
    def disable_functionalities_pre_load(self): # Disable functionalities before loading data
        self.prev_button.config(state='disabled')
        self.next_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.ROI_combobox.config(state='disabled')
    

    def enable_functionalities_post_load(self): # Enable functionalities after loading data
        self.load_button.config(state='normal')

    def process_segment_data(self, segment_data): # Process the segment data and display it in the treeview widget

        if 'disabled' in self.save_button.state():
            self.save_button.config(state='normal')

        if 'disabled' in self.ROI_combobox.state():
            self.ROI_combobox.state(['!disabled', 'readonly'])

        segment_x_array = segment_data['segment_x_array']
        segment_y_array = segment_data['segment_y_array']
        mean_values_green = segment_data['mean_values'][0]
        mean_values_red= segment_data['mean_values'][1]
        coordinates = [(x, y) for x, y in zip(segment_x_array, segment_y_array)]
        means_data = (list(range(1, len(mean_values_green))), mean_values_green[:-1],mean_values_red[:-1])

        if self.ROI_data == {}:
            self.selected_ROI_index = 1
        else:
            self.selected_ROI_index = max(self.ROI_data.keys()) + 1

        if 'disabled' in self.prev_button.state():
            self.prev_button.config(state='normal')

        if 'disabled' in self.next_button.state():
            self.next_button.config(state='normal')

        self.ROI_data[self.selected_ROI_index] = {
            'coord': coordinates, 
            'means': {
                0: [means_data[0], means_data[1]],
                1: [means_data[0], means_data[2]]
                }
            }    

        self.total_pages[self.selected_ROI_index] = math.ceil(len(self.ROI_data[self.selected_ROI_index]['means'][1][0]) / self.page_size)
        self.current_page[self.selected_ROI_index] = 0 

        self.update_ROI_combobox()
        self.selected_ROI_var.set(self.selected_ROI_index)
        self.generate_graphs()
        self.new_selection_display_data()
        
    def data_tree_clean_rebuild(self): # Clean and rebuild the data tree
        self.tree.delete(*self.tree.get_children())
        self.tree.heading('#1', text='Index')
        self.tree.heading('#2', text='Means Green')
        self.tree.heading('#3', text='Means Red') 

    def clean_seg_data(self): # Clean the segment data and reset the ROI data dictionary
        self.ROI_data = {}
        self.data_tree_clean_rebuild()
        self.selected_ROI_index = 0
        self.current_page = {}
        self.total_pages = {}
        self.update_ROI_combobox()    

    def new_selection_display_data(self): # Display the data for the selected ROI
        self.data_tree_clean_rebuild()
        self.display_data()

    # def display_data(self): # Display the data in the treeview widget for the selected ROI
    #     self.selected_ROI_index = int(self.selected_ROI_var.get())
    #     self.update_ROI_combobox()

    #     start_row = self.current_page[self.selected_ROI_index] * self.page_size
    #     end_row = start_row + self.page_size

    #     mean_values_green = self.ROI_data[self.selected_ROI_index]['means'][0]
    #     mean_values_red = self.ROI_data[self.selected_ROI_index]['means'][1]

    #     mean_intensities_green = mean_values_green[start_row:end_row]
    #     mean_intensities_red = mean_values_red[start_row:end_row]   

    #     # for i, (index, mean_intensity_green, mean_intensity_red) in enumerate(zip(range(start_row, end_row), mean_intensities_green[1], mean_intensities_red[1]), start=start_row + 1):
    #     #     self.tree.insert('', 'end', values=[index, round(mean_intensity_green, 2), round(mean_intensity_red, 2)])

    #     # Antes de inserir os novos dados, limpe a Treeview:
    #     self.tree.delete(*self.tree.get_children())

    #     # Agora itere sobre os dados da fatia usando as listas completas:
    #     for index, mean_intensity_green, mean_intensity_red in zip(range(start_row, end_row), mean_intensities_green, mean_intensities_red):
    #         self.tree.insert('', 'end', values=[index, round(mean_intensity_green, 2), round(mean_intensity_red, 2)])

    def display_data(self):  # Display the data in the treeview widget for the selected ROI
        try:
            self.selected_ROI_index = int(self.selected_ROI_var.get())
        except ValueError:
            print("ROI selecionada inválida.")
            return

        start_row = self.current_page[self.selected_ROI_index] * self.page_size
        end_row = start_row + self.page_size

        # Aqui, extraímos a lista de índices e a lista de valores para cada canal
        indices_green = self.ROI_data[self.selected_ROI_index]['means'][0][0]
        mean_values_green = self.ROI_data[self.selected_ROI_index]['means'][0][1]
        indices_red = self.ROI_data[self.selected_ROI_index]['means'][1][0]
        mean_values_red = self.ROI_data[self.selected_ROI_index]['means'][1][1]

        # Faz a fatia para a página atual
        indices_slice = indices_green[start_row:end_row]
        green_slice = mean_values_green[start_row:end_row]
        red_slice = mean_values_red[start_row:end_row]

        # Limpa a treeview antes de inserir novos dados
        self.tree.delete(*self.tree.get_children())

        for index, mg, mr in zip(indices_slice, green_slice, red_slice):
            self.tree.insert('', 'end', values=[index, round(mg, 2), round(mr, 2)])

    def save_as_csv(self): # Save the data as a CSV file
        project_path = self.get_project_path()
        file_path = os.path.join(project_path, "data_viewer_setup.csv")

        with open(file_path, 'w', newline='') as csvfile: # Open the CSV file and write the data
            csv_writer = csv.writer(csvfile)
            image_path = self.GUI.get_image_path()
            image_name = os.path.basename(image_path)
            csv_writer.writerow([f'Image-{image_name}'])
            temp_row = ['ROI', 'Coord', 'Intensity_Mean_Green','Intensity_Mean_Red']
            csv_writer.writerow(temp_row)

            for roi_index, roi_data in self.ROI_data.items():
                row_data = [roi_index, str(roi_data['coord']), str(roi_data['means'][0]),str(roi_data['means'][1])]
                csv_writer.writerow(row_data)

        messagebox.showinfo("Save Successful", f"Data saved to project folder.")

    def load_csv(self, file_path=None): # Load the data from a CSV file 
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                with open(file_path, 'r') as csvfile: # Open the CSV file and read the data
                    csv_reader = csv.reader(csvfile)
                    header = next(csv_reader)
                    if header[0].startswith('Image-'):
                        if header[0] != f'Image-{os.path.basename(self.GUI.get_image_path())}':
                            raise ValueError("Data from different image")
                    else:
                        raise ValueError("Invalid CSV format")

                    data = [row for row in csv_reader]

                    self.process_data_from_csv(header, data)

                    messagebox.showinfo("Load Successful", f"Data loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data from {file_path}: {e}")

    def process_data_from_csv(self, header, data): # Process the data from the CSV file
        self.data_tree_clean_rebuild()
        self.GUI.reset_context_for_segments_csv()
        # Process the new data
        for row in data:
            if row[0] != 'ROI':
                roi_index = int(row[0])
                coord = eval(row[1])
                means_green = eval(row[2])
                means_red=eval(row[3])
                
                means = (means_green, means_red)

                next_index = 1
                if len(self.ROI_data) != 0:
                    next_index = max(self.ROI_data.keys()) + 1

                mean_values_green = means[0]
                mean_values_red = means[1]

                self.ROI_data[next_index] = {
                    'coord': coord,
                    'means': {
                        0: mean_values_green,  
                        1: mean_values_red     
                    }
                }
                self.GUI.draw_segments_from_csv(coord) # Draw the segments from the CSV file

                self.current_page[next_index] = 0
                self.total_pages[next_index] = (len(self.ROI_data[next_index]['means'][1]) / self.page_size) + 1
                self.selected_ROI_index = next_index
            
        self.update_ROI_combobox()
        self.display_data()
        self.generate_graphs()
    
    def generate_graphs(self): # Generate the graphs for the selected ROI
         
        self.graph_viewer.set_ROI_data(self.ROI_data)
        self.graph_viewer.update_displayed_ROI(self.selected_ROI_index)
        self.graph_viewer.process_to_graph(channel=0)  
        self.graph_viewer.process_to_graph(channel=1)

    def update_ROI_combobox(self): # Update the ROI combobox
        self.ROI_combobox['values'] = list(self.ROI_data.keys())
        self.ROI_combobox.set(self.selected_ROI_index)

    def handle_combobox_selection(self, event): # Handle the selection of the ROI combobox
        self.selected_ROI_index = int(self.selected_ROI_var.get())
        self.new_selection_display_data()
        self.graph_viewer.update_displayed_ROI(self.selected_ROI_index)

    def show_previous_page(self): # Show the previous page of the data
        if self.current_page[self.selected_ROI_index] > 0:
            self.current_page[self.selected_ROI_index] -= 1
            self.new_selection_display_data()
    
    def show_next_page(self): # Show the next page of the data
        if self.current_page[self.selected_ROI_index] < (self.total_pages[self.selected_ROI_index] - 1):
            self.current_page[self.selected_ROI_index] += 1
            self.new_selection_display_data()
    
    def get_project_path(self): # Get the project path
        return self.GUI.get_project_path()

    def set_graph_viewer(self, graph_viewer): # Set the graph viewer
        self.graph_viewer = graph_viewer

    def get_dic_ROI(self): # Get the ROI data
        return self.ROI_data
