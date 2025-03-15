# GUI.py
# Central file for the GUI of the software, responsible for creating the main screen of the software and connecting all the different components of the software.
# The GUI class is a subclass of the Toplevel class from the tkinter module. 
# The GUI class contains methods for creating the main screen of the software, including the file explorer, image viewer, data viewer, and graph viewer components. 
# The GUI class also contains methods for handling file operations, such as opening, saving, and creating new files. 
# The GUI class also contains methods for displaying information about the software and handling user input.
from tkinter import ttk
import tkinter.messagebox
import datetime
import tkinter as tk
from tkinter import OptionMenu, StringVar, filedialog
from image_viewer import ImageViewer
from data_viewer import DataViewer  
from graph_viewer import GraphViewer
from file_viewer import FileViewer
from load_screen import LoadScreen

class GUI(tk.Toplevel): 
    def __init__(self, master=None, app=None, image_path=None):
        # Initialize the GUI class with the master window and the application object.
        super().__init__(master) 
        self.title("Main Screen")  
        self.geometry("1200x800")
        self.configure(bg='white')
        
        self.image_viewer = None
        
        self.main_container = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5, sashrelief=tk.SUNKEN) # Main container for the different components of the software
        self.main_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
    
        self.msg = tk.Label( self.main_container, text="Main Screen") 
        self.msg["font"] = ("Verdana", "12", "bold") 

        self.project_path = None
        
        file_explorer_frame = FileViewer( self.main_container, image_viewer=self.image_viewer) # File explorer component for browsing files
        self.main_container.add(file_explorer_frame, minsize=100)  
    
        self.image_viewer = ImageViewer( self.main_container,self) # Image viewer component for displaying images
        self.main_container.add(self.image_viewer, minsize=200)  
        self.image_viewer.disable_functionalities_pre_load()

        self.data_graph_frame = tk.PanedWindow( self.main_container, orient=tk.VERTICAL, sashwidth=8, sashrelief=tk.SUNKEN) # Data and graph viewer components
        self.main_container.add(self.data_graph_frame, minsize=200) 

        self.data_viewer = DataViewer(self.data_graph_frame, self) # Data viewer component for displaying data
        self.image_viewer.set_data_viewer(self.data_viewer)
        self.data_graph_frame.add(self.data_viewer, minsize=200)  
        self.data_viewer.disable_functionalities_pre_load()

        self.graph_viewer = GraphViewer(self.data_graph_frame) # Graph viewer component for displaying graphs
        self.graph_viewer.disable_functionalities_pre_load()
        self.image_viewer.set_graph_viewer(self.graph_viewer)
        self.data_viewer.set_graph_viewer(self.graph_viewer)
        self.data_graph_frame.add(self.graph_viewer, minsize=200)  
        
        self.load_screen = LoadScreen(self.master, app=self, image_viewer=self.image_viewer) # Load screen component for loading images and creating projects
        
        self.menubar = tk.Menu(self) # Menu bar for the software 
        self.master.config(menu=self.menubar)
        
        self.file_menu = tk.Menu(self.menubar, tearoff=0) # File menu in the menu bar
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.open_file_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Open File", menu=self.open_file_menu)
        self.open_file_menu.add_command(label="Open CSV", command=lambda: self.open_specific_file("csv"))
        self.open_file_menu.add_command(label="Open TIFF", command=lambda: self.open_specific_file("tiff"))

        self.open_recent_menu = tk.Menu(self.file_menu, tearoff=0) 
        self.file_menu.add_cascade(label="Open Recent", menu=self.open_recent_menu)

        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save All", accelerator="Ctrl+K", command=self.save_all)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", accelerator="Ctrl+E", command=self.quit_software)

        self.edit_menu = tk.Menu(self.menubar, tearoff=0) # Edit menu in the menu bar
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        
        self.view_menu = tk.Menu(self.menubar, tearoff=0) # View menu in the menu bar
        self.menubar.add_cascade(label="View", menu=self.view_menu)

        self.roi_menu = tk.Menu(self.view_menu, tearoff=0) 
        self.view_menu.add_cascade(label="ROI Segmentations", menu=self.roi_menu)

        self.roi_visibility_vars = []

        self.help_menu = tk.Menu(self.menubar, tearoff=0) # Help menu in the menu bar
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        self.config(menu=self.menubar)
        self.menubar.entryconfig("File", state="disabled") # Disable the file menu until an image is loaded
    
        self.night_mode = tk.BooleanVar()
        self.night_mode.set(False)

        self.bind_shortcuts()
        
        self.recent_files = [] 
        self.protocol("WM_DELETE_WINDOW", self.quit_software) 
    
    def update_ROI_visibility_list(self): # Update the list of ROI segmentations in the view menu
        self.roi_visibility_vars.clear()
        self.roi_menu.delete(0, tk.END)  
        for i in range(len(self.image_viewer.ROI_objects)):
            visibility_var = tk.BooleanVar(value=True)
            if (not self.image_viewer.ROI_objects[i]['seg'][0].get_visible()):
                visibility_var = tk.BooleanVar(value=False)
            self.roi_visibility_vars.append(visibility_var)
            self.roi_menu.add_checkbutton(label=f"ROI {i+1}", variable=visibility_var,
                                     command=lambda index=i: self.toggle_roi_visibility(index))

    def toggle_roi_visibility(self, index): # Toggle the visibility of a specific ROI segmentation
        visibility = self.roi_visibility_vars[index].get()
        self.image_viewer.toggle_ROI_visibility(index, visibility)

    def quit_software(self): # Quit the software
        self.quit()
    
    def enable_functionalities_post_load(self): # Enable functionalities after loading an image
        self.image_viewer.enable_functionalities_post_load()
        self.data_viewer.enable_functionalities_post_load()
        self.graph_viewer.enable_functionalities_post_load()
        self.menubar.entryconfig("File", state="normal")

    def get_project_path(self): # Get the path of the project
        return self.load_screen.get_project_path()
    
    def get_dic_ROI(self):  # Get the dictionary of ROI segmentations
        return self.data_viewer.get_dic_ROI()
    
    def get_image_path(self): # Get the path of the image
        return self.load_screen.get_image_path()
    
    def draw_segments_from_csv(self, coordinates): # Draw segments from a CSV file
        self.image_viewer.draw_segments_from_csv(coordinates)

    def reset_context_for_segments_csv(self): # Reset the context for the segments CSV file
        self.image_viewer.reset_context_for_segments_csv()

    def bind_shortcuts(self): # Bind keyboard shortcuts to different functions
        self.bind("<Control-K>", lambda event: self.save_all())
        self.bind("<Alt-F4>", lambda event: self.quit())
        self.bind("<Control-E>", lambda event: self.quit())
        self.bind("<Control-R>", lambda event: self.open_recent_file(self.recent_files[0]) if self.recent_files else None)
        
    def open_specific_file(self, file_type): # Open a specific file type (CSV or TIFF) for file menu
        file_extension = "*.csv" if file_type == "csv" else "*.tiff"
        file_path = filedialog.askopenfilename(filetypes=[(f"{file_type.upper()} Files", file_extension)])
        if file_path:
            if file_type == "csv":
                self.data_viewer.load_csv(file_path) 
            else:
                self.clean_display_seg_data()
                self.image_viewer.load_image(file_path)
            with open(self.load_screen.identification_file_path, "w") as identification_file:
                identification_file.write(f"\nPath-{self.load_screen.selected_file}")
    
    def clean_display_seg_data(self): # Clean the display of the segmentation and its data
        self.data_viewer.clean_seg_data()
        self.image_viewer.clean_display_seg()
            
    def update_recent_files(self, file_path): # Update the list of recent files
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)

        self.recent_files = self.recent_files[:5]
    
        self.update_open_recent_menu()
        
    def update_open_recent_menu(self): # Update the open recent menu
        self.open_recent_menu.delete(0, tk.END)

        for i, file_path in enumerate(self.recent_files):
            accelerator = f"Ctrl+{i + 1}" if i < 9 else ""  
            self.open_recent_menu.add_command(
                label=f"Recent File {i + 1}",
                accelerator=accelerator,
                command=lambda path=file_path: self.open_recent_file(path)
            )
            
    def open_recent_file(self, file_path): # Open a recent file
        self.image_viewer.load_image(file_path)
        self.load_screen.selected_file = file_path
        self.update_recent_files(file_path)

    def save_all(self): # Save all the data and ROI segmentations
        if self.data_viewer.ROI_data:
            self.data_viewer.save_as_csv()
        with open(self.load_screen.identification_file_path, "w") as identification_file:
            identification_file.write(f"\nProject: {self.load_screen.project_name}")
            identification_file.write(f"\nPath-{self.load_screen.selected_file}")
    
    def show_about(self): # Show information about the software
        about_info = (
            "MedicAnalysis\n\n"
            "Author: Aya, Carlos, Marc, Tom\n"
            "Version: 1.0\n"
            f"Date: {datetime.date.today()}\n"
            "Copyright © 2024 PE BMD. All rights reserved."
        )

        tkinter.messagebox.showinfo("About", about_info)