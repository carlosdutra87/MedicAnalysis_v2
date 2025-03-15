# file_viewer.py
# Contains the FileViewer class, responsible for displaying the files and directories in the selected directory.
import tkinter as tk
from tkinter import ttk, filedialog
import os

class FileViewer(tk.Frame):
    def __init__(self, master=None, image_viewer=None):
        # Initialize the FileViewer class with the given parameters.
        super().__init__(master, bg='white')

        self.image_viewer = image_viewer

        # Create a vertical scroll bar
        self.y_scrollbar = ttk.Scrollbar(self, orient='vertical')
        self.y_scrollbar.pack(side='right', fill='y')

        self.tree = ttk.Treeview(self, yscrollcommand=self.y_scrollbar.set)
        self.tree.pack(expand=True, fill=tk.BOTH)

        # Set up the first layer nodes
        self.tree.heading('#0', text='Directory:', anchor='w')

        # Bind double-click event to load the selected file
        self.tree.bind("<Double-1>", self.load_selected_file)

        self.load_default_directory()  # Load the default directory initially

    def load_default_directory(self): # Load the default directory
        default_directory = "C:/MedicAnalysis"

        if not os.path.exists(default_directory):
            os.makedirs(default_directory)

        default_node = self.tree.insert("", "end", text=default_directory, open=True, tags=('drive',))
        self.load_directory_content(default_directory, default_node)

    def load_selected_file(self, event): # Load the selected file
        item = self.tree.selection()

        if item:
            # Get the values associated with the selected item
            values = self.tree.item(item[0], 'values')

            # Check if there are values and the index is within range
            if values and len(values) > 1:
                # Extract the file path from the values
                file_path = values[1]

                # If it's a drive, load its content
                if 'drive' in self.tree.item(item, 'tags'):
                    self.load_directory_content(file_path, item)
                else:
                    self.image_viewer.load_image(file_path)

    def load_directory_content(self, directory, parent_node): # Load the content of the directory
        self.tree.delete(*self.tree.get_children(parent_node))

        # Add folders and files to the tree
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                folder_node = self.tree.insert(parent_node, "end", text=item, open=False, tags=('folder',))
                self.tree.insert(folder_node, "end", values=("", ""), open=False, tags=('dummy',))  # Dummy item for placeholder
                # Recursively load the content of the subdirectory
                self.load_directory_content(item_path, folder_node)
            else:
                self.tree.insert(parent_node, "end", values=(item, item_path), open=False, tags=('file',))
    
    def clear_file_path(self):
        self.tree.delete(*self.tree.get_children())  # Clear the tree
        self.load_default_directory()  # Reload the default directory