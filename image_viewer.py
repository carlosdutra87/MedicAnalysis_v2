# image_viewer.py
# File containing the ImageViewer class for processing and displaying images, prepare the segmentations and configure the image display parameters.
import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import os
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import ImageSequence
import matplotlib.pyplot as plt
from tkinter import PhotoImage
import tifffile 
from tkinter import messagebox
import matplotlib.text as text
from pprint import pprint
from cython_kernels.kernels import fast_mean_pixels, fast_temporal_average, normalize_stack

class ImageViewer(ttk.Frame):
    # Initialize the ImageViewer class with the given parameters.
    def __init__(self, master,GUI=None):
        super().__init__(master)
        self.canaux = {0:np.zeros((1,1,1)), 1:np.zeros((1,1,1))}
        self.current_index = 0 #Index for Slider
        self.normalized_image_array_red = np.zeros((1,1,1))
        self.normalized_image_array_green = np.zeros((1,1,1))
       
        self.GUI=GUI        
        self.is_image_black = True
        self.saturation_pixels_min = 0.1
        self.saturation_pixels_max = 99.8
        self.contrast_value = 0
        self.brightness_value = 0        

        self.data_viewer = None
        self.graph_viewer=None

        self.current_time = tk.DoubleVar()
        self.current_time.set(0)
        
        # Add a variable to track whether panning mode is enabled
        self.panning_mode_enabled = False
        self.panning_start_x = None
        self.panning_start_y = None
        
        # Add a variable to track whether the user is currently zooming
        self.zooming = False

        # Image array
        self.normalized_image_array = np.array([])
        self.selected_index = 0

        # Placeholder image
        self.placeholder_image = Image.new("RGB", (400, 350), "lightgray")
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_image)
        
        # Create a container for the image and parameters
        self.image_container = ttk.Frame(self)
        self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a Matplotlib figure and axis for image display
        self.figure = plt.Figure(figsize=(5, 5))
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.image_container)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.draw()
        
        # Label to display current parameters
        self.parameters_label = tk.Label(self.image_container, text="Contrast: 0\nBrightness: 0\nThreshold Min: 0\nThreshold Max: 0  ", bd=1, relief=tk.SOLID, width=20, height=4)
        self.parameters_label.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)

        self.selected_channel = 1

        # Red channel button
        self.color_mode = 'gray'
        self.red_button = tk.Button(self.image_container, text="Red Channel", command=self.select_red_channel)
        self.red_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)
        self.red_button.config(relief=SUNKEN)
        self.is_red_button = True

        # Green channel button
        self.color_mode = 'gray'
        self.green_button= tk.Button(self.image_container, text="Green Channel", command=self.select_green_channel)
        self.green_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)
        self.is_green_button = False

        # Color change button
        self.color_mode = 'gray'
        self.is_color_changed = False
        self.color_change_button = tk.Button(self.image_container, text="Change Color", command=self.change_color)
        self.color_change_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)
        
        # Bind events for zoom using mouse wheel
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Bind events for panning
        self.canvas.mpl_connect('button_press_event', self.on_pan_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_pan_motion)
        self.canvas.mpl_connect('button_release_event', self.on_pan_release)
        
        # Bind events for zoom start and end
        self.canvas.mpl_connect('button_press_event', self.on_zoom_start)
        self.canvas.mpl_connect('button_release_event', self.on_zoom_end)
        
        # Zoom selection variables
        self.zoom_start_x = None
        self.zoom_start_y = None
        self.zoom_rect = None
        
        # Add a variable to track zoom mode
        self.zoom_mode_enabled = False
        
        self.zoom_mode = False  # Initialize zoom_mode attribute
        
        # Bind events for zoom selection
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        # Load magnifying glass icons using PIL
        zoom_in_image = Image.open("zoom-in.png")
        zoom_out_image = Image.open("zoom-out.png")

        # Resize the images if needed
        icon_size = (20, 20)
        zoom_in_image = zoom_in_image.resize(icon_size, Image.LANCZOS)
        zoom_out_image = zoom_out_image.resize(icon_size, Image.LANCZOS)
        
        # Create PhotoImage objects
        self.zoom_in_icon = ImageTk.PhotoImage(zoom_in_image)
        self.zoom_out_icon = ImageTk.PhotoImage(zoom_out_image)
        
        # Zoom in button
        self.zoom_in_button = ttk.Button(self.image_container, image=self.zoom_in_icon, command=self.zoom_in)
        self.zoom_in_button.image = self.zoom_in_icon  # Keep a reference
        self.zoom_in_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)

        # Zoom out button
        self.zoom_out_button = ttk.Button(self.image_container, image=self.zoom_out_icon, command=self.zoom_out)
        self.zoom_out_button.image = self.zoom_out_icon  # Keep a reference
        self.zoom_out_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)
        
        # Segmentation button with icon
        segmentation_icon = Image.open("segmentation.png")
        segmentation_icon = segmentation_icon.resize((20, 20), Image.LANCZOS)
        segmentation_photo = ImageTk.PhotoImage(segmentation_icon)

        self.segmentation_button = ttk.Button(self.image_container, command=self.toggle_segmentation_mode, image=segmentation_photo, compound="left")
        self.segmentation_button.image = segmentation_photo  # Keep a reference
        self.segmentation_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)

        # Segmentation variables
        self.segmentation_mode_enabled = False
        self.segment_x_points = []
        self.segment_y_points = []
        self.x_in = []
        self.y_in = []
        self.listes_graphs=[]
        self.temp_ROI_points = [[] for _ in range(250)]
        self.ROI_index = 0
        self.ROI_objects = {}
        self.id_color=-1 
        self.marker_styles_point = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'w.']
        self.marker_styles_line = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'w-']

        #Command Show_tags
        self.show_tags_var = tk.BooleanVar()
        self.show_tags_var.set(False) 

        self.show_tags_checkbutton = tk.Checkbutton(self.image_container, text="Show Tags", variable=self.show_tags_var, command=self.update_displayed_image)
        self.show_tags_checkbutton.pack()

        self.tag_artists = []

        # Bind event for segmentation
        self.canvas.mpl_connect('button_press_event', self.on_segment_click)
        
        # Zoom factor for zooming in and out
        self.zoom_factor = 1.1

        # Set initial zoom level
        self.current_zoom_level = 1.0
        self.previous_x = None
        self.previous_y = None
        
        # Create a container for parameters and sliders
        self.parameters_container = ttk.Frame(self)
        self.parameters_container.pack(side=tk.TOP, pady=1)

        # Brightness adjustment
        self.brightness_label = ttk.Label(self.parameters_container, text="Brightness=")
        self.brightness_label.grid(row=0, column=0, padx=5, pady=5)
        self.brightness_slider = ttk.Scale(self.parameters_container, from_=-100, to=100, orient=tk.HORIZONTAL, command=self.update_brightness)
        self.brightness_slider.grid(row=0, column=1, padx=10, pady=5)

        # Contrast adjustment
        self.contrast_label = ttk.Label(self.parameters_container, text="Contrast=")
        self.contrast_label.grid(row=0, column=2, padx=5, pady=5)
        self.contrast_slider = ttk.Scale(self.parameters_container, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_contrast)
        self.contrast_slider.grid(row=0, column=3, padx=10, pady=5)

        # Temporal averaging entry
        self.moyennage_label = ttk.Label(self.parameters_container, text="Temporal averaging=")
        self.moyennage_label.grid(row=0, column=4, padx=5, pady=5)
        self.moyennage_entry = ttk.Entry(self.parameters_container, width=10)
        self.moyennage_entry.grid(row=0, column=5, padx=10, pady=5)
        self.moyennage_entry.bind("<Return>", self.update_window_size)

        # Threshold adjustment
        self.threshold_min_label = ttk.Label(self.parameters_container, text="Min Threshold=")
        self.threshold_min_label.grid(row=1, column=0, padx=5, pady=5)
        self.threshold_min_slider = ttk.Scale(self.parameters_container, from_=0, to=254, orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_min_slider.grid(row=1, column=1,padx=10,pady=5)

        self.threshold_max_label = ttk.Label(self.parameters_container,text="Max Threshold=")
        self.threshold_max_label.grid(row=1,column=2,padx=5,pady=5)
        self.threshold_max_slider = ttk.Scale(self.parameters_container ,from_=1,to =255 ,orient=tk.HORIZONTAL ,command=self.update_threshold )
        self.threshold_max_slider.grid(row = 1,column = 3,padx = 10,pady = 5 )
        self.threshold_max_slider.set(255)

        # Save button
        self.save_button = tk.Button(self.parameters_container,text="Apply to all images ",command=self.save_parameters )
        self.save_button.grid(row = 1,column = 4,pady = 5 )
        # Temporal averaging variables
        self.window_size = 1  # Initial window size
        self.images_for_temporal_averaging = []  # List to store images for averaging
                
        # Keep a reference to the original image for reset functionality
        self.original_image = None
        self.image = None
        self.color_mode = 'gray'  # Initial color mode
        
        self.reset_image()
        self.canaux = {}

    def disable_functionalities_pre_load(self): # Disable functionalities before loading the image
        self.red_button.config(state="disabled")
        self.green_button.config(state="disabled")
        self.color_change_button.config(state="disabled")
        self.zoom_in_button.config(state="disabled")
        self.zoom_out_button.config(state="disabled")
        self.segmentation_button.config(state="disabled")
        self.contrast_slider.config(state="disabled")
        self.brightness_slider.config(state="disabled")
        self.threshold_min_slider.config(state="disabled")
        self.threshold_max_slider.config(state="disabled")
        self.moyennage_entry.config(state="disabled")
        self.save_button.config(state="disabled")
        self.show_tags_checkbutton.config(state="disabled")
    
    def enable_functionalities_post_load(self): # Enable functionalities after loading the image
        self.red_button.config(state="normal")
        self.green_button.config(state="normal")
        self.color_change_button.config(state="normal")
        self.zoom_in_button.config(state="normal")
        self.zoom_out_button.config(state="normal")
        self.segmentation_button.config(state="normal")
        self.contrast_slider.config(state="normal")
        self.brightness_slider.config(state="normal")
        self.threshold_min_slider.config(state="normal")
        self.threshold_max_slider.config(state="normal")
        self.moyennage_entry.config(state="normal")
        self.save_button.config(state="normal")
        self.show_tags_checkbutton.config(state="normal")

    def set_data_viewer(self, data_viewer): # Set the data viewer
        self.data_viewer = data_viewer

    def set_graph_viewer(self, graph_viewer): # Set the graph viewer
        self.graph_viewer = graph_viewer

    def save_parameters(self): # Save the parameters to all images
        
        self.save_button.config(relief=SUNKEN)
        if self.selected_channel == 0:
            self.canaux[0] = np.clip(self.normalized_image_array_green * (1.0 + self.contrast_value)+ self.brightness_value, 0, 255).astype(np.uint8)
            self.canaux[0][self.normalized_image_array_green<self.threshold_min] = 0
            self.canaux[0][self.normalized_image_array_green>self.threshold_max] = 255
        elif self.selected_channel == 1:
            self.canaux[1] = np.clip(self.normalized_image_array_red * (1.0 + self.contrast_value)+ self.brightness_value, 0, 255).astype(np.uint8)
            self.canaux[1][self.normalized_image_array_red<self.threshold_min] = 0
            self.canaux[1][self.normalized_image_array_red>self.threshold_max] = 255
        self.save_button.config(relief=RAISED)

    def clean_display_seg(self): # Clear the segmentations from the display for new data
        self.axis.clear()
        self.ROI_objects = {}
        self.ROI_index = 0
        self.update_ROI_visibility_list()
        self.show_tags_var.set(False)

    # def load_image(self, image_path): # Load the image from the given path
    #     self.original_image = tifffile.imread(image_path)
    #     image_width, image_height = self.original_image[0,0].shape
        
    #     red_images = np.copy(self.original_image[1]) 
    #     green_images = np.copy(self.original_image[0])
        
    #     self.placeholder_image = Image.new("RGB", (image_width, image_height), "lightgray")
    #     self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_image)
   
    #     self.image_container = tk.Frame(self, width=image_width, height=image_height)
    #     self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True) 
        
    #     min_value_red = np.percentile(red_images[0],0.1) 
    #     max_value_red = np.percentile(red_images[0],99.8)
    #     self.normalized_image_array_red = (255*np.clip(red_images,min_value_red,max_value_red)/(max_value_red- min_value_red)).astype(np.uint8) 
        
    #     del red_images
    #     min_value_green = np.percentile(green_images[0],0.1)
    #     max_value_green = np.percentile(green_images[0],99.8)
    #     self.normalized_image_array_green = (255*np.clip(green_images,min_value_green,max_value_green)/(max_value_green- min_value_green)).astype(np.uint8) 
        
    #     if np.mean(green_images[0])>2**15: # Check if the image is inverted
    #         self.normalized_image_array_green = 255- self.normalized_image_array_green
    #         self.normalized_image_array_red = 255- self.normalized_image_array_red
    #         self.is_image_black = False

    #     del green_images

    #     self.canaux = {0: np.copy(self.normalized_image_array_green),
    #                     1: np.copy(self.normalized_image_array_red)}
        
    #     self.image_display = self.axis.imshow(self.canaux[self.selected_channel][self.current_index], cmap=self.color_mode)
        
    #     self.image_display = self.axis.imshow(self.normalized_image_array_red[self.current_index], cmap='gray')
    #     ax_slider = self.figure.add_axes([0.2, 0.05, 0.65, 0.03])
    #     self.slider = Slider(ax_slider, 'Image', 0, self.canaux[1].shape[0]-1, valinit=0)
    #     print(self.canaux[1].shape[0])
    #     self.slider.on_changed(self.update_image)

    def load_image(self, image_path):
        self.original_image = tifffile.imread(image_path)
        image_width, image_height = self.original_image[0, 0].shape

        green_stack = self.original_image[0]
        red_stack = self.original_image[1]

        green_u16 = green_stack.astype(np.uint16)
        red_u16   = red_stack.astype(np.uint16)

        self.normalized_image_array_green = normalize_stack(green_u16, 0.1, 99.8)
        self.normalized_image_array_red   = normalize_stack(red_u16,   0.1, 99.8)

        if np.mean(green_stack[0]) > 2**15:
            self.normalized_image_array_green = 255 - self.normalized_image_array_green
            self.normalized_image_array_red = 255 - self.normalized_image_array_red
            self.is_image_black = False

        self.placeholder_image = Image.new("RGB", (image_width, image_height), "lightgray")
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_image)
        self.image_container = tk.Frame(self, width=image_width, height=image_height)
        self.image_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canaux = {
            0: self.normalized_image_array_green,
            1: self.normalized_image_array_red
        }

        self.image_display = self.axis.imshow(
            self.canaux[self.selected_channel][self.current_index],
            cmap=self.color_mode
        )

        ax_slider = self.figure.add_axes([0.2, 0.05, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Image', 0, self.canaux[1].shape[0] - 1, valinit=0)
        self.slider.on_changed(self.update_image)

    def update_image(self,value = None): # Update the image display
        self.current_index = int(self.slider.val)
        self.image_display.set_data(self.canaux[self.selected_channel][self.current_index])
        self.update_displayed_image()
        
    def update_time_slider(self, max_time): # Update the time slider
        self.time_slider.configure(to=max_time)
        
    def get_parameters(self): # Get the current parameters
        moyennage = int(self.moyennage_entry.get())
        contrast = self.contrast_slider.get()
        return moyennage, contrast
    
    def update_contrast(self, *args): # Update the contrast value
        self.contrast_value = round(self.contrast_slider.get() / 100.0, 2)  # Round to two decimal places

        # Apply contrast adjustment
        self.contrast_brightness_threshold()
        self.update_displayed_image()
        self.update_parameters_label()

    def contrast_brightness_threshold(self): # Apply contrast, brightness, and threshold adjustments
        if self.selected_channel == 0:
            self.canaux[0][self.current_index] = np.clip(self.normalized_image_array_green[self.current_index] * (1.0 + self.contrast_value)+ self.brightness_value, 0, 255).astype(np.uint8)  # Normalize to [0, 1]
            self.canaux[0][self.current_index][self.normalized_image_array_green[self.current_index]<self.threshold_min] = 0
            self.canaux[0][self.current_index][self.normalized_image_array_green[self.current_index]>self.threshold_max] = 255
        elif self.selected_channel == 1:
            self.canaux[1][self.current_index] = np.clip(self.normalized_image_array_red[self.current_index] * (1.0 + self.contrast_value)+ self.brightness_value, 0, 255).astype(np.uint8)  # Normalize to [0, 1]
            self.canaux[1][self.current_index][self.normalized_image_array_red[self.current_index]<self.threshold_min] = 0
            self.canaux[1][self.current_index][self.normalized_image_array_red[self.current_index]>self.threshold_max] = 255
    
    def update_brightness(self, *args): # Update the brightness value
        self.brightness_value = round(self.brightness_slider.get(), 2)  # Round to two decimal places
        self.contrast_brightness_threshold()
        self.update_displayed_image()
        self.update_parameters_label()

    def load_image_at_index(self, time): # Load the image at the given time index
        if not self.normalized_image_array_red.any():
            return  # No images loaded yet

        if time < 0 or time >= len(self.normalized_image_array_red):
            return  # Handle the case when the specified time is out of bounds

        selected_image_path = self.normalized_image_array_red[time]
        self.update_displayed_image()
        
    def update_image_slider(self): # Update the image slider 
        self.selected_index = int(self.image_slider.get())
        print("Selected Index:", self.selected_index)
        self.update_displayed_image(self.selected_index)
    
    def reset_image(self): # Reset the image to its original state
        if self.original_image:
            self.image = self.normalized_image_array_red.copy()
            self.image_slider.set(0)
            self.contrast_slider.set(0)  # Update the contrast slider
            self.brightness_slider.set(0)  # Update the brightness slider
            self.threshold_min_slider.set(0)  # Update the min threshold slider
            self.threshold_max_slider.set(1)  # Update the max threshold slider
            self.axis.clear()
            self.axis.imshow(self.image, cmap='gray')
            self.canvas.draw_idle()
            self.color_change_button.pack(side=tk.TOP, anchor=tk.SW, padx=10, pady=10)  # Pack the button again
            
    def update_window_size(self, val = None): # Update the window size for temporal averaging
        window_size = int(self.moyennage_entry.get())
        self.window_size = window_size
        self.temporal_averaging()
        self.update_displayed_image() 

    # def temporal_averaging(self): # Apply temporal averaging to the images
    #     nb_images = self.canaux[1].shape[0]
    #     for i in range(nb_images-self.window_size+1):
    #         self.normalized_image_array_green[i] = np.mean(self.normalized_image_array_green[i:i+self.window_size,:,:], axis = 0)
    #         self.normalized_image_array_red[i] = np.mean(self.normalized_image_array_red[i:i+self.window_size,:,:], axis = 0)
    #     self.canaux = {0: np.copy(self.normalized_image_array_green),
    #                     1: np.copy(self.normalized_image_array_red)}

    def temporal_averaging(self):
        window = self.window_size
        stack_g = self.normalized_image_array_green   # uint8[T,H,W]
        stack_r = self.normalized_image_array_red     # uint8[T,H,W]

        out_g = fast_temporal_average(stack_g, window)
        out_r = fast_temporal_average(stack_r, window)

        self.normalized_image_array_green[:out_g.shape[0]] = out_g
        self.normalized_image_array_red[:out_r.shape[0]] = out_r
        self.canaux = {0: out_g.copy(), 1: out_r.copy()}

    def update_threshold(self, *args): # Update the threshold values
        self.threshold_min = round(self.threshold_min_slider.get())  
        self.threshold_max = round(self.threshold_max_slider.get())  

        if self.threshold_min >= self.threshold_max:
            # Adjust the values to ensure threshold_min is always less than threshold_max
            self.threshold_max = max(self.threshold_max, self.threshold_min + 1)  # Adjust threshold_max to be slightly higher than threshold_min
            self.threshold_max_slider.set(self.threshold_max)  # Update the slider value
        if self.threshold_max <= self.threshold_min:
            # Adjust the values to ensure threshold_max is always greater than threshold_min
            self.threshold_min = min(self.threshold_min, self.threshold_max - 1)  # Adjust threshold_min to be slightly lower than threshold_max
            self.threshold_min_slider.set(self.threshold_min)  # Update the slider value
        self.contrast_brightness_threshold()
        self.update_displayed_image()  
        self.update_parameters_label()
        

    def update_parameters_label(self): # Update the parameters label with the current values
        parameters_text = f"Contrast: {self.contrast_value}\nBrightness: {self.brightness_value}\nThreshold Min: {self.threshold_min}\nThreshold Max: {self.threshold_max}  "
        self.parameters_label.config(text=parameters_text)
    
    def update_displayed_image(self): # Update the displayed image with the current parameters
        if self.is_green_button:
            self.axis.imshow(self.canaux[0][self.current_index], cmap=self.color_mode)
        if self.is_red_button:
            self.axis.imshow(self.canaux[1][self.current_index], cmap=self.color_mode)

        if(self.is_green_button and self.is_red_button):
            self.axis.imshow(np.dstack((self.canaux[1][self.current_index], self.canaux[0][self.current_index], np.zeros_like(self.canaux[0][self.current_index]))))
      
        dic = self.get_dic_ROI()

        if not self.show_tags_var.get():
            self.remove_all_tags()  
        if self.show_tags_var.get():
            for index in dic.keys(): 
                if not self.tag_artists or index-1 >= len(self.tag_artists):
                    coords=dic.get(index, {}).get('coord') 
                    liste_x=[x for x, y in coords]
                    liste_y=[y for x, y in coords]
                    self.create_tag(index, min(liste_x), min(liste_y))
                self.display_tag(index)

        self.canvas.draw_idle()

    def apply_color_mode(self, image): # Apply the selected color mode to the image
        if self.color_mode == 'gray_r' and self.is_color_changed:
            return 1.0 - image.reshape(self.original_image[0,0].shape)  # Reshape to the original shape
        else:
            return image
        
    def select_green_channel(self): # Select the green channel
        self.selected_channel = 0

        if self.green_button.config('relief')[-1] == 'sunken':
            self.green_button.config(relief=RAISED)
        else:
            self.green_button.config(relief=SUNKEN)

        self.is_green_button = not self.is_green_button
        self.update_image()

    def select_red_channel(self): # Select the red channel
        self.selected_channel = 1

        if self.red_button.config('relief')[-1] == 'sunken':
            self.red_button.config(relief=RAISED)
        else:
            self.red_button.config(relief=SUNKEN)

        self.is_red_button = not self.is_red_button
        self.update_image()

    def change_color(self): # Change the color mode of the image
        self.is_color_changed = not self.is_color_changed
        self.color_mode = 'gray_r' if self.is_color_changed else 'gray'
        self.update_displayed_image()
        self.color_change_button.config(text="Change Color" if not self.is_color_changed else "Revert Color")
  
    def zoom_in(self): # Zoom in the image
        xlim = self.axis.get_xlim()  # Get current x-axis limits
        ylim = self.axis.get_ylim()  # Get current y-axis limits
        zoom_factor = 1.1  # Zoom factor
        new_xlim = (xlim[0] / zoom_factor, xlim[1] / zoom_factor)  # New x-axis limits
        new_ylim = (ylim[0] / zoom_factor, ylim[1] / zoom_factor)  # New y-axis limits
        self.axis.set_xlim(new_xlim)  # Set new x-axis limits
        self.axis.set_ylim(new_ylim)  # Set new y-axis limits
        self.canvas.draw_idle()  # Redraw canvas
        
    def zoom_out(self): # Zoom out the image
        xlim = self.axis.get_xlim()  # Get current x-axis limits
        ylim = self.axis.get_ylim()  # Get current y-axis limits
        zoom_factor = 1.1  # Zoom factor
        new_xlim = (xlim[0] * zoom_factor, xlim[1] * zoom_factor)  # New x-axis limits
        new_ylim = (ylim[0] * zoom_factor, ylim[1] * zoom_factor)  # New y-axis limits
        self.axis.set_xlim(new_xlim)  # Set new x-axis limits
        self.axis.set_ylim(new_ylim)  # Set new y-axis limits
        self.canvas.draw_idle()  # Redraw canvas
        
    def reset_zoom(self): # Reset the zoom level
        self.current_zoom_level = 1.0
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.canvas.draw_idle()
        
    def toggle_zoom_mode(self): # Toggle the zoom mode
        self.zoom_mode = not self.zoom_mode
        self.panning_mode_enabled = False
            
    def on_press(self, event): # Handle the press event for zoom selection 
        self.previous_x = event.x
        self.previous_y = event.y
        

    def on_motion(self, event): # Handle the motion event for zoom selection
        if self.zoom_mode and self.zooming:
            dx = event.x - self.previous_x
            dy = event.y - self.previous_y

            xlim = self.axis.get_xlim()
            ylim = self.axis.get_ylim()
            self.axis.set_xlim(xlim[0] - dx * 0.01, xlim[1] - dx * 0.01)
            self.axis.set_ylim(ylim[0] - dy * 0.01, ylim[1] - dy * 0.01)
        
            self.canvas.draw_idle()
 
        self.previous_x = event.x
        self.previous_y = event.y
    
    def on_release(self, event): # Handle the release event for zoom selection
        self.previous_x = None
        self.previous_y = None
        
    def on_scroll(self, event): # Handle the scroll event for zooming
        if not self.zoom_mode:
            return
        
        zoom_factor = 1.1 if event.button == 'up' else 1 / 1.1
        self.current_zoom_level *= zoom_factor
        xlim = self.axis.get_xlim()
        ylim = self.axis.get_ylim()
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor
        self.axis.set_xlim(center_x - new_width / 2, center_x + new_width / 2)
        self.axis.set_ylim(center_y - new_height / 2, center_y + new_height / 2)
       
        self.canvas.draw_idle()

    def on_pan_press(self, event): # Handle the press event for panning
        if not self.panning_mode_enabled:
            return
        
        self.panning_start_x = event.xdata
        self.panning_start_y = event.ydata

    def on_pan_motion(self, event): # Handle the motion event for panning
        if not self.panning_mode_enabled or self.panning_start_x is None or self.panning_start_y is None:
            return
        current_x = event.xdata
        current_y = event.ydata
        if current_x is not None and current_y is not None:
            delta_x = current_x - self.panning_start_x
            delta_y = current_y - self.panning_start_y
            self.axis.set_xlim(self.axis.get_xlim() - delta_x)
            self.axis.set_ylim(self.axis.get_ylim() - delta_y)
            self.canvas.draw_idle()

    def on_pan_release(self, event): # Handle the release event for panning
        self.panning_start_x = None
        self.panning_start_y = None
        
    def on_zoom_start(self, event): # Handle the start event for zooming
        if self.zoom_mode_enabled:
            self.zooming = True
            self.canvas.config(cursor='hand')

    def on_zoom_end(self, event): # Handle the end event for zooming
        if self.zooming:
            self.zooming = False
            self.canvas.config(cursor='')    

    def toggle_segmentation_mode(self): # Toggle the segmentation mode on/off
        self.segmentation_mode_enabled = not self.segmentation_mode_enabled

        if self.segmentation_mode_enabled:
            # Clear previous segment points
            self.update_color()
            self.segment_x_points = []
            self.segment_y_points = []
            self.temp_ROI_points = []
            self.segmentation_button.config(text="Stop Segmentation")
        else:
            # Process segments and update the displayed image
            self.process_segments()
            self.segmentation_button.config(text="Start Segmentation")
 
    def on_segment_click(self, event): # Handle the click event for segmentation
        # Check if segmentation mode is enabled
        if not self.segmentation_mode_enabled:
            return

        # Add the clicked point to the segment points
        self.segment_x_points.append(event.xdata)
        self.segment_y_points.append(event.ydata)

        # Draw a red dot at the clicked point
        self.temp_ROI_points.append(self.axis.plot(event.xdata, event.ydata, self.marker_styles_point[self.id_color]))
        self.canvas.draw()

    def reset_context_for_segments_csv(self): # Reset the context for segments from CSV [DISABLED]
        return

    def draw_segments_from_csv(self, coordinates): # Draw segments from CSV 
        self.temp_ROI_points = [] 
        
        if (len(coordinates) < 3):
            messagebox.showerror("Error", "Not enough points to form a segment in one of the ROIs imported.")
            return

        self.update_color()
        
        segment_x_points = []
        segment_y_points = []

        for pair in coordinates:
            segment_x_points.append(pair[0])
            segment_y_points.append(pair[1])
            self.temp_ROI_points.append(self.axis.plot(segment_x_points, segment_y_points, self.marker_styles_point[self.id_color]))
        
        temp_segment = self.axis.plot(segment_x_points, segment_y_points, self.marker_styles_line[self.id_color])
        temp_points = self.temp_ROI_points[:]

        self.ROI_objects[self.ROI_index] = { # Add the segment to the ROI objects dictionary
                'seg': temp_segment,
                'points': temp_points
                }
        self.update_ROI_visibility_list()
        self.canvas.draw()
        self.ROI_index += 1

    def process_segments(self): # Process the segments for segmentation
        if len(self.segment_x_points) < 3:
            return  # At least 3 points needed to form a segment

        # Add the first point to the end to create a closed segment
        self.segment_x_points.append(self.segment_x_points[0])
        self.segment_y_points.append(self.segment_y_points[0])

        # Convert segment points to NumPy array
        segment_x_array = np.array(self.segment_x_points)
        segment_y_array = np.array(self.segment_y_points)

        # Draw the closed segment on the displayed image
        temp_segment = self.axis.plot(segment_x_array, segment_y_array, self.marker_styles_line[self.id_color])
        temp_points = self.temp_ROI_points[:]
        self.temp_ROI_points = []

        self.ROI_objects[self.ROI_index] = {
            'seg': temp_segment,
            'points': temp_points
        }
        self.ROI_index += 1
        self.update_ROI_visibility_list()
        self.canvas.draw()

        # Get coordinates of points inside the polygon
        self.list_in()

        # Calculate mean over time for the segment
        mean_values = self.mean_over_time()

        # Create a data structure with relevant information
        segment_data = {
            'segment_x_array': segment_x_array,
            'segment_y_array': segment_y_array,
            'mean_values': 
                        {
                        0: mean_values[0],
                        1: mean_values[1],
                    }
        }

        self.listes_graphs.append((segment_data['mean_values'][0]))
        self.listes_graphs.append((segment_data['mean_values'][1]))

        self.data_viewer.process_segment_data(segment_data)
        self.update_displayed_image()
        dic=self.get_dic_ROI()

    def in_polygon(self, test): # Check if a point is inside the polygon
        x, y = test
        nb_corners = len(self.segment_x_points)
        i, j = nb_corners - 1, nb_corners - 1
        odd_nodes = False
        for i in range(nb_corners):
            if (self.segment_y_points[i] < y and self.segment_y_points[j] >= y) or (self.segment_y_points[j] < y and self.segment_y_points[i] >= y):
                if (self.segment_x_points[i] + (y - self.segment_y_points[i]) / (self.segment_y_points[j] - self.segment_y_points[i]) * (self.segment_x_points[j] - self.segment_x_points[i]) < x):
                    odd_nodes = not odd_nodes
            j = i
        return odd_nodes
    
    def list_in(self): # Get the list of points inside the polygon
        left = int(min(self.segment_x_points))  # Pixels inside the polygon are inside a known square
        right = int(max(self.segment_x_points))
        top = int(max(self.segment_y_points))
        bot = int(min(self.segment_y_points))
        res_x = []
        res_y = []
        for x in range(left, right, 1):
            for y in range(bot, top, 1):
                if self.in_polygon((x, y)):
                    res_x.append(x)
                    res_y.append(y)
        self.x_in=res_x
        self.y_in=res_y

    def mean(self,image): # Calculate the mean of the image
        out = 0
        for i in range(0, len(self.x_in)):
            out += image[self.y_in[i]][self.x_in[i]]
        out = out / len(self.x_in)
        return out
    
    # def mean_over_time(self): # Calculate the mean over time for the region of interest
    #     try:
    #         mean_values = [[],[]]
    #         images = self.original_image

    #         for i in range(len(images[0])):
    #             mean_value_green = 2**16-self.mean(images[0][i])
    #             mean_value_red = 2**16-self.mean(images[1][i])
    #             mean_values[0].append(mean_value_green)
    #             mean_values[1].append(mean_value_red)

    #         return mean_values

    #     except Exception as e:
    #         print("Error calculating mean over time:", e)
    #         return None

    def mean_over_time(self):
        images = self.original_image  # shape (2, T, H, W)
        xs = np.array(self.x_in, dtype=np.int32)
        ys = np.array(self.y_in, dtype=np.int32)
        T = images.shape[1]
        mean_values = [[], []]

        for channel in (0, 1):
            arr = images[channel]
            for i in range(T):
                image_uint16 = arr[i].astype(np.uint16)
                m = 2**16 - fast_mean_pixels(image_uint16, xs, ys)
                mean_values[channel].append(m)
        return mean_values

    def toggle_ROI_visibility(self, index, visibility): # Toggle the visibility of the region of interest
        self.ROI_objects[index]['seg'][0].set_visible(visibility)
        if self.show_tags_var.get():
            self.display_tag(index+1)
        for points_list in self.ROI_objects[index]['points']:
            for point in points_list:
                point.set_visible(visibility)
        self.canvas.draw_idle()
            
    def update_color(self): # Update the color for the region of interest
        if self.id_color<7:
            self.id_color+=1
        else : 
            self.id_color=0

    def update_ROI_visibility_list(self): # Update the region of interest visibility list
        self.GUI.update_ROI_visibility_list()
    
    def create_tag(self, id, x, y): # Create a tag for the region of interest
        tag_artist = self.axis.text(x, y, f"ROI {id}", color='white', fontsize=10, ha='center', va='center')
        self.tag_artists.append(tag_artist)
    
    def display_tag(self, id): # Display the tag for the region of interest
        if not self.ROI_objects[id-1]['seg'][0].get_visible():
            self.tag_artists[id-1].set_visible(False)
        else:
            self.tag_artists[id-1].set_visible(True)
 
    def remove_all_tags(self): # Remove all tags from the display
        for tag_artist in self.tag_artists:
            tag_artist.set_visible(False)

    def get_dic_ROI(self): # Get the dictionary of region of interest
        return self.GUI.get_dic_ROI()