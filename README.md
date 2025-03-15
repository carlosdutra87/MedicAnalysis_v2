# MedicAnalysis v1.0
# École des Mines de Saint-Étienne

Produced by: Aya El Janoussi, Carlos Henrique Dutra Filho, Marc Gleyze and Tom Bourbon
2024

---

## Instructions

MedicAnalysis is an medical imaging analysis tool developed to assist researchers in the École des Mines de Saint-Étienne Bio-Medical department.

### Loading an Image / Creating a project

1. Click on the **Load Image** button to load an image.
2. Select the desired TIFF image file (.tiff or .tif) from your computer.

- This process will create a project folder with the inserted name in your computer (C:\MedicAnalysis\Projects\-yourprojectname-).
- The reference to the .tiff file in your project will be guarded from your first selection. If the location is changed, please edit the identification.txt file inside the project folder.

### Loading a project

1. Click on the **Load project** button to load a project.
2. Select the desired folder from your computer.

- The reference to the .tiff file in your project will be guarded from your first selection. If the location is changed, please edit the identification.txt file inside the project folder.

### User guide for the GUI

#### Once the software is open, you'll find a few different sections:

- **Toolbar**: Located at the top, it contains the traditional software menus.
- **File explorer**: Located on the left, it contains the file tree of the software folder created on the user's computer and allows you to browse through the files.
- **Image display**: Located in the center of the screen, it contains the display of the selected .tiff file, allowing the user to edit brightness, contrast, temporal mediation, select the frame using the image slider, use the mouse function as a hand to move the image and zoom-in and zoom-out, as well as the segmentation function which will be dealt with separately.
- **Data display**: Located on the top right-hand side of the screen, this is where you can browse the segmentation data, going through the pages covering all the frames of the loaded image, with the possibility of importing/exporting data from .csv files for analysis in the software.
- **Graph display**: Located on the lower right-hand side of the screen, this is where the graphs plotted with the data obtained from the user's segmentation are displayed. Using the same selection tool as for the Data display, you can select to display the graph and data for the ROI you want to analyze at any given time, as well as the graph comparison mechanism, where you can plot multiple ROIs simultaneously.
- **Segmentation**: The "Start segmentation" button starts segmentation, so future clicks are used to circle a ROI. At the end of the selection of points surrounding the ROI, the user can stop the segmentation with the dedicated "Stop segmentation" button. Red and green channel graphs are displayed directly in the graph viewer window. To view the graphs of another ROI, use the drop-down menu in the data viewer window.

### .exe WARNING: If you have the .exe file for the MedicAnalysis software, *run it in a folder with the images included in this repo (zoom-in.png; zoom-out.png; segmentation.png)*.


