# 3D Racing Track Scenario
3D Racing Track Scenario is a 3D Modeling and Augmented Reality with JavaScript, WebGL and HTML Project.

## Table of Contents
1. [Description](#description)
2. [Program and Version](#program-and-version)
3. [Features](#features)
4. [Usage](#usage)
5. [Author](#author)
6. [Contact](#contact)
7. [FAQ](#faq)

## Description
3D Racing Track Scenario is a 3D Modeling and Augmented Reality with JavaScript, WebGL and HTML Project.

## Program and Version
- **Programming Language**: Python
- **Development Environment**: Jupyter Notebook

## Features

In this scenario, some objects were added, such as two cars, four benches, some trees, and other objects. The track was created using two cylinders.

For the movement of the cars, a transformation was applied, consisting of a translation to the origin + rotation + translation from the origin. The rotation is around the Y axis, so when performing the rotation and translating from the origin to the next position, these actions are based on the angles **angRotation1** and **angRotation2** for cars 1 and 2, respectively.

When the "c" key is pressed to switch to the car's viewpoint, it "follows" the same transformation applied to the car.

To manipulate the viewpoint, day/night scenario, and lights, the following keys were used:

- **a** - Rotate left
- **d** - Rotate right
- **s** - Move backward
- **w** - Move forward
- **p** - Top view (increases Y)
- **l** - Bottom view (decreases Y)
- **c** - Key to switch viewpoint
- **n** - Key to change day/night
- **1** - Light 1 (white + RGB)
- **2** - Light 2 (white + RGB)

Regarding the viewpoint, it is manipulated through several variables:

- When rotating left or right, the angle **rotateAngle** is increased or decreased, respectively.
- When moving forward or backward, **positionX** and **positionY** are increased or decreased, respectively. These variables are calculated using the formulas:

    ```plaintext
    positionX +/-= 0.5 * cos(rotateAngle);
    positionZ +/-= 0.5 * sin(rotateAngle);
    ```

To achieve a higher or lower view, the variable **up** is increased or decreased, respectively.

The previous variables allow defining the **cameraPosition** and **cameraTarget** for the viewpoint as follows:

```plaintext
cameraPosition = (positionX, 0.2 + up, positionZ);
cameraTarget = (positionX + cos(rotateAngle), 0.2 + up, positionZ + sin(rotateAngle));
```

## Usage
How to run the project:  

## Usage

To use this Jupyter Notebook, follow these steps:

1. **Open the Jupyter Notebook**:
   - Navigate to the folder where the Jupyter Notebook (`.ipynb`) file is located.
   - Launch Jupyter Notebook by running the following command in your terminal:
     ```bash
     jupyter notebook
     ```
   - Click on the notebook file to open it in your web browser.

2. **Run Cells in Order**:
   - Start with the first cell, which usually contains necessary imports or initializations.
   - To execute a cell, click on it and press `Shift + Enter`. This will run the selected cell and move to the next cell.
   - Alternatively, you can click on the **Run** button in the toolbar at the top of the notebook.
   - Repeat this process to run each cell in sequence. Ensure that you run cells in the order they appear, as some cells may depend on variables or functions defined in previous cells.

3. **Running All Cells**:
   - If you want to run all cells at once, go to the menu bar and select:
     ```
     Cell > Run All
     ```
   - This will execute all cells in the notebook from top to bottom.

4. **Modify Inputs as Needed**:
   - If the notebook contains input cells for parameters or configurations, feel free to modify them as needed before running the subsequent cells.

5. **Save Your Work**:
   - To save your progress, click on the **Save** icon or press `Ctrl + S`.

### Note
Running cells in order is crucial to ensure that all dependencies are met and that the notebook functions as intended. If you encounter any errors, make sure to check that all previous cells were executed correctly.


## Author:
Tânia Gonçalves

## Contact:
For further information or support, contact gtaniadc@gmail.com.

## FAQ
**Q1: What is the purpose of this program?**  
**A1:** The program synchronizes a source folder with a replica folder, ensuring the replica is an exact copy of the source.

**Q2: How often does the synchronization occur?**  
**A2:** The synchronization occurs at intervals specified by the 'syncInterval' parameter, which must be entered in seconds.

**Q3: What happens if a file is deleted from the source folder?**  
**A3:** If a file is deleted from the source folder, it will also be deleted from the replica folder during the next synchronization.

**Q4: Are subfolders also synchronized?**  
**A4:** Yes, all subfolders and their contents are synchronized along with the main folder.

**Q5: What if a file in the replica folder is modified?**  
**A5:** If a file in the replica folder is modified, it will be overwritten by the corresponding file from the source folder during the next synchronization.

**Q6: What programming language and environment are used for this program?**  
**A6:** The program is written in Python 3.9.7 and developed using Spyder in the Anaconda 3 environment.

**Q7: Can I configure the synchronization process?**  
**A7:** Yes, you can configure the synchronization process by setting the appropriate parameters in the program, such as the source folder, replica folder, and syncInterval.

**Q8: How do I run the program?**  
**A8:** You can run the program by executing the following command:  
python sync_folders_task.py /path/to/source /path/to/replica syncInterval /path/to/logfile.log  
Make sure to specify the 'syncInterval' in seconds.

**Q9: Who should I contact for further information or support?**  
**A9:** For further information or support, you can contact the project maintainer at gtaniadc@gmail.com.
