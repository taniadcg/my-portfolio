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

## FAQ

**Q1: What is the purpose of the 3D Racing Track Scenario?**  
**A1:** The purpose of the 3D Racing Track Scenario is to demonstrate 3D modeling and augmented reality using JavaScript, WebGL, and HTML, allowing users to interact with a virtual racing environment.

**Q2: What objects are included in the scenario?**  
**A2:** The scenario includes various objects such as two cars, four benches, trees, and a racing track created using two cylinders.

**Q3: How can I manipulate the cars in the scenario?**  
**A3:** The movement of the cars can be controlled using the following keys:
- **w** - Move forward
- **s** - Move backward
- **a** - Rotate left
- **d** - Rotate right

**Q4: How do I switch to the car's viewpoint?**  
**A4:** To switch to the car's viewpoint, press the **c** key. This will follow the same transformations applied to the car.

**Q5: What keys are used to change the view and lighting?**  
**A5:** The following keys are used to change the viewpoint, toggle between day and night, and control lights:
- **p** - Top view (increases Y)
- **l** - Bottom view (decreases Y)
- **n** - Toggle day/night
- **1** - Activate Light 1 (white + RGB)
- **2** - Activate Light 2 (white + RGB)

**Q6: How is the viewpoint controlled?**  
**A6:** The viewpoint is controlled by adjusting several variables, including `rotateAngle`, `positionX`, `positionY`, and `up`. These adjustments are made based on user input to provide an immersive experience.

**Q7: What happens when I press the "c" key?**  
**A7:** Pressing the "c" key allows you to switch to the car's viewpoint, following the transformations applied to the car, providing a first-person perspective of the racing environment.

**Q8: What programming languages and technologies are used in this project?**  
**A8:** This project is developed using JavaScript for scripting, WebGL for rendering graphics, and HTML for structuring the web application.

**Q9: How do I run the Jupyter Notebook associated with this project?**  
**A9:** To run the Jupyter Notebook, follow these steps:
1. Navigate to the folder where the Jupyter Notebook file is located.
2. Launch Jupyter Notebook by running `jupyter notebook` in your terminal.
3. Open the notebook file and run the cells in order by pressing `Shift + Enter`.

**Q10: Who should I contact for further information or support?**  
**A10:** For further information or support, you can contact the project maintainer at gtaniadc@gmail.com.
