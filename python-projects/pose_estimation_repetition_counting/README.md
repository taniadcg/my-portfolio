# Pose Estimation and Repetition Counting

## Table of Contents
1. [Description](#description)
2. [Program and Version](#program-and-version)
3. [Features](#features)
4. [Usage](#usage)
5. [Author](#author)
6. [Contact](#contact)
7. [FAQ](#faq) 

## Description 
This project performs pose estimation and repetition counting using the Mediapipe library. It analyzes video input to track poses and count repetitions of exercises.

## Program and Version
- **Programming Language**: Python
- **Version**: 3.9.7
- **Development Environment**: Spyder (Anaconda 3)

## Features
- **Pose Estimation**: Utilizes Mediapipe's BlazePose for real-time pose tracking.
- **Repetition Counting**: Counts repetitions of specific exercises based on pose classification.
- **Create Plot**: Generates a plot showing pose classifications over time.
- **Create Video**: Outputs a video with visualized pose estimation and repetition counting.
- **Create CSV File**: Saves frames when pose changes occur to a CSV file.

## Usage
To run the project, use the following command:  
```bash
python pose_estimation_repetition_counting.py
```
**Note**: Make sure to adapt the paths to access the video/images of the exercise you want to analyze for pose estimation and repetition counting.

## Author
Tânia Gonçalves 

## Contact
For further information or support, contact [gtaniadc@gmail.com](mailto:gtaniadc@gmail.com).

## FAQ

**Q1: What is the purpose of this program?**  
**A1:** The program performs pose estimation and counts repetitions of exercises by analyzing video input using the Mediapipe library.

**Q2: What libraries does this project use?**  
**A2:** The project uses Mediapipe for pose estimation and OpenCV for video processing.

**Q3: How do I adapt the input paths?**  
**A3:** Modify the `input_path` and `pose_samples_folder` variables in the code to point to the location of your video files and pose class CSVs.

**Q4: What output files does the program generate?**  
**A4:** The program generates a video file showing pose estimation and repetition counting, a plot image of classifications, and a CSV file logging frames where pose changes occur.

**Q5: How does the repetition counting work?**  
**A5:** Repetition counting is done by tracking specific poses over time and determining when the user completes a full repetition based on defined thresholds.

**Q6: Can I modify the exercises analyzed by the program?**  
**A6:** Yes, you can change the `class_name` variable in the code to analyze different exercises. Ensure you have the corresponding pose data available.

**Q7: How can I visualize the results?**  
**A7:** The program creates a video output that visually indicates the detected poses and counts, along with a plot of pose classifications.

**Q8: What should I do if I encounter errors while running the code?**  
**A8:** Check that all paths are correctly set and that you have all required libraries installed. If issues persist, refer to the error messages for troubleshooting.

**Q9: Who should I contact for further information or support?**  
**A9:** For further information or support, you can contact the project maintainer at [gtaniadc@gmail.com](mailto:gtaniadc@gmail.com).
