# CNN for CIFAR-10 Image Classification

## Table of Contents
1. [Description](#description)
2. [Program and Version](#program-and-version)
3. [Features](#features)
4. [Usage](#usage)
5. [Author](#author)
6. [Contact](#contact)
7. [FAQ](#faq)

## Description
This project implements a Convolutional Neural Network (CNN) from scratch to classify images from the CIFAR-10 dataset. The model is trained to recognize 10 distinct classes of objects, including airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Various techniques like dropout, batch normalization, and data augmentation are used to improve model accuracy.

## Program and Version
- **Programming Language**: Python
- **Development Environment**: Google Colab / Jupyter Notebook

## Features
- **Custom CNN Architecture**: A deep learning model built from scratch for CIFAR-10 classification.
- **Model Evaluation**: Utilizes metrics such as mean-Class Accuracy (mCA), F1-Score, and Confusion Matrix to assess the model's performance.
- **Data Augmentation**: Enhances training with techniques like flipping, cropping, and normalization.
- **Dropout and Regularization**: Prevents overfitting by implementing dropout layers and batch normalization.
- **Model Saving**: Saves the trained CNN model for later use.

## Usage
To run the project, follow these steps:

1. **Clone the repository** and navigate to the project folder:
   ```bash
   git clone https://github.com/yourusername/cnn-cifar10-project.git
   cd cnn-cifar10-project
   ```
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the CIFAR-10 dataset** and run the notebook:
- Open the cnn_cifar10_classification.ipynb notebook in Google Colab or Jupyter Notebook.
- Follow the notebook's instructions to train the CNN and evaluate the results.

4. **Train the model**: Run the notebook cells to download the dataset, build the CNN model, and train it using CIFAR-10.

5. **Model Evaluation**: After training, the notebook will output metrics such as accuracy, F1-Score, and display the confusion matrix.

## Author
Tânia Gonçalves

## Contact
For further information or support, contact [gtaniadc@gmail.com](mailto:gtaniadc@gmail.com).

## FAQ

**Q1: What is the purpose of this project?**  
**A1:** The project aims to build a custom Convolutional Neural Network (CNN) from scratch to classify images from the CIFAR-10 dataset.

**Q2: What libraries does this project use?**  
**A2:** The project primarily uses PyTorch for building the CNN, torchvision for dataset handling, and matplotlib for plotting results.

**Q3: How do I adjust hyperparameters?**  
**A3:** You can modify hyperparameters such as learning rate, batch size, and number of epochs directly in the notebook, in the model training section.

**Q4: How do I save my trained model?**  
**A4:** The notebook includes code to save the trained model as a `.pth` file, which can be used for future predictions.

**Q5: Can I modify the CNN architecture?**  
**A5:** Yes, you can modify the number of layers, add/remove convolutional layers, change activation functions, or introduce new regularization methods like dropout or batch normalization in the notebook.

**Q6: What is the dataset used in this project?**  
**A6:** The project uses the CIFAR-10 dataset, a collection of 60,000 32x32 color images categorized into 10 different classes.

**Q7: How do I visualize the model's performance?**  
**A7:** The notebook includes visualizations such as loss and accuracy curves, as well as a confusion matrix to display the model's classification performance.

**Q8: How can I improve the model's accuracy?**  
**A8:** You can try modifying the architecture (e.g., adding layers, using different activation functions), adjusting hyperparameters, or using advanced techniques like learning rate scheduling or early stopping.

**Q9: Who should I contact for further information or support?**  
**A9:** For any questions or issues, feel free to reach out to the project maintainer at [gtaniadc@gmail.com](mailto:gtaniadc@gmail.com).
