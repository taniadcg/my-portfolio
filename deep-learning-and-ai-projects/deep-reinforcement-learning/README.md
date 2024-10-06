# Deep Reinforcement Learning: CartPole and CarRacing Environments

## Table of Contents
1. [Description](#description)
2. [Program and Version](#program-and-version)
3. [Implementation](#implementation)
4. [Usage](#usage)
5. [Author](#author)
6. [Contact](#contact)
7. [FAQ](#faq)

## Description
This project demonstrates the application of Deep Reinforcement Learning (DeepRL) techniques to control agents in two simulation environments: **CartPole-v0** and **CarRacing-v0**. The project includes a detailed analysis of different reinforcement learning algorithms, including DQN, Double DQN, and Double Dueling DQN, utilizing both affordance-based and image-based state representations.

**Note:** Some folders and features in this project are still under development and may not be fully implemented. Please check back later for updates, or feel free to explore the current features that are available.

## Program and Version

## Implementation

### Simulation Environments 
The following OpenAI Gym environments are used in this project:

- **CartPole-v0:** 
  - In this environment, a pole is attached to a cart that moves along a frictionless track. The agent's goal is to balance the pole by controlling the cart's velocity.
  - The state representation includes:
    - Cart position
    - Cart velocity
    - Pole angle
    - Pole angular velocity
  - Actions available: Move cart left or right.
  - The agent receives a reward of 1 for every time step until termination conditions are met.

- **CarRacing-v0:** 
  - This is a more complex environment where the agent controls a car to race through a track.
  - The state representation consists of a 96x96 RGB image.
  - Actions available:
    - Steering (range: [-1, 1])
    - Gas (range: [0, 1])
    - Brake (range: [0, 1])
  - The reward model includes:
    - -0.1 for every frame
    - +1000/N for every new track tile visited (N: total tiles visited)
    - -100 if the car goes off the track

### Affordance-based State Representation (CartPole-v0)
- Implemented a Deep Q-Network (DQN) to model the CartPole environment.
- Techniques used include:
  - Epsilon-greedy strategy for exploration.
  - Comparison of DQN, Double DQN, and Double Dueling DQN algorithms.
  - Implementation of a priority replay buffer.
  - Hyperparameter tuning and architecture adjustments, including adding dropout layers and changing activation functions.
- Achieved a total reward of over 195 in the last 20 episodes.

### Image-based State Representation (CarRacing-v0)
- Developed a Convolutional Neural Network (CNN) to process RGB images from the CarRacing environment.
- Techniques utilized:
  - Epsilon-greedy exploration.
  - Implementation of DQN variants (Double DQN, Double Dueling DQN).
  - Hyperparameter tuning and modifications to the action set and reward model.
  - Adapted a pretrained model (VGG16) without freezing its weights.
- Successfully obtained a total reward of over 200 in the last 20 episodes.

## Requirements
To run this project, ensure you have the following libraries installed:

```bash
pip install gym
pip install keras
pip install tensorflow
pip install matplotlib
pip install opencv-python
```
## Usage

1. **Clone the repository:**
   Open your terminal or command prompt and run the following command to clone the project to your local machine:
   ```bash
   git clone https://github.com/yourusername/deep-reinforcement-learning.git
   cd deep-reinforcement-learning
   ```
2. **Install dependencies**:
   ```bash
   pip install gym keras tensorflow matplotlib opencv-python
   ```
3. **Run the notebooks**: Launch Jupyter Notebook and open the provided .ipynb files to train the agents in CartPole and CarRacing environments.

4. **Visualize performance**: Use the code within the notebooks to generate performance plots and analyze results.

## Author
Tânia Gonçalves

## Contact
For further information or support, contact [gtaniadc@gmail.com](mailto:gtaniadc@gmail.com).

## FAQ

**Q1: What is the goal of this project?**  
**A1:** The goal is to learn and apply Deep Reinforcement Learning (DeepRL) techniques to control agents in simulation environments, specifically using the OpenAI Gym's CartPole-v0 and CarRacing-v0 environments.

**Q2: What environments are explored?**  
**A2:** The project explores the CartPole-v0 and CarRacing-v0 environments from OpenAI Gym. 

**Q3: What are the key tasks in the project?**  
**A3:** The tasks involve developing a Deep Neural Network for the CartPole-v0 environment and a Convolutional Neural Network for the CarRacing-v0 environment, using various reinforcement learning algorithms.

**Q4: Which reinforcement learning techniques are implemented?**  
**A4:** Techniques such as DQN, Double DQN, Double Dueling DQN, and epsilon-greedy exploration are implemented, along with potential modifications to the neural network architecture.

**Q5: What are the performance evaluation metrics used?**  
**A5:** Performance is evaluated based on the total reward obtained per episode and the number of episodes required to meet the goal conditions.

**Q6: How do I set up the project?**  
**A6:** Ensure that you have the required libraries installed and that you can run the Jupyter notebooks provided in the repository.

**Q7: What should I do if I encounter errors while running the code?**  
**A7:** Check for installation issues, verify that the code paths are correct, and review error messages for troubleshooting.

**Q8: Who should I contact for further information or support?**  
**A8:** For further information or support, you can contact me at [gtaniadc@gmail.com](mailto:gtaniadc@gmail.com).
