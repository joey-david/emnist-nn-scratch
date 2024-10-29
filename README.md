# Neural Network from Scratch for Character Recognition

A pure NumPy implementation of a neural network for character recognition, trained on EMNIST and BHMSDS datasets. This project demonstrates a neural network implementation from scratch, without frameworks like PyTorch and Tensorflow. The neural network focuses on character recognition using a fully-connected architecture; it processes images and classifies them into 51 different character classes, including numbers, letters, and mathematical symbols.

## Table of Contents
1. [Features](#features)
2. [Try it Yourself](#try-it-yourself)
2. [Project Structure](#project-structure)
3. [Network Architecture](#network-architecture)
4. [Implementation Details](#implementation-details)
5. [Training Process](#training-process)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Current Limitations](#current-limitations)
8. [Acknowledgments](#acknowledgments)

## Features
- Pure NumPy implementation of forward and backpropagation
- ReLU activation for hidden layers
- Softmax activation with cross-entropy loss
- Early stopping mechanism to prevent overfitting
- Mini-batch gradient descent
- Automated hyperparameter tuning suite
- Training progress visualization
- Model checkpoint saving
## Try it Yourself

After cloning the repository, you can experiment with the model by following these steps:

- **Run a complete hyperparameter benchmark:**
  ```bash
  python3 nn_tweaking.py
  ```

- **Train the model with custom parameters:**
  ```bash
  python3 neural_network.py
  ```

- **Access saved weights and biases:**
  - Check the `models/` directory for saved model parameters.

## Project Structure
```
├── data/
│   ├── bhmsds_processed_symbols/    # Processed BHMSDS dataset
│   ├── bhmsds_symbols/              # Raw BHMSDS dataset
│   ├── emnist/                      # EMNIST dataset
│   ├── processed_dataset.csv        # Combined processed dataset
├── models/                          # Saved model weights and biases
├── dataset.py                       # Dataset processing utilities
├── neural_network.py                # Core neural network implementation
├── nn_tweaking.py                   # Hyperparameter optimization tools
├── build_data.py                    # Dataset construction script
├── hyperparameter_analysis.png      # Dataset construction script
└── README.md                        # This file
```

## Network Architecture
- Input Layer: 784 neurons (28x28 flattened images)
- Hidden Layer 1: Configurable neurons (default: 96) with ReLU
- Hidden Layer 2: Configurable neurons (default: 64) with ReLU
- Output Layer: 51 neurons with softmax activation

Network initialization uses He initialization for weights:
- W0: $$\text{randn}(n_{l1}, 784) * \sqrt{2/784}$$
- W1: $$\text{randn}(n_{l2}, n_{l1}) * \sqrt{2/n_{l1}}$$
- W2: $$\text{randn}(51, n_{l2}) * \sqrt{2/n_{l2}}$$

## Implementation Details
Key components:
- Forward Propagation:
  - Layer computations: Z = WX + b
  - ReLU activation: max(0, x)
  - Softmax with numerical stability: shifted_logits approach
- Backpropagation:
  - Cross-entropy loss gradient computation
  - Chain rule application through layers
  - Gradient descent parameter updates
- Mini-batch Processing:
  - Configurable batch size (default: 48)
  - Normalized input (0-1 range)
  - One-hot encoded targets

## Training Process
Default hyperparameters:
```python
SPLIT = 0.9
LEARNING_RATE = 0.002
BATCH_SIZE = 32
EPOCH_NUMS = 32
NUM_CLASSES  = 51
NEURONS_L1 = 96
NEURONS_L2 = 64
```

The training loop includes:
- Mini-batch gradient descent
- Early stopping (triggers if accuracy plateaus for 5 epochs)
- Live progress monitoring
- Cost and accuracy tracking
- Training time measurement
- Sample prediction display

## Hyperparameter Tuning
The nn_tweaking.py module provides:
- Automated testing of hyperparameter combinations
- Training progress visualization
- Performance metric tracking:
  - Loss over time
  - Training accuracy
  - Test accuracy
  - Training time per epoch
- Model checkpoint saving for best configurations

## Hyperparameter Analysis
<a href="hyperparameter_analysis_20241029_0136.png" target="_blank">
  <img src="hyperparameter_analysis_20241029_0136.png" alt="Hyperparameter Analysis" width="600">
</a>

## Current Limitations
1. Character Confusion Issues:
   - I vs l
   - Forward/backward slashes vs I
   - 1 vs I
   - 0 vs O

    Could be improved by adding dimensional features (height/width ratios)

2. Potential Improvements

   - Implement dropout to prevent overfitting
   - Introduce weight inertia (momentum) to accelerate gradient descent
   - Experiment with different activation functions (e.g., Leaky ReLU, ELU)
   - Improve character confusion handling with advanced feature extraction

3. Architectural Limitations:
   - Fixed network depth
   - Limited to supervised learning
   - No data augmentation

## Acknowledgments
- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
- BHMSDS Dataset: https://github.com/wblachowski/bhmsds