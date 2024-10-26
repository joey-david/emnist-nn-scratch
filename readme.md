# Neural Network from Scratch for Character Recognition

This project implements a fully-connected neural network from scratch in Python. The network is trained on the MNIST dataset (and optionally EMNIST or other datasets) to recognize handwritten characters. Using only NumPy for numerical computations, this code demonstrates forward propagation, backpropagation, and weight updates to train a network without relying on external deep learning libraries like TensorFlow or PyTorch.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Architecture and Training](#training-and-evaluation)
4. [Acknowledgments](#acknowledgments)

---

## Features

- Implements forward and backpropagation for a 4-layer neural network (input layer, hidden layers, and output layer).
- Uses Cross Entropy Loss loss with softmax activation.
- Includes temperature scaling for the softmax layer.
- Training and testing routines with accuracy monitoring.

## Project Structure

```
├── data/                        # dataset construction
├── build_data.py                # Main neural network implementation
├── network.py                   # Main neural network implementation
```

## Architecture and Training

### Model Architecture

The neural network architecture is a simple feedforward network with the following layers:

- Input Layer: Receives 784-dimensional input vectors (28x28 pixel images).
- Hidden Layers: Two hidden layers with 80 neurons each, using ReLU activation.
- Output Layer: A softmax layer with 51 output neurons, representing the probability distribution over 51 different characters.

### Training Process

The network is trained using mini-batch gradient descent with backpropagation. The training process involves:

- Forward Propagation: Input data is fed through the network, calculating activations at each layer.
- Loss Calculation: The cross-entropy loss is calculated between the predicted output and the true labels.
- Backpropagation: Gradients of the loss with respect to the weights and biases are computed.
- Weight Update: Weights and biases are updated using gradient descent.

## Acknowledgments

DCohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from http://arxiv.org/abs/1702.05373
https://github.com/wblachowski/bhmsds