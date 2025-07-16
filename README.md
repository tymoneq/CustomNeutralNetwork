### About The Project
This repository contains a custom-built, lightweight neural network library implemented from scratch in C++. The primary goal of this project is to provide a deep understanding of the fundamental concepts behind neural networks, including forward propagation, backpropagation, and gradient descent, without relying on high-level machine learning frameworks.

### Features
Feedforward Neural Network (Multi-Layer Perceptron): Implementation of a basic neural network architecture.

Customizable Layers: Supports defining networks with arbitrary numbers of hidden layers and neurons per layer.

Activation Functions: Includes common activation functions such as:

Sigmoid
Tanh (Hyperbolic Tangent)

Loss Function: Utilizes Cross Entropy Loss function for classification problems

Backpropagation Algorithm: Implements the core backpropagation algorithm for efficient gradient calculation.

Gradient Descent Optimization: Uses Stochastic Gradient Descent (SGD) for weight and bias updates.


### Getting Started
To get a local copy up and running, follow these simple steps.

## Prerequisites
C++ Compiler: A C++11 compatible compiler (or newer, e.g., C++17 recommended).

g++ (GNU C++ Compiler)

Clang

MSVC (Microsoft Visual C++)

CMake: Version [e.g., 3.10+] or higher.

## Installation
Clone the repo:

```Bash

git clone https://github.com/tymoneq/CustomNeutralNetwork.git
cd CustomNeutralNetwork
```
Build the project using CMake:

```Bash

mkdir build
cd build
cmake ..
cmake --build .
```
This will compile the executable (myapp) in the build directory.

Usage
After building, you can run the main example to see the neural network in action.

```Bash
./build/myapp
```
The example will classify types of wheat that are in the seeds_dataset.txt



To use the NeuralNetwork class in your own C++ applications:

Include the necessary header files (e.g., NeuralNetwork.h).

Create an instance of the NeuralNetwork class, specifying layer sizes and a learning rate.

Prepare your input and target data using Eigen VectorXd.

Call the train() method for each training epoch.

Use the predict() method for predictions.

### Contributing
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
Distributed under the MIT.
