# Pima Diabetes Prediction with Custom Deep Learning Framework

A deep learning project implementing a custom PyTorch-like framework (MyTorch) from scratch and using it to predict diabetes using the Pima Indians Diabetes Dataset.

## Project Overview

This project consists of two main components:

1. **MyTorch Library**: A custom implementation of a deep learning framework with automatic differentiation, built from scratch using NumPy
2. **Diabetes Prediction**: Multi-Layer Perceptron (MLP) trained on the Pima Indians Diabetes Dataset to predict diabetes occurrence

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [MyTorch Framework](#mytorch-framework)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Phases](#project-phases)
- [Results](#results)
- [Technical Details](#technical-details)

## Features

### MyTorch Framework Components

**Core Components:**
- Custom Tensor class with automatic differentiation
- Computational graph construction for backpropagation
- Model abstraction layer

**Neural Network Layers:**
- Linear (Fully Connected)
- Conv2D (2D Convolutional)
- MaxPool2D
- AvgPool2D

**Activation Functions:**
- ReLU (Rectified Linear Unit)
- Sigmoid
- Tanh (Hyperbolic Tangent)
- LeakyReLU
- Softmax
- Step Function

**Loss Functions:**
- Mean Squared Error (MSE)
- Cross Entropy (CE)

**Optimizers:**
- Stochastic Gradient Descent (SGD)
- Momentum
- Adam
- RMSprop

**Utilities:**
- DataLoader for batch processing
- Weight Initializers (Xavier, He, etc.)
- Flatten operation

## Project Structure

```
CI Project 1/
├── mytorch/                    # Custom deep learning framework
│   ├── __init__.py
│   ├── tensor.py              # Tensor implementation with autograd
│   ├── model.py               # Base model class
│   ├── activation/            # Activation functions
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   ├── tanh.py
│   │   ├── leaky_relu.py
│   │   ├── softmax.py
│   │   └── step.py
│   ├── layer/                 # Neural network layers
│   │   ├── linear.py
│   │   ├── conv2d.py
│   │   ├── max_pool2d.py
│   │   └── avg_pool2d.py
│   ├── loss/                  # Loss functions
│   │   ├── mse.py
│   │   └── ce.py
│   ├── optimizer/             # Optimization algorithms
│   │   ├── sgd.py
│   │   ├── momentum.py
│   │   ├── adam.py
│   │   └── rmsprop.py
│   └── util/                  # Utility functions
│       ├── data_loader.py
│       ├── initializer.py
│       └── flatten.py
├── phase1.ipynb               # Data preprocessing
├── phase2.ipynb               # Model training with MyTorch
├── phase3.ipynb               # Activation function comparison
├── diabetes.csv               # Original dataset
├── diabetes_preprocessed.csv  # Preprocessed dataset
└── best_model.pth            # Saved model weights
```

## MyTorch Framework

MyTorch is a lightweight deep learning framework built from scratch using NumPy. It implements:

### Automatic Differentiation
The Tensor class tracks computational dependencies and automatically computes gradients through backpropagation:

```python
from mytorch import Tensor

x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2 + 1
y.backward()
print(x.grad)  # Gradients computed automatically
```

### Model Building
Define custom models by inheriting from the Model base class:

```python
from mytorch.model import Model
from mytorch.layer import Linear
from mytorch.activation import relu, sigmoid

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(8, 64)
        self.fc2 = Linear(64, 32)
        self.fc3 = Linear(32, 1)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = sigmoid(self.fc3(x))
        return x
```

## Dataset

**Pima Indians Diabetes Dataset**

- Source: UCI Machine Learning Repository
- Samples: 768 instances
- Features: 8 clinical measurements
  - Pregnancies
  - Glucose concentration
  - Blood pressure
  - Skin thickness
  - Insulin level
  - Body Mass Index (BMI)
  - Diabetes pedigree function
  - Age
- Target: Binary classification (0: No diabetes, 1: Diabetes)

### Data Preprocessing

1. **Missing Value Imputation**: Zero values in certain features (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) are replaced with mean values stratified by class
2. **Feature Scaling**: StandardScaler applied to normalize all features
3. **Train-Validation-Test Split**: 60-20-20 split with stratification

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd "CI Project 1"

# Install required packages
pip install numpy pandas matplotlib scikit-learn

# For phase 1 (optional, uses PyTorch for comparison)
pip install torch
```

## Usage

### Running the Project

The project is organized into three phases as Jupyter notebooks:

```bash
# Phase 1: Data Preprocessing
jupyter notebook phase1.ipynb

# Phase 2: Training MLP with MyTorch
jupyter notebook phase2.ipynb

# Phase 3: Activation Function Comparison
jupyter notebook phase3.ipynb
```

### Using MyTorch Framework

```python
from mytorch import Tensor
from mytorch.model import Model
from mytorch.layer import Linear
from mytorch.activation import relu, sigmoid
from mytorch.loss import MeanSquaredError
from mytorch.optimizer import SGD

# Create model
model = DiabetesMLP()

# Create optimizer
optimizer = SGD(model.parameters(), learning_rate=0.05)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_train)
    
    # Compute loss
    loss = loss_fn(predictions, y_train)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
```

## Project Phases

### Phase 1: Data Preprocessing

- Load Pima Indians Diabetes Dataset
- Handle missing values (zeros replaced with class-wise means)
- Feature scaling using StandardScaler
- Create train-validation-test splits
- Save preprocessed data

### Phase 2: MLP Training with MyTorch

- Build 3-layer MLP architecture (8 -> 64 -> 32 -> 1)
- Implement training loop using custom MyTorch library
- Track training and validation metrics
- Achieve target accuracy: 70% train, 60% test
- Save best performing model

**Architecture:**
```
Input Layer (8 features)
    ↓
Hidden Layer 1 (64 neurons, ReLU)
    ↓
Hidden Layer 2 (32 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
```

### Phase 3: Activation Function Comparison

Systematic comparison of four activation functions:
- Sigmoid
- Tanh
- ReLU
- LeakyReLU

**Fixed Parameters:**
- Architecture: Same 3-layer MLP
- Optimizer: SGD with learning rate 0.05
- Epochs: 50
- Batch size: 64
- Random seed: 42

**Evaluation Metrics:**
- Training and validation accuracy
- Precision, Recall, F1-score
- Confusion matrix
- ROC curve and AUC

**Visualization:**
- Training loss curves
- Accuracy comparison
- ROC curves
- Performance metrics table

## Results

### Model Performance

**Phase 2 Results:**
- Training Accuracy: >70%
- Validation Accuracy: >60%
- Test Accuracy: >60%

### Activation Function Comparison

The project includes comprehensive analysis comparing different activation functions with visualizations showing:
- Convergence speed
- Final accuracy achieved
- Overfitting behavior
- Loss trajectories

## Technical Details

### Tensor Implementation

The custom Tensor class implements:
- Automatic gradient computation using reverse-mode autodiff
- Computational graph tracking through dependencies
- Broadcasting support for element-wise operations
- Matrix operations (matmul, transpose, reshape)

### Optimization Algorithms

**SGD**: Basic stochastic gradient descent
```
θ = θ - α∇L(θ)
```

**Momentum**: Accelerated gradient descent
```
v = βv - α∇L(θ)
θ = θ + v
```

**Adam**: Adaptive moment estimation
```
m = β₁m + (1-β₁)∇L(θ)
v = β₂v + (1-β₂)(∇L(θ))²
θ = θ - α·m/√(v+ε)
```

**RMSprop**: Root mean square propagation
```
E[g²] = γE[g²] + (1-γ)g²
θ = θ - α·g/√(E[g²]+ε)
```

### Weight Initialization

Supported initialization methods:
- **Xavier/Glorot**: Optimal for tanh/sigmoid activations
- **He**: Optimal for ReLU activations
- **Random**: Uniform or normal distribution

## License

This project is available under the MIT License.

## Acknowledgments

- Pima Indians Diabetes Dataset from UCI Machine Learning Repository
- Inspired by PyTorch and TensorFlow frameworks
- Built as part of Computational Intelligence coursework

## Author

Computational Intelligence Course Project

## Future Improvements

- Add batch normalization layers
- Implement dropout for regularization
- Add more advanced optimizers (AdaGrad, Nadam)
- Support for recurrent layers (LSTM, GRU)
- GPU acceleration using CuPy
- Automatic model serialization/deserialization
- Training callbacks and early stopping
- Learning rate scheduling
