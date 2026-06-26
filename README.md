# MNIST Handwritten Digit Classifier

This repository contains a PyTorch implementation of a handwritten digit classifier using the MNIST dataset. The project demonstrates custom training loops, feedforward neural network architectures, hyperparameter optimization sweeps, and a convolutional neural network (CNN) model comparison.

## Features

- **Custom PyTorch Training Loop (`main.py`)**:
  - Full data preprocessing (stratified 60/20/20 train/validation/test split and normalization).
  - Explicit SGD training loop with cross-entropy loss tracking.
  - Generates loss/accuracy curves and confusion matrices.
- **Configurable FNN Architecture**:
  - Supports configurable depth, layer sizes, and dropout settings.
- **Hyperparameter Optimization Sweeps**:
  - Sweeps learning rates (`0.1`, `0.01`, `0.001`).
  - Sweeps batch sizes (`32`, `64`, `128`).
  - Compares different hidden layer topologies.
- **Convolutional Neural Network (CNN)**:
  - CNN bonus model featuring custom layer normalization, pooling, and dropout to compare performance against the standard FNN.

## Structure

- `main.py`: The entry point script that handles data prep, model definitions, sweeps, and evaluations.
- `analysis.md`: Detailed structural and architectural explanation.
- `results/`: Contains generated metric curves and confusion matrix plots.
- `data/`: Local storage for the MNIST dataset (downloaded at runtime).

## Getting Started

### Prerequisites

Install the dependencies:
```bash
pip install -r requirements.txt
```

### Running the Model

To run the full suite (training baseline, sweeps, and CNN evaluation):
```bash
python main.py
```

## Results

Performance plots, sweeps comparisons, and confusion matrices are generated in the `results/` folder.
Detailed execution notes can be found in `results/run_notes.md`.
