# MNIST Neural Network Classification Analysis

This analysis accompanies the implementation in [main.py](main.py) which implements a digit classifier using PyTorch.

## Section 1: Data Preparation
- **Implementation:** [main.py (L49-L89)](main.py#L49-L89)

The MNIST data is loaded with torchvision, combined into one pool, normalized, and split into 60% training, 20% validation, and 20% testing with stratification. The script then creates PyTorch DataLoader objects for repeatable training and evaluation.

## Section 2: Neural Network Architectures
- **FNN Implementation:** [main.py (L110-L127)](main.py#L110-L127)
- **CNN Implementation:** [main.py (L131-L177)](main.py#L131-L177)

The feed-forward network is configurable through hidden-layer sizes and optional dropout. The convolutional network adds convolution layers, optional dropout, and optional layer normalization to compare architectural choices.

## Section 3: Training Loop
- **Implementation:** [main.py (L184-L245)](main.py#L184-L245)

The training loop uses stochastic gradient descent with cross-entropy loss. Each epoch records training and validation loss and accuracy, keeping the evaluation flow explicit instead of hiding it inside a framework helper.

## Section 4: Baseline Training
- **Implementation:** [main.py (L325-L349)](main.py#L325-L349)

The baseline run uses the default learning rate, batch size, and two-hidden-layer architecture. Its curves and confusion matrix provide the reference point for later comparisons.

## Section 5: Hyperparameter Sweeps
- **Implementation:** [main.py (L351-L456)](main.py#L351-L456)

The analysis changes one factor at a time: learning rate, batch size, and network architecture. The best feed-forward configuration is selected using validation accuracy and then evaluated on the held-out test set.

## Section 6: CNN and Summary
- **Implementation:** [main.py (L459-L526)](main.py#L459-L526)

The CNN bonus compares dropout and layer normalization settings. The summary prints the baseline feed-forward accuracy, best feed-forward accuracy, and best CNN accuracy when the full script is allowed to finish.
