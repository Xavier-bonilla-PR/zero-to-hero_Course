import numpy as np
import matplotlib.pyplot as plt
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

# Generate more complex synthetic data with multiple features and non-linear relationships
np.random.seed(42)
n_samples = 1000
n_features = 3

# Generate complex features
X = np.random.randn(n_samples, n_features)
# Add some non-linear transformations
X[:, 1] = np.sin(X[:, 0]) + X[:, 1]
X[:, 2] = np.exp(X[:, 2] * 0.1)

# Generate target with non-linear relationship
y = (2 * np.sin(X[:, 0]) + 
     0.5 * X[:, 1]**2 + 
     0.1 * np.exp(X[:, 2]) + 
     np.random.randn(n_samples) * 0.1)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()

# Custom activation function using micrograd
def tanh(x: Value) -> Value:
    # Implementing tanh using existing ops
    exp_2x = (x * 2.0).exp()
    return (exp_2x - 1.0) / (exp_2x + 1.0)

# Data loader for mini-batches
class DataLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
    def __iter__(self):
        indices = np.random.permutation(len(self.X))
        for i in range(0, len(self.X), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            X_batch = [[Value(x) for x in row] for row in self.X[batch_idx]]
            y_batch = [Value(y) for y in self.y[batch_idx]]
            yield X_batch, y_batch

# Modified MLP with custom initialization and dropout
class ModifiedMLP:
    def __init__(self, n_inputs, layer_sizes, dropout_rate=0.1):
        self.mlp = MLP(n_inputs, layer_sizes)
        self.dropout_rate = dropout_rate
        self.training = True
        
        # Xavier initialization
        for p in self.mlp.parameters():
            n_inputs = len(self.mlp.layers[0].neurons[0].w)
            p.data = np.random.randn() * np.sqrt(2.0 / n_inputs)
    
    def __call__(self, x, apply_dropout=True):
        if self.training and apply_dropout:
            # Apply dropout during training
            mask = np.random.binomial(1, 1-self.dropout_rate, len(x))
            x = [xi if m else Value(0.0) for xi, m in zip(x, mask)]
        return self.mlp(x)
    
    def parameters(self):
        return self.mlp.parameters()
    
    def zero_grad(self):
        self.mlp.zero_grad()

# Training configuration
batch_size = 32
epochs = 150
learning_rate = 0.01
model = ModifiedMLP(n_features, [64, 32, 16, 1], dropout_rate=0.1)
dataloader = DataLoader(X, y, batch_size)

# Learning rate scheduler
def cosine_scheduler(epoch, max_epochs, initial_lr):
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

# Training metrics
train_losses = []
val_losses = []

# Training loop with validation
for epoch in range(epochs):
    epoch_losses = []
    model.training = True
    current_lr = cosine_scheduler(epoch, epochs, learning_rate)
    
    for X_batch, y_batch in DataLoader(X[:800], y[:800], batch_size):  # Training set
        # Forward pass
        y_pred = [model(x) for x in X_batch]
        
        # Compute loss with L2 regularization
        mse_loss = sum((yp - yt) * (yp - yt) for yp, yt in zip(y_pred, y_batch))
        l2_reg = sum(p * p for p in model.parameters())
        loss = mse_loss * (1.0 / len(y_pred)) + 0.01 * l2_reg
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update with gradient clipping
        grad_norm = np.sqrt(sum(p.grad ** 2 for p in model.parameters()))
        clip_value = 1.0
        
        if grad_norm > clip_value:
            scaling_factor = clip_value / grad_norm
        else:
            scaling_factor = 1.0
            
        for p in model.parameters():
            p.data -= current_lr * (p.grad * scaling_factor)
        
        epoch_losses.append(loss.data)
    
    # Validation phase
    model.training = False
    val_predictions = []
    val_targets = []
    
    with np.nditer([X[800:], y[800:]], flags=['external_loop']) as it:
        for x_val, y_val in it:
            x_val_value = [Value(xi) for xi in x_val]
            pred = model(x_val_value, apply_dropout=False).data
            val_predictions.append(pred)
            val_targets.append(y_val)
    
    val_mse = np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2)
    val_losses.append(val_mse)
    train_losses.append(np.mean(epoch_losses))
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_mse:.4f}, LR: {current_lr:.6f}')

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Training and Validation Loss
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

# Plot 2: Learning Rate Schedule
plt.subplot(1, 3, 2)
lr_schedule = [cosine_scheduler(e, epochs, learning_rate) for e in range(epochs)]
plt.plot(lr_schedule)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')

# Plot 3: Predictions vs Actual
model.training = False
test_predictions = []
for x_test in X[800:]:
    x_test_value = [Value(xi) for xi in x_test]
    pred = model(x_test_value, apply_dropout=False).data
    test_predictions.append(pred)

plt.subplot(1, 3, 3)
plt.scatter(y[800:], test_predictions, alpha=0.5)
plt.plot([y[800:].min(), y[800:].max()], [y[800:].min(), y[800:].max()], 'r--')
plt.title('Predictions vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
