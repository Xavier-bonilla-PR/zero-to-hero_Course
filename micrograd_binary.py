# #micrograd can be used for:
# #binary classification, regression problems, time series predictions, multi-class classification, anomaly detection, recommendation system, feature selection/importance

# #binary classification:

# ######################################################################
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from micrograd.engine import Value
from micrograd.nn import MLP

# Set seeds for reproducibility
np.random.seed(1337)
random.seed(1337)

# Generate balanced synthetic credit card transaction data
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_classes=2,
    n_informative=3,
    n_redundant=1,
    weights=[0.5, 0.5],  # Balanced dataset: 50% normal, 50% fraudulent
    random_state=1337
)
y = y*2 - 1  # Convert to -1 (fraud), 1 (normal)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

def visualize_data(X, y):
    feature_names = ['Amount', 'Time', 'Distance', 'Frequency']
    
    # 1. Pair Plot
    plt.figure(figsize=(15, 12))
    
    # Create multiple subplots
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i*4 + j + 1)
            if i == j:
                # Histogram on diagonal
                plt.hist(X[:, i][y == 1], alpha=0.5, label='Normal', bins=20, color='#A8D8EA')
                plt.hist(X[:, i][y == -1], alpha=0.5, label='Fraud', bins=20, color='#FFB3B3')
                plt.xlabel(feature_names[i])
                if i == 0:
                    plt.legend()
            else:
                # Scatter plot on off-diagonal
                plt.scatter(X[:, j][y == 1], X[:, i][y == 1], 
                          alpha=0.5, label='Normal', s=5, color='#A8D8EA')
                plt.scatter(X[:, j][y == -1], X[:, i][y == -1], 
                          alpha=0.5, label='Fraud', s=5, color='#FFB3B3')
                plt.xlabel(feature_names[j])
                plt.ylabel(feature_names[i])
    
    plt.tight_layout()
    plt.show()
    
    # 2. Correlation matrix
    plt.figure(figsize=(8, 6))
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, 
                annot=True, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

def loss_fn(pred, y_true):
    """
    Compute hinge loss (without class weights since data is balanced)
    pred: Value object (raw model output)
    y_true: -1 or 1
    """
    # Simple hinge loss without weights since dataset is balanced
    loss = (Value(1.0) - y_true * pred).relu()
    return loss

def evaluate_metrics(y_true, y_pred):
    """
    Calculate fraud detection specific metrics
    """
    tp = np.sum((y_pred == -1) & (y_true == -1))
    fp = np.sum((y_pred == -1) & (y_true == 1))
    fn = np.sum((y_pred == 1) & (y_true == -1))
    tn = np.sum((y_pred == 1) & (y_true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    }

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, learning_rate=0.01):
    n_samples = len(X_train)
    metrics_history = {
        'train_loss': [],
        'train_metrics': [],
        'test_metrics': []
    }
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_losses = []
        predictions = []
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            batch_loss = Value(0.0)
            batch_preds = []
            
            for x, y in zip(X_batch, y_batch):
                x = [Value(xi) for xi in x]
                pred = model(x)
                batch_preds.append(pred)
                loss = loss_fn(pred, y)
                batch_loss = batch_loss + loss
            
            # Average loss
            batch_loss = batch_loss * (1.0 / len(X_batch))
            
            # Backward pass
            model.zero_grad()
            batch_loss.backward()
            
            # Update parameters
            for p in model.parameters():
                p.data -= learning_rate * p.grad
            
            epoch_losses.append(batch_loss.data)
            predictions.extend([1 if p.data > 0 else -1 for p in batch_preds])
        
        # Calculate metrics
        train_predictions = predict(model, X_train)
        test_predictions = predict(model, X_test)
        
        train_metrics = evaluate_metrics(y_train, train_predictions)
        test_metrics = evaluate_metrics(y_test, test_predictions)
        
        metrics_history['train_loss'].append(np.mean(epoch_losses))
        metrics_history['train_metrics'].append(train_metrics)
        metrics_history['test_metrics'].append(test_metrics)
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train - Loss: {np.mean(epoch_losses):.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"Test  - Accuracy: {test_metrics['accuracy']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}")
    
    return metrics_history

def predict(model, X):
    predictions = []
    for x in X:
        x = [Value(xi) for xi in x]
        pred = model(x)
        predictions.append(1 if pred.data > 0 else -1)
    return np.array(predictions)

def visualize_results(metrics_history):
    epochs = range(len(metrics_history['train_loss']))
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics_history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    train_acc = [m['accuracy'] for m in metrics_history['train_metrics']]
    test_acc = [m['accuracy'] for m in metrics_history['test_metrics']]
    plt.plot(epochs, train_acc, label='Train')
    plt.plot(epochs, test_acc, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 3, 3)
    train_f1 = [m['f1'] for m in metrics_history['train_metrics']]
    test_f1 = [m['f1'] for m in metrics_history['test_metrics']]
    plt.plot(epochs, train_f1, label='Train')
    plt.plot(epochs, test_f1, label='Test')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(metrics):
    cm = metrics['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    cm_matrix = np.array([[cm['tn'], cm['fp']], 
                         [cm['fn'], cm['tp']]])
    
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Visualize initial data
print("Visualizing initial data distribution...")
visualize_data(X, y)

# Print class distribution
print("\nClass Distribution:")
print(f"Normal transactions: {np.sum(y == 1)}")
print(f"Fraudulent transactions: {np.sum(y == -1)}")

# Initialize model
model = MLP(4, [32, 16, 1])  # 4 features, two hidden layers
print("\nModel Architecture:")
print(model)
print(f"Number of parameters: {len(model.parameters())}")

# Train model
print("\nTraining model...")
metrics_history = train_model(
    model, X_train, y_train, X_test, y_test,
    epochs=100,
    batch_size=32,
    learning_rate=0.01
)

# Visualize results
print("\nVisualizing training results...")
visualize_results(metrics_history)

# Final evaluation
final_train_pred = predict(model, X_train)
final_test_pred = predict(model, X_test)

print("\nFinal Model Performance:")
print("\nTraining Set Metrics:")
train_metrics = evaluate_metrics(y_train, final_train_pred)
for metric, value in train_metrics.items():
    if metric != 'confusion_matrix':
        print(f"  {metric}: {value:.4f}")

print("\nTest Set Metrics:")
test_metrics = evaluate_metrics(y_test, final_test_pred)
for metric, value in test_metrics.items():
    if metric != 'confusion_matrix':
        print(f"  {metric}: {value:.4f}")

# Plot confusion matrix for test set
print("\nTest Set Confusion Matrix:")
plot_confusion_matrix(test_metrics)
