# Monitoring Model Training in PyTorch with Callbacks and Logging - GeeksforGeeks
[Link to Article](https://www.geeksforgeeks.org/monitoring-model-training-in-pytorch-with-callbacks-and-logging/)

Monitoring model training is crucial for understanding the performance and behavior of your machine learning models. PyTorch provides several mechanisms to facilitate this, including the use of callbacks and logging. This article will guide you through the process of using these tools effectively.

Table of Content

- [Monitoring Model Training in PyTorch with Callbacks and Logging - GeeksforGeeks](#monitoring-model-training-in-pytorch-with-callbacks-and-logging---geeksforgeeks)
  - [Understanding Callbacks and Logging in PyTorch](#understanding-callbacks-and-logging-in-pytorch)
  - [Implementing Callbacks and Logging in PyTorch](#implementing-callbacks-and-logging-in-pytorch)
    - [Step 1: Installing necessary libraries](#step-1-installing-necessary-libraries)
    - [Step 2: Importing Necessary Libraries](#step-2-importing-necessary-libraries)
    - [Step 3: Creating a Custom Logging Callback](#step-3-creating-a-custom-logging-callback)
    - [Step 4: Defining Dataset and DataLoader](#step-4-defining-dataset-and-dataloader)
    - [Step 5: Defining the Model](#step-5-defining-the-model)
    - [Step 6: Creating an Early Stopping Callback](#step-6-creating-an-early-stopping-callback)
    - [Step 7: Defining Training Function with Callbacks](#step-7-defining-training-function-with-callbacks)
    - [Step 8: Train the Model](#step-8-train-the-model)
    - [Step 9. Visualizing Training Loss and Accuracy Over Epochs](#step-9-visualizing-training-loss-and-accuracy-over-epochs)
  - [How to use callbacks and logging in PyTorch for monitoring model training? – FAQ’s](#how-to-use-callbacks-and-logging-in-pytorch-for-monitoring-model-training--faqs)
    - [1. What are PyTorch callbacks?](#1-what-are-pytorch-callbacks)
    - [2. Why use logging in PyTorch?](#2-why-use-logging-in-pytorch)
    - [3. How to implement callbacks and logging in PyTorch?](#3-how-to-implement-callbacks-and-logging-in-pytorch)

Understanding Callbacks and Logging in PyTorch
----------------------------------------------

*   Callbacks in PyTorch are functions or classes that can be used to insert custom logic at various stages of the training loop. They are useful for tasks like logging, early stopping, learning rate scheduling, and saving models. While PyTorch does not have a built-in callback system like some other frameworks (e.g., Keras), you can implement callbacks by customizing the training loop or using third-party libraries like `pytorch-lightning` or `ignite`.
*   Logging is an important part of training models to keep track of metrics like loss and accuracy over time. PyTorch does not provide a built-in logging system, but you can use Python’s `logging` module or integrate with logging libraries such as TensorBoard or `wandb` (Weights and Biases).

Logging involves recording information about the training process, which can include Loss values, Accuracy scores, Time taken for each epoch or batch, Any other metric or state of interest.

Implementing Callbacks and Logging in PyTorch
---------------------------------------------

### Step 1: Installing necessary libraries

Begin by setting up your environment to ensure you have PyTorch installed.

```
pip install torch torchvision
```


### Step 2: Importing Necessary Libraries

Import the necessary libraries for building and training your model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from time import time
```

### Step 3: Creating a Custom Logging Callback

Define a custom callback class for logging training progress. The ****TrainingLogger**** class will handle logging at the beginning and end of each epoch, as well as after every specified number of batches.

```python
training_logs = []
class TrainingLogger:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs['epoch_time'] = elapsed_time  # Add epoch time to logs
        training_logs.append(logs)  # Collect training logs

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            logging.info(f"Batch {batch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")
```

### Step 4: Defining Dataset and DataLoader

Create a dataset and DataLoader for loading data in batches. For simplicity, we’ll use randomly generated data.

```python
class RandomDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.num_samples = num_samples
        self.num_features = num_features
        self.data = torch.randn(num_samples, num_features)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = RandomDataset(1000, 20)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Step 5: Defining the Model

Define a simple neural network model. Here we are using a single fully connected layer.

```python
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleNN(20)
```

### Step 6: Creating an Early Stopping Callback

Define an early stopping callback to stop training when performance stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['loss']
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logging.info("Early stopping triggered.")
                return True
        return False
```

### Step 7: Defining Training Function with Callbacks

Create a training function that uses the custom callback for logging. This function will train the model for a specified number of epochs and log progress.

```python
def train_model(model, dataloader, criterion, optimizer, epochs, callback, early_stopping_callback=None):
    model.train()
    
    for epoch in range(epochs):
        callback.on_epoch_begin(epoch)
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            
            callback.on_batch_end(batch_idx, logs={
                'loss': running_loss / (batch_idx + 1),
                'accuracy': correct_predictions / total_predictions
            })
        
        logs = {
            'loss': running_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions
        }
        callback.on_epoch_end(epoch, logs)
        
        if early_stopping_callback and early_stopping_callback.on_epoch_end(epoch, logs):
            break
```

### Step 8: Train the Model

Initialize the criterion, optimizer, and callback, then start training the model. The callback will log progress during training.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
callback = TrainingLogger(log_interval=10)
early_stopping_callback = EarlyStopping(patience=2)

train_model(model, dataloader, criterion, optimizer, epochs=5, callback=callback, early_stopping_callback=early_stopping_callback)
```

****Output:****

```
Epoch 1 starting.
Batch 10: Loss = 0.7636, Accuracy = 0.4688
Batch 20: Loss = 0.7596, Accuracy = 0.4750
Batch 30: Loss = 0.7543, Accuracy = 0.4833
Epoch 1 finished in 0.08 seconds.
Epoch 2 starting.
Batch 10: Loss = 0.7277, Accuracy = 0.5062
Batch 20: Loss = 0.7469, Accuracy = 0.4828
Batch 30: Loss = 0.7394, Accuracy = 0.4896
Epoch 2 finished in 0.06 seconds.
Epoch 3 starting.
Batch 10: Loss = 0.7269, Accuracy = 0.4938
Batch 20: Loss = 0.7267, Accuracy = 0.4891
Batch 30: Loss = 0.7274, Accuracy = 0.4885
Epoch 3 finished in 0.05 seconds.
Epoch 4 starting.
Batch 10: Loss = 0.7284, Accuracy = 0.4531
Batch 20: Loss = 0.7194, Accuracy = 0.4797
Batch 30: Loss = 0.7176, Accuracy = 0.4896
Epoch 4 finished in 0.05 seconds.
Epoch 5 starting.
Batch 10: Loss = 0.7091, Accuracy = 0.4969
Batch 20: Loss = 0.7074, Accuracy = 0.5047
Batch 30: Loss = 0.7108, Accuracy = 0.4969
Epoch 5 finished in 0.04 seconds.
```


The output is a series of log messages generated by the ****TrainingLogger**** callback during the training process of a neural network. These messages provide insights into the progress of training, including the start and end of each epoch and batch-level performance metrics. These logs are crucial for monitoring the model’s training process, allowing you to detect issues like overfitting early and make necessary adjustments to improve model performance. For e.g.

*   Epoch 1
    *   ****Batch 10: Loss = 0.7439, Accuracy = 0.5531****
    *   After processing 10 batches, the average loss is 0.7439 and the accuracy is 55.31%.
*   ****Batch 20: Loss = 0.8000, Accuracy = 0.5156****
    *   After processing 20 batches, the average loss has increased to 0.8000, and the accuracy has decreased to 51.56%.
*   ****Batch 30: Loss = 0.7787, Accuracy = 0.5354****
    *   After processing 30 batches, the average loss is 0.7787, and the accuracy has slightly improved to 53.54%.

Similarly we can interpret other epochs as shown in output image.

### Step 9. Visualizing Training Loss and Accuracy Over Epochs

****1\. Loss over epochs and accuracy over epochs****

We will extract the loss and accuracy values from the logs generated during training. These plots help visualize the trends in loss reduction and accuracy improvement over the course of training.

```python
import matplotlib.pyplot as plt

# Extract loss and accuracy values from logs
loss_values = [logs['loss'] for logs in training_logs]
accuracy_values = [logs['accuracy'] for logs in training_logs]
epochs = range(1, len(loss_values) + 1)

# Plot loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_values, label='Loss', marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy_values, label='Accuracy', marker='o', color='green')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

****Output:****

![Screenshot-2024-05-28-120335](https://media.geeksforgeeks.org/wp-content/uploads/20240528120451/Screenshot-2024-05-28-120335.png)

Training loss over epochs

![Screenshot-2024-05-28-120350](https://media.geeksforgeeks.org/wp-content/uploads/20240528120450/Screenshot-2024-05-28-120350.jpg)

Training accuracy over epochs

****2\. Training time visualization****

We can extract the time taken for each epoch from the training logs. Then using a bar plot, we will visualize the time taken for each epoch. This plot provides insights into the training duration and helps identify any outliers.

```python
# Extract epoch times
epoch_times = [logs['epoch_time'] for logs in training_logs]

# Plot training time per epoch
plt.figure(figsize=(10, 5))
plt.bar(epochs, epoch_times, color='orange')
plt.title('Training Time per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Time (seconds)')
plt.grid(axis='y')
plt.show()
```

****Output:****

![Screenshot-2024-05-28-120611](https://media.geeksforgeeks.org/wp-content/uploads/20240528120734/Screenshot-2024-05-28-120611.png)

Training time visualization

How to use callbacks and logging in PyTorch for monitoring model training? – FAQ’s
----------------------------------------------------------------------------------

### 1\. What are PyTorch callbacks?

> PyTorch callbacks are functions triggered at specific points during model training, allowing for custom actions like logging, early stopping, or checkpointing.

### 2\. Why use logging in PyTorch?

> Logging in PyTorch records training progress, including metrics like loss and accuracy, facilitating real-time monitoring and performance evaluation.

### 3\. How to implement callbacks and logging in PyTorch?

> Implement custom callbacks for desired actions and integrate them into the training loop for seamless logging and monitoring during model training.
