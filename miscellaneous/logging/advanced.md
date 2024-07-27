There are several ways we can enhance the TrainingLogger to make it more advanced and versatile. I'll outline some potential improvements and additions:



```python
import logging
import os
from time import time
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class AdvancedTrainingLogger:
    def __init__(self, log_interval=10, log_file='./logs/output.log', metrics=None):
        self.log_interval = log_interval
        self.log_file = log_file
        self.metrics = metrics or ['loss', 'accuracy']
        self.start_time = time()
        self.history = defaultdict(list)
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_file),
                                logging.StreamHandler()
                            ])

    def on_training_begin(self):
        logging.info("Training started.")
        self.start_time = time()

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        if logs:
            logs['epoch_time'] = elapsed_time
            for metric in self.metrics:
                if metric in logs:
                    self.history[metric].append(logs[metric])
            self.history['epoch'].append(epoch + 1)
        self._save_history()

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0 and logs:
            log_message = f"Batch {batch + 1}: "
            log_message += ", ".join([f"{metric.capitalize()} = {logs.get(metric, 'N/A'):.4f}" for metric in self.metrics])
            logging.info(log_message)

    def on_training_end(self):
        total_time = time() - self.start_time
        logging.info(f"Training completed in {total_time:.2f} seconds.")
        self._save_history()
        self._plot_metrics()

    def _save_history(self):
        with open(f"{os.path.splitext(self.log_file)[0]}_history.json", 'w') as f:
            json.dump(self.history, f)

    def _plot_metrics(self):
        plt.figure(figsize=(12, 6))
        for metric in self.metrics:
            if metric in self.history:
                plt.plot(self.history['epoch'], self.history[metric], label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Training Metrics')
        plt.legend()
        plt.savefig(f"{os.path.splitext(self.log_file)[0]}_plot.png")
        plt.close()

    def log_custom_metric(self, metric_name, value):
        logging.info(f"Custom metric - {metric_name}: {value}")
        self.history[metric_name].append(value)

    def set_additional_info(self, info):
        with open(f"{os.path.splitext(self.log_file)[0]}_info.json", 'w') as f:
            json.dump(info, f)

```

Here are the key enhancements in this advanced version:

1. **Flexible Metric Tracking**: The logger now accepts a list of metrics to track, allowing for easy customization.

2. **Training Lifecycle Methods**: Added `on_training_begin()` and `on_training_end()` methods to capture the entire training process.

3. **History Tracking**: All tracked metrics are stored in a `history` dictionary, allowing for easy access to the full training history.

4. **JSON History Export**: The training history is saved as a JSON file after each epoch and at the end of training.

5. **Metric Visualization**: At the end of training, a plot of all tracked metrics is generated and saved as an image.

6. **Custom Metric Logging**: Added a `log_custom_metric()` method to allow logging of arbitrary metrics during training.

7. **Additional Info Logging**: A `set_additional_info()` method allows saving of additional training information (like hyperparameters) as a separate JSON file.

8. **Improved Batch Logging**: The batch logging now dynamically includes all tracked metrics.

To use this advanced logger, you would typically set it up like this:

```python
logger = AdvancedTrainingLogger(log_interval=10, 
                                log_file='./logs/advanced_training.log', 
                                metrics=['loss', 'accuracy', 'val_loss', 'val_accuracy'])

logger.on_training_begin()

for epoch in range(num_epochs):
    logger.on_epoch_begin(epoch)
    
    for batch in range(num_batches):
        # ... your training code here ...
        logger.on_batch_end(batch, {'loss': current_loss, 'accuracy': current_accuracy})
    
    # After validation
    logger.on_epoch_end(epoch, {
        'loss': epoch_loss, 
        'accuracy': epoch_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })

    # Log any custom metrics
    logger.log_custom_metric('learning_rate', current_lr)

logger.on_training_end()

# Log additional information
logger.set_additional_info({
    'model_name': 'MyAwesomeModel',
    'optimizer': 'Adam',
    'batch_size': 32,
    'initial_lr': 0.001
})
```

This advanced logger provides a more comprehensive tracking of your training process, with features that make it easier to analyze and visualize your results. You can further expand on this by adding more visualization options, integrating with monitoring tools like TensorBoard, or adding early stopping functionality based on the tracked metrics.