Modify the `TrainingLogger` class to output logs to a file, with the default location set to `./logs/output.log`. Here's the updated version:



```python
import logging
import os
from time import time

training_logs = []

class TrainingLogger:
    def __init__(self, log_interval=10, log_file='./logs/output.log'):
        self.log_interval = log_interval
        self.log_file = log_file
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Set up logging to file and console
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[
                                logging.FileHandler(self.log_file),
                                logging.StreamHandler()
                            ])

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        if logs is not None:
            logs['epoch_time'] = elapsed_time  # Add epoch time to logs
            training_logs.append(logs)  # Collect training logs

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0 and logs is not None:
            logging.info(f"Batch {batch + 1}: Loss = {logs.get('loss', 'N/A'):.4f}, Accuracy = {logs.get('accuracy', 'N/A'):.4f}")

```

Here are the key changes and additions to the `TrainingLogger` class:

1. We've added a `log_file` parameter to the `__init__` method, with a default value of `'./logs/output.log'`.

2. We use `os.makedirs()` to ensure that the directory for the log file exists. The `exist_ok=True` parameter prevents an error if the directory already exists.

3. The `logging.basicConfig()` call has been modified to:
   - Use a more detailed format that includes timestamps.
   - Set up two handlers: a `FileHandler` for writing to the log file, and a `StreamHandler` for outputting to the console.

4. We've added some null checks (`if logs is not None`) to prevent errors if `logs` is not provided.

5. In the `on_batch_end` method, we use `logs.get()` with a default value to safely access 'loss' and 'accuracy', in case they're not present in the logs dictionary.

This updated version will now write logs to both the specified file (default: `./logs/output.log`) and the console. The log file will contain timestamped entries for epoch starts, epoch ends (with duration), and batch information at the specified intervals.

To use this logger, you would typically create an instance at the start of your training script and then call its methods at the appropriate points in your training loop. For example:

```python
logger = TrainingLogger(log_interval=10, log_file='./my_custom_logs/training.log')

for epoch in range(num_epochs):
    logger.on_epoch_begin(epoch)
    
    for batch in range(num_batches):
        # ... your training code here ...
        logger.on_batch_end(batch, {'loss': current_loss, 'accuracy': current_accuracy})
    
    logger.on_epoch_end(epoch, {'loss': epoch_loss, 'accuracy': epoch_accuracy})
```

This setup will provide you with detailed logs in your specified file, making it easier to track and analyze your model's training progress.