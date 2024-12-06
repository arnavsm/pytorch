# Guide to Scripts: Understanding and Executing Them

Scripts, especially `.sh` (shell) scripts, are commonly used in Kaggle repositories and other projects for automating repetitive tasks like data preprocessing, training models, and generating predictions. This guide explains how to use and create scripts effectively.

## What Are Shell Scripts?

Shell scripts are plain text files containing a series of commands that are executed in sequence by the shell (e.g., Bash). They are useful for automating workflows.

## How to Execute a Script

### 1. Check the Script Type

Most scripts in Kaggle repositories are `.sh` files (Bash scripts), but you may also encounter Python (`.py`) or other types:
- Bash Script: `.sh`
- Python Script: `.py`

### 2. Grant Execute Permissions

Before running a shell script, ensure it has execute permissions:

```bash
chmod +x script.sh
```

### 3. Execute the Script
- Run Directly (if the file starts with a shebang like `#!/bin/bash`):
  ```bash
  ./script.sh
  ```

- Run with Bash (explicitly specify the shell):
  ```bash
  bash script.sh
  ```

- Run a Python Script:
  ```bash
  python script.py
  ```

### 4. Passing Arguments

Scripts often accept arguments to make them more flexible:
- Example of running a script with arguments:
  ```bash
  ./script.sh input.csv output.csv
  ```

Inside the script, these arguments are accessed as `$1`, `$2`, etc.

## Example of a Shell Script

**Script: train_model.sh**

```bash
#!/bin/bash
# Train a model using a specific dataset

# Arguments
DATASET_PATH=$1     # First argument: path to the dataset
EPOCHS=${2:-10}     # Second argument: number of epochs (default: 10)

# Activate Conda environment
echo "Activating environment..."
conda activate kaggle_env

# Train the model
echo "Training the model..."
python src/train.py --data $DATASET_PATH --epochs $EPOCHS

# Deactivate environment
echo "Deactivating environment..."
conda deactivate

echo "Training completed!"
```

### How to Run the Script
1. Make it executable:
   ```bash
   chmod +x train_model.sh
   ```

2. Run it with arguments:
   ```bash
   ./train_model.sh data/train.csv 20
   ```
   - This trains the model using `data/train.csv` for 20 epochs.

3. Run it without specifying the second argument (it uses the default value):
   ```bash
   ./train_model.sh data/train.csv
   ```

### Breaking Down the Script
1. **Shebang**: `#!/bin/bash`
   - Tells the system to execute the script with Bash.
2. **Arguments**:
   - `$1`: First argument passed to the script.
   - `${2:-10}`: Second argument with a default value of 10.
3. **Activating a Conda Environment**:
   - Use `conda activate` to prepare the environment.
4. **Executing a Python Script**:
   - Runs `train.py` with the specified arguments.
5. **Deactivating the Environment**:
   - Cleans up after the script finishes.
6. **Echo Statements**:
   - Print progress messages to the terminal.

## Creating Your Own Script

### Structure

Here's a typical template for a shell script:

```bash
#!/bin/bash

# Print the script usage
if [ $# -lt 1 ]; then
    echo "Usage: $0 <arg1> [arg2]"
    exit 1
fi

# Parse arguments
ARG1=$1
ARG2=${2:-default_value}

# Main logic
echo "Running with ARG1=$ARG1 and ARG2=$ARG2"

# Execute commands
<your_commands_here>
```

### Example: preprocess_data.sh

This script preprocesses a dataset and saves the output:

```bash
#!/bin/bash

# Arguments
INPUT_FILE=$1
OUTPUT_FILE=${2:-preprocessed_data.csv}

# Print usage if no arguments are provided
if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <input_file> [output_file]"
    exit 1
fi

# Run the preprocessing Python script
python src/preprocess.py --input $INPUT_FILE --output $OUTPUT_FILE

echo "Preprocessing complete. Output saved to $OUTPUT_FILE."
```

## Advanced Shell Script Features

1. **Conditional Execution**:
   ```bash
   if [ condition ]; then
       # Commands
   else
       # Other commands
   fi
   ```

2. **Loops**:
   ```bash
   for FILE in *.csv; do
       echo "Processing $FILE"
   done
   ```

3. **Error Handling**:
   ```bash
   set -e  # Exit on any error
   ```

4. **Logging**:
   ```bash
   LOG_FILE="script.log"
   echo "Starting process..." >> $LOG_FILE
   ```

## Full Workflow Example

### Repository Directory Structure

```
project/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── preprocess.py
│   └── train.py
├── scripts/
│   ├── preprocess_data.sh
│   └── train_model.sh
├── environment.yaml
└── README.md
```

### Run the Workflow
1. Clone the Repository:
   ```bash
   git clone https://github.com/example/kaggle-competition.git
   cd kaggle-competition
   ```

2. Set Up the Environment:
   ```bash
   conda env create -f environment.yaml
   conda activate kaggle_env
   ```

3. Preprocess the Data:
   ```bash
   ./scripts/preprocess_data.sh data/train.csv processed_train.csv
   ```

4. Train the Model:
   ```bash
   ./scripts/train_model.sh processed_train.csv 50
   ```

Let me know if you need help with specific scripts or examples!