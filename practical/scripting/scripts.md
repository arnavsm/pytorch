# Guide to Shell Scripts: Understanding and Executing Them

Shell scripts (usually .sh files) are commonly used to automate tasks in projects like data preprocessing, training models, or running workflows. Below is a comprehensive guide to understanding, making them executable, and running them.

## What Are Shell Scripts?

Shell scripts are text files containing a series of commands to be executed by a shell (like Bash). These scripts are commonly used for automating tasks and workflows.

## How to Execute a Script

### 1. Make the Script Executable

Before running a shell script, ensure it has execute permissions. This allows the shell to treat the script as an executable program.

To make a script executable:

```bash
chmod +x script.sh
```

This command adds execute permissions to script.sh.

### 2. Execute the Script

Once the script is executable, you have a few options to run it:
- Run Directly (if the script starts with a shebang like `#!/bin/bash`):

```bash
./script.sh
```

- Run with Bash (if you don't want to modify the file permissions):

```bash
bash script.sh
```

- Run with sh (if you're using a different shell or sh is your default shell):

```bash
sh script.sh
```

Note: The `./` is used to tell the shell to look for the script in the current directory. Without this, the shell will look for the script in directories listed in your PATH.

### 3. Passing Arguments to the Script

Many scripts take input parameters (arguments) to make them more flexible. For example, if a script takes an input file and an output file:

```bash
./script.sh input.csv output.csv
```

Inside the script, these arguments are accessed using `$1`, `$2`, etc.

## Example of a Shell Script

Script: `train_model.sh`

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

This trains the model using data/train.csv for 20 epochs.

3. Run it with the default number of epochs (10):

```bash
./train_model.sh data/train.csv
```

### Breaking Down the Script
1. **Shebang (`#!/bin/bash`):**
   - This line indicates that the script should be run using Bash.
2. **Arguments:**
   - `$1`: The first argument passed to the script (data/train.csv).
   - `${2:-10}`: The second argument (number of epochs) with a default value of 10.
3. **Activate Conda Environment:**
   - `conda activate kaggle_env` activates the required environment for running the script.
4. **Run Python Command:**
   - `python src/train.py --data $DATASET_PATH --epochs $EPOCHS` runs the Python training script with the provided arguments.
5. **Deactivate Conda Environment:**
   - `conda deactivate` deactivates the environment after execution.
6. **Echo Statements:**
   - The echo commands print messages to the terminal for tracking progress.

## Creating Your Own Shell Script

### Basic Template

```bash
#!/bin/bash

# Check if arguments are provided
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

### 1. Conditional Execution
Use if conditions for branching logic:

```bash
if [ condition ]; then
    # Commands
else
    # Other commands
fi
```

### 2. Loops
Loop through files or iterate over values:

```bash
for FILE in *.csv; do
    echo "Processing $FILE"
done
```

### 3. Error Handling
Enable immediate exit on errors:

```bash
set -e  # Exit on any error
```

### 4. Logging
Redirect output to a log file:

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

## Summary
1. Make a script executable: `chmod +x script.sh`
2. Run the script:
   - Directly: `./script.sh`
   - With Bash: `bash script.sh`
   - With sh: `sh script.sh`
3. Pass arguments to scripts as needed.
4. Use `echo` and other commands for logging and output messages.

Let me know if you need more help or further clarification!
