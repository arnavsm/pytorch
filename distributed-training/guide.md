# Distributed Training in PyTorch

## Introduction
Distributed training in PyTorch allows you to train deep learning models across multiple GPUs and machines, significantly reducing training time and enabling the training of larger models. This guide covers different approaches to distributed training and their implementation.

## Table of Contents
- [Distributed Training in PyTorch](#distributed-training-in-pytorch)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Data Parallel vs Distributed Data Parallel](#data-parallel-vs-distributed-data-parallel)
    - [DataParallel (DP)](#dataparallel-dp)
    - [DistributedDataParallel (DDP)](#distributeddataparallel-ddp)
  - [Setting Up Distributed Training](#setting-up-distributed-training)
    - [Basic Setup](#basic-setup)
    - [Dataset and DataLoader Setup](#dataset-and-dataloader-setup)
  - [DistributedDataParallel Implementation](#distributeddataparallel-implementation)
    - [Complete Training Script](#complete-training-script)
  - [Multi-Node Training](#multi-node-training)
    - [Launch Script](#launch-script)
  - [Best Practices](#best-practices)
  - [Common Issues and Solutions](#common-issues-and-solutions)

## Data Parallel vs Distributed Data Parallel

### DataParallel (DP)
- Simplest form of model parallelism
- Single-process, multi-thread
- Higher overhead due to GIL (Global Interpreter Lock)
- Suitable for small-scale training

```python
model = nn.DataParallel(model)
```

### DistributedDataParallel (DDP)
- Multi-process parallelism
- Better performance than DP
- Each GPU runs its own process
- Supports both single-node and multi-node setups

## Setting Up Distributed Training

### Basic Setup
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
```

### Dataset and DataLoader Setup
```python
from torch.utils.data.distributed import DistributedSampler

def create_data_loader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return loader
```

## DistributedDataParallel Implementation

### Complete Training Script
```python
def train(rank, world_size, epochs):
    # Setup distribution
    setup(rank, world_size)
    
    # Create model and move it to GPU
    model = YourModel().to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Setup data
    train_loader = create_data_loader(train_dataset, batch_size, rank, world_size)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0:  # Print only on master process
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
    
    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        args=(world_size, 10),  # 10 epochs
        nprocs=world_size,
        join=True
    )
```

## Multi-Node Training

### Launch Script
```python
# launch.py
import os
import subprocess

def launch_training():
    world_size = 8  # Total number of GPUs across all nodes
    nodes = ["node1", "node2"]  # List of node addresses
    node_rank = 0  # Rank of this node
    
    # Environment variables for distributed training
    os.environ["MASTER_ADDR"] = nodes[0]
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = str(node_rank)
    
    # Launch training script
    subprocess.call([
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node=4",  # GPUs per node
        "--nnodes=2",          # Total number of nodes
        "--node_rank=0",       # Rank of this node
        "--master_addr=node1", # Master node address
        "--master_port=12355", # Master port
        "train.py"            # Your training script
    ])
```

## Best Practices

1. **Data Loading**
   - Use `num_workers` in DataLoader for efficient data loading
   - Enable `pin_memory=True` for faster data transfer to GPU
   - Use appropriate batch size per GPU

2. **Model Initialization**
   - Initialize model weights on CPU first
   - Move model to GPU after DistributedDataParallel wrapping
   - Use `model.module` to access the underlying model

3. **Performance Optimization**
   - Use NCCL backend for GPU training
   - Set appropriate batch size per GPU
   - Enable gradient accumulation for larger effective batch sizes
   ```python
   accumulation_steps = 4
   for i, (data, target) in enumerate(train_loader):
       output = model(data)
       loss = criterion(output, target) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Debugging**
   - Use `torch.distributed.barrier()` for synchronization
   - Print debugging information only on master process (rank 0)
   - Set appropriate timeout values for initialization

## Common Issues and Solutions

1. **Process Group Initialization Failures**
   - Ensure all nodes can communicate
   - Check firewall settings
   - Verify consistent world size across nodes

2. **Memory Issues**
   - Adjust batch size per GPU
   - Enable gradient accumulation
   - Use mixed precision training

3. **Uneven Data Distribution**
   - Use DistributedSampler
   - Set `drop_last=True` in DataLoader for consistent batches

Remember to clean up resources after training:
```python
try:
    # Training code here
finally:
    dist.destroy_process_group()
```