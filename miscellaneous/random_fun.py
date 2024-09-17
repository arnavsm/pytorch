import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# Fun Fact 1: Hooks for Debugging Models
print("Fun Fact 1: Using hooks to debug PyTorch models")

def print_layer_output(name):
    def hook(model, input, output):
        print(f"Layer: {name}")
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item()}")
        print(f"Output std: {output.std().item()}")
        print("------------------------")
    return hook

def debug_model_with_hooks(model, input_size):
    # Register hooks for each layer
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            layer.register_forward_hook(print_layer_output(name))
    
    # Forward pass with random input
    x = torch.randn(input_size)
    model(x)

# Example usage
resnet18 = models.resnet18()
debug_model_with_hooks(resnet18, (1, 3, 224, 224))

# Fun Fact 2: Custom Autograd Functions
print("\nFun Fact 2: Creating custom autograd functions")

class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

custom_relu = CustomReLU.apply

# Example usage
x = torch.randn(5, 5, requires_grad=True)
y = custom_relu(x)
z = y.sum()
z.backward()
print("Custom ReLU gradient:", x.grad)

# Fun Fact 3: Tensor Memory Sharing
print("\nFun Fact 3: Tensor memory sharing with .view() and .reshape()")

def demonstrate_memory_sharing():
    x = torch.randn(4, 4)
    y = x.view(16)
    z = x.reshape(16)
    
    print("Original tensor:", x)
    print("View tensor:", y)
    print("Reshaped tensor:", z)
    
    x[0, 0] = 100
    print("\nAfter modifying x:")
    print("Original tensor:", x)
    print("View tensor:", y)
    print("Reshaped tensor:", z)
    
    print("\nMemory address of x:", x.data_ptr())
    print("Memory address of y:", y.data_ptr())
    print("Memory address of z:", z.data_ptr())

demonstrate_memory_sharing()

# Fun Fact 4: Custom Dataset and DataLoader
print("\nFun Fact 4: Creating custom datasets and dataloaders")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(1000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: Data shape: {data.shape}, Labels shape: {labels.shape}")
    if batch_idx == 2:
        break

# Fun Fact 5: Model Surgery
print("\nFun Fact 5: Performing model surgery")

def replace_relu_with_leaky(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(negative_slope=0.1))
        else:
            replace_relu_with_leaky(child)

resnet18 = models.resnet18()
print("Before surgery:")
print(resnet18)

replace_relu_with_leaky(resnet18)
print("\nAfter surgery:")
print(resnet18)

# Fun Fact 6: Custom Optimizers
print("\nFun Fact 6: Implementing custom optimizers")

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = dict(lr=lr)
        super(CustomSGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)
        
        return loss

# Example usage
model = nn.Linear(10, 1)
optimizer = CustomSGD(model.parameters(), lr=0.1)

for _ in range(5):
    optimizer.zero_grad()
    output = model(torch.randn(5, 10))
    loss = F.mse_loss(output, torch.randn(5, 1))
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item():.4f}")

# Fun Fact 7: Gradient Accumulation
print("\nFun Fact 7: Gradient accumulation for larger batch sizes")

def train_with_gradient_accumulation(model, dataloader, num_accumulation_steps):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    for batch_idx, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss = loss / num_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if batch_idx == 10:  # Just for demonstration
            break

model = nn.Linear(10, 2)
dataset = CustomDataset(1000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
train_with_gradient_accumulation(model, dataloader, num_accumulation_steps=4)

# Fun Fact 8: Mixed Precision Training
print("\nFun Fact 8: Mixed precision training")

def mixed_precision_training():
    model = nn.Linear(1000, 100).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    for _ in range(10):  # Just for demonstration
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output = model(torch.randn(100, 1000).cuda())
            loss = F.mse_loss(output, torch.randn(100, 100).cuda())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Loss: {loss.item():.4f}")

if torch.cuda.is_available():
    mixed_precision_training()
else:
    print("CUDA not available, skipping mixed precision training example")

# Fun Fact 9: Custom Loss Functions
print("\nFun Fact 9: Implementing custom loss functions")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Example usage
model = nn.Linear(10, 5)
criterion = FocalLoss(alpha=0.5, gamma=2)
optimizer = torch.optim.Adam(model.parameters())

for _ in range(5):
    optimizer.zero_grad()
    output = model(torch.randn(100, 10))
    target = torch.randint(0, 5, (100,))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"Focal Loss: {loss.item():.4f}")

# Fun Fact 10: Model Quantization
print("\nFun Fact 10: Model quantization for efficiency")

def quantize_model():
    model = models.resnet18(pretrained=True).eval()
    
    # Fuse Conv, BN and ReLU layers
    model = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']])
    
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data
    input_fp32 = torch.randn(1, 3, 224, 224)
    model_prepared(input_fp32)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    # Compare model sizes
    torch.save(model.state_dict(), "model_fp32.pth")
    torch.save(model_quantized.state_dict(), "model_int8.pth")
    
    fp32_size = os.path.getsize("model_fp32.pth") / (1024 * 1024)
    int8_size = os.path.getsize("model_int8.pth") / (1024 * 1024)
    
    print(f"FP32 model size: {fp32_size:.2f} MB")
    print(f"INT8 model size: {int8_size:.2f} MB")
    print(f"Compression ratio: {fp32_size / int8_size:.2f}x")
    
    # Clean up
    os.remove("model_fp32.pth")
    os.remove("model_int8.pth")

quantize_model()

# Fun Fact 11: Implementing Attention Mechanism
print("\nFun Fact 11: Implementing attention mechanism")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
        
    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

# Example usage
hidden_size = 256
attention = Attention(hidden_size)
hidden = torch.randn(1, 1, hidden_size)
encoder_outputs = torch.randn(10, 1, hidden_size)
attn_weights = attention(hidden, encoder_outputs)
print("Attention weights shape:", attn_weights.shape)

# Fun Fact 12: Implementing Gradient Clipping
print("\nFun Fact 12: Implementing gradient clipping")

def train_with_gradient_clipping(model, dataloader, clip_value):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        if batch_idx == 50:  # Just for demonstration
            break

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)
dataset = CustomDataset(1000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
train_with_gradient_clipping(model, dataloader, clip_value=1.0)

# Fun Fact 13: Implementing Layer Normalization
print("\nFun Fact 13: Implementing layer normalization")

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Example usage
layer_norm = LayerNorm(10)
input_tensor = torch.randn(20, 10)
normalized_output = layer_norm(input_tensor)
print("Layer normalized output shape:", normalized_output.shape)

# Fun Fact 14: Implementing a Simple Transformer Block
print("\nFun Fact 14: Implementing a simple Transformer block")

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Example usage
transformer_block = TransformerBlock(d_model=512, nhead=8)
input_tensor = torch.randn(10, 32, 512)  # (seq_len, batch_size, d_model)
output = transformer_block(input_tensor)
print("Transformer block output shape:", output.shape)

# Fun Fact 15: Implementing Cosine Annealing Learning Rate Scheduler
print("\nFun Fact 15: Implementing cosine annealing learning rate scheduler")

def cosine_annealing_lr(initial_lr, current_step, total_steps):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * current_step / total_steps))

def train_with_cosine_annealing(model, dataloader, initial_lr, total_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)
    total_steps = len(dataloader) * total_epochs
    
    for epoch in range(total_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            current_step = epoch * len(dataloader) + batch_idx
            lr = cosine_annealing_lr(initial_lr, current_step, total_steps)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, LR: {lr:.6f}, Loss: {loss.item():.4f}")
            
            if batch_idx == 50:  # Just for demonstration
                break

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)
dataset = CustomDataset(1000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
train_with_cosine_annealing(model, dataloader, initial_lr=0.1, total_epochs=5)

# Fun Fact 16: Implementing Exponential Moving Average (EMA)
print("\nFun Fact 16: Implementing Exponential Moving Average (EMA)")

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Example usage
model = nn.Linear(10, 5)
ema = EMA(model, decay=0.999)
ema.register()

for _ in range(100):
    # Training loop
    loss = torch.rand(1).item()
    loss.backward()
    
    # Update EMA
    ema.update()

# Apply EMA for inference
ema.apply_shadow()
# Perform inference here
# ...
# Restore original parameters
ema.restore()

# Fun Fact 17: Implementing Focal Loss
print("\nFun Fact 17: Implementing Focal Loss")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Example usage
model = nn.Linear(10, 5)
criterion = FocalLoss(alpha=0.25, gamma=2)
optimizer = torch.optim.Adam(model.parameters())

for _ in range(10):
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 5, (100,))
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Focal Loss: {loss.item():.4f}")

# Fun Fact 18: Implementing Cutout Data Augmentation
print("\nFun Fact 18: Implementing Cutout Data Augmentation")

def cutout(image, n_holes, length):
    h, w = image.size(1), image.size(2)
    mask = torch.ones((h, w), device=image.device)
    
    for _ in range(n_holes):
        y = torch.randint(h, (1,))
        x = torch.randint(w, (1,))
        
        y1 = torch.clamp(y - length // 2, 0, h)
        y2 = torch.clamp(y + length // 2, 0, h)
        x1 = torch.clamp(x - length // 2, 0, w)
        x2 = torch.clamp(x + length // 2, 0, w)
        
        mask[y1:y2, x1:x2] = 0
    
    mask = mask.expand_as(image)
    image = image * mask
    
    return image

# Example usage
image = torch.randn(3, 32, 32)
augmented_image = cutout(image, n_holes=1, length=16)
print("Original image shape:", image.shape)
print("Augmented image shape:", augmented_image.shape)

# Fun Fact 19: Implementing Mixup Data Augmentation
print("\nFun Fact 19: Implementing Mixup Data Augmentation")

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Example usage
criterion = nn.CrossEntropyLoss()
model = nn.Linear(10, 5)
optimizer = torch.optim.Adam(model.parameters())

for _ in range(10):
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 5, (100,))
    
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
    outputs = model(inputs)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Mixup Loss: {loss.item():.4f}")

# Fun Fact 20: Implementing Label Smoothing
print("\nFun Fact 20: Implementing Label Smoothing")

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Example usage
model = nn.Linear(10, 5)
criterion = LabelSmoothingLoss(classes=5, smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters())

for _ in range(10):
    inputs = torch.randn(100, 10)
    targets = torch.randint(0, 5, (100,))
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Label Smoothing Loss: {loss.item():.4f}")

print("\nThese fun facts and implementations showcase various advanced techniques and concepts in PyTorch!")





