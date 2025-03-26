import torch

def tensor_creation():
    """Create basic tensors for demonstration"""
    x = torch.arange(6)  # 1D Tensor
    y = x.view(2, 3)  # Reshape (2x3)
    return x, y

def reshape_vs_view():
    """Difference between reshape() and view()"""
    x = torch.arange(6)
    y = x.view(2, 3)  # Only works if contiguous
    z = x.reshape(2, 3)  # Works regardless of memory layout
    return y, z

def permute_vs_transpose():
    """Difference between permute() and transpose()"""
    a = torch.randn(2, 3, 4)  # Shape (2,3,4)
    b = a.transpose(1, 2)  # Swap 2nd and 3rd dim -> shape (2,4,3)
    c = a.permute(2, 0, 1)  # Reorder -> shape (4,2,3)
    return b, c

def dot_and_cross_product():
    """Demonstrates dot product and cross product"""
    vec1 = torch.tensor([1.0, 2.0, 3.0])
    vec2 = torch.tensor([4.0, 5.0, 6.0])
    
    dot_product = torch.dot(vec1, vec2)  # 1*4 + 2*5 + 3*6
    cross_product = torch.cross(vec1, vec2)  # [(-3,6,-3)] for 3D vectors
    return dot_product, cross_product

def matrix_multiplication():
    """Demonstrates different types of matrix multiplication"""
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    
    matmul_result = torch.matmul(A, B)  # Standard matrix multiplication
    mm_result = A @ B  # Alternative syntax for matmul
    return matmul_result, mm_result

def batch_matrix_multiplication():
    """Batch matrix multiplication (3D tensors)"""
    A = torch.randn(10, 3, 4)  # 10 matrices of shape (3,4)
    B = torch.randn(10, 4, 5)  # 10 matrices of shape (4,5)
    C = torch.matmul(A, B)  # Output shape (10,3,5)
    return C

def einsum_example():
    """Demonstrates Einstein summation for advanced matrix ops"""
    A = torch.randn(2, 3, 4)
    B = torch.randn(2, 4, 5)
    C = torch.einsum('ijk,ikm->ijm', A, B)  # (2,3,4) @ (2,4,5) -> (2,3,5)
    return C

def expand_vs_repeat():
    """Difference between expand and repeat"""
    x = torch.tensor([[1], [2], [3]])  # Shape: (3,1)
    expanded = x.expand(3, 4)  # Shape: (3,4) - no memory copy
    repeated = x.repeat(3, 2)  # Shape: (9,2) - creates a copy
    return expanded, repeated

def squeeze_unsqueeze():
    """Demonstrates squeeze() and unsqueeze()"""
    x = torch.tensor([1, 2, 3])  # Shape: (3,)
    x_unsqueezed = x.unsqueeze(0)  # Shape: (1,3)
    x_squeezed = x_unsqueezed.squeeze()  # Back to (3,)
    return x_unsqueezed, x_squeezed

def contiguous_memory():
    """Ensures memory contiguity for view()"""
    x = torch.randn(2, 3)
    y = x.t()  # Transpose makes it non-contiguous
    y_contiguous = y.contiguous().view(6)  # Now it works
    return y_contiguous

def broadcasting_example():
    """Demonstrates broadcasting in tensor operations"""
    A = torch.randn(3, 1)
    B = torch.randn(1, 4)
    C = A + B  # A expands to (3,4), B expands to (3,4)
    return C

# --- Running the functions ---
if __name__ == "__main__":
    print("Tensor Creation:", tensor_creation())
    print("Reshape vs View:", reshape_vs_view())
    print("Permute vs Transpose:", permute_vs_transpose())
    print("Dot and Cross Product:", dot_and_cross_product())
    print("Matrix Multiplication:", matrix_multiplication())
    print("Batch Matrix Multiplication:", batch_matrix_multiplication())
    print("Einsum Example:", einsum_example())
    print("Expand vs Repeat:", expand_vs_repeat())
    print("Squeeze vs Unsqueeze:", squeeze_unsqueeze())
    print("Contiguous Memory:", contiguous_memory())
    print("Broadcasting Example:", broadcasting_example())


# vector x vector
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
torch.matmul(tensor1, tensor2).size()
# matrix x vector
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()
# batched matrix x broadcasted vector
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
torch.matmul(tensor1, tensor2).size()
# batched matrix x batched matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
torch.matmul(tensor1, tensor2).size()
# batched matrix x broadcasted matrix
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size()
