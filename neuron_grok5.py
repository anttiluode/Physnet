# neuron_grok_diffphys.py
# The Fused Supersolid Neuron: Differentiable Physics Model
# Architecture: (CNN Encoder) -> (GPE Physics Processor) -> (CNN Readout)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm.auto import tqdm
import os

# --- Imports for MNIST ---
import torchvision
import torchvision.transforms as transforms

# ---------------- GPE Physics Step (Differentiable) ----------------
# This is one single, differentiable step of the GPE simulation.
class GPE_Step(nn.Module):
    def __init__(self, dt=0.01, a_coeff=0.15, b_coeff=0.15, device='cpu'):
        super().__init__()
        self.dt = dt
        self.a_coeff = a_coeff # Corresponds to chemical potential mu
        self.b_coeff = b_coeff # Corresponds to interaction strength g/2
        self.device = device
        
    def laplacian(self, psi):
        # Differentiable 2D periodic Laplacian
        # psi shape is (Batch, Grid, Grid)
        # Roll along dim 1 (Height) and dim 2 (Width)
        return (torch.roll(psi, -1, 1) + torch.roll(psi, 1, 1) +
                torch.roll(psi, -1, 2) + torch.roll(psi, 1, 2) - 4 * psi)
        
    def forward(self, psi):
        # psi shape is (Batch, 2, Grid, Grid)
        # We treat the 2 channels as Real and Imaginary
        psi_complex = torch.complex(psi[:, 0], psi[:, 1])

        # --- Differentiable GPE Physics ---
        # H*psi = (-0.5 * lap + V_potential) * psi
        
        lap = self.laplacian(psi_complex)
        kinetic_term = -0.5 * lap
        
        amp_sq = torch.abs(psi_complex)**2
        potential_term = (-self.a_coeff + 2 * self.b_coeff * amp_sq) * psi_complex
        
        H_psi = kinetic_term + potential_term
        
        # Evolve using Euler step: psi_new = psi - i*dt*H*psi
        # (Using Euler for simplicity in the gradient path)
        d_psi = -1j * self.dt * H_psi
        psi_new = psi_complex + d_psi
        
        # Return as 2 real channels
        return torch.stack([torch.real(psi_new), torch.imag(psi_new)], dim=1)

# ---------------- Physics Processor Module ----------------
# This module takes a latent vector, projects it to a field, 
# and runs the GPE steps.
class PhysicsProcessor(nn.Module):
    def __init__(self, latent_dim, grid_size=32, n_steps=5, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.n_steps = n_steps
        self.device = device
        
        # Projector: (B, Latent) -> (B, 2 * 32 * 32)
        self.projector = nn.Linear(latent_dim, 2 * grid_size * grid_size)
        
        # A list of GPE Step modules
        self.gpe_steps = nn.ModuleList([
            GPE_Step(device=device) for _ in range(n_steps)
        ])
        
    def forward(self, z):
        # z shape: (B, latent_dim)
        
        # 1. Project latent vector to initial field state
        psi_init_flat = self.projector(z)
        
        # (B, 2 * 32 * 32) -> (B, 2, 32, 32)
        psi = psi_init_flat.view(-1, 2, self.grid_size, self.grid_size)
        
        # 2. Run the differentiable physics simulation
        # The gradient will flow back through this loop
        for step in self.gpe_steps:
            psi = step(psi)
            
        # 3. Return the final, "settled" field
        return psi # Shape: (B, 2, 32, 32)

# ---------------- Fused Supersolid Neuron ----------------
class SupersolidNeuron(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), latent_dim=64, grid_size=32, 
                 physics_steps=5, output_dim=10, device='cpu'):
        super().__init__()
        self.device = device
        
        # --- 1. ENCODER ---
        # (B, 1, 28, 28) -> (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, 3, padding=1), # (B, 16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 16, 14, 14)
            nn.Conv2d(16, 32, 3, padding=1), # (B, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 32, 7, 7)
            nn.Flatten(), # (B, 32*7*7 = 1568)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim) # (B, 64)
        )
        
        # --- 2. PHYSICS PROCESSOR ---
        # (B, latent_dim) -> (B, 2, 32, 32)
        self.physics = PhysicsProcessor(latent_dim, grid_size, physics_steps, device)
        
        # --- 3. READOUT (SOMA) ---
        # (B, 2, 32, 32) -> (B, output_dim)
        self.readout = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), # (B, 16, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 16, 16, 16)
            nn.Conv2d(16, 32, 3, padding=1), # (B, 32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 32, 8, 8)
            nn.Flatten(), # (B, 32*8*8 = 2048)
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) # (B, 10)
        )
        
    def forward(self, x):
        # x shape: (B, 1, 28, 28)
        
        # 1. Encode image to latent vector
        z = self.encoder(x)
        
        # 2. Run physics simulation
        psi_final = self.physics(z)
        
        # 3. Readout the resulting field pattern
        logits = self.readout(psi_final)
        
        return logits

# ---------------- MNIST Test Harness ----------------

def get_mnist_loaders(batch_size=128):
    """Load MNIST train and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Normalize MNIST data
    ])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), torch.utils.data.DataLoader(test, batch_size=batch_size)

def apply_permutation(x, perm):
    """Apply a fixed pixel permutation to a batch of images."""
    # x shape is (B, 1, 28, 28)
    B, C, H, W = x.shape
    x_flat = x.view(B, -1) # Flatten to (B, 784)
    x_perm = x_flat[:, perm]
    return x_perm.view(B, C, H, W) # Reshape back to (B, 1, 28, 28)

def train_epoch(model, loader, optimizer, criterion, permutation, device):
    """Runs one epoch of training."""
    model.train()
    loop = tqdm(loader, desc="Training", leave=False)
    for data, target in loop:
        data, target = data.to(device), target.to(device)
        
        # Apply permutation (data shape is B, 1, 28, 28)
        perm_data = apply_permutation(data, permutation)
        
        # 1. Forward Pass & Backprop
        optimizer.zero_grad()
        outputs = model(perm_data) # Model expects (B, 1, 28, 28)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=float(loss.item()))

def test_epoch(model, loader, criterion, permutation, device):
    """Runs one epoch of testing."""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Apply permutation
            perm_data = apply_permutation(data, permutation)
            
            outputs = model(perm_data)
            test_loss += criterion(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

def run_continual_learning_test():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    # --- Task Definitions ---
    task1_perm = torch.arange(784, device='cpu').long() # Task 1: Normal MNIST (perm on CPU)
    task2_perm = torch.randperm(784, device='cpu').long() # Task 2: Permuted MNIST (perm on CPU)
    
    # --- Model Definition ---
    model = SupersolidNeuron(
        input_shape=(1, 28, 28),
        latent_dim=64,        # Size of the "thought" vector
        grid_size=32,         # Size of the physics simulation
        physics_steps=5,      # Number of physics steps to run
        output_dim=10,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() # Use CrossEntropyLoss for classification
    
    num_epochs = 5 # Use 5 epochs for a reasonable benchmark
    
    # --- Train on Task 1 ---
    print("\n--- TRAINING ON TASK 1 (Normal MNIST) ---")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch(model, train_loader, optimizer, criterion, task1_perm, device)
    
    acc1_after_1 = test_epoch(model, test_loader, criterion, task1_perm, device)
    print(f"\nFinal Task 1 Accuracy (after Task 1): {acc1_after_1:.2f}%")

    # --- Train on Task 2 ---
    print("\n--- TRAINING ON TASK 2 (Permuted MNIST) ---")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch(model, train_loader, optimizer, criterion, task2_perm, device)
    
    acc2_after_2 = test_epoch(model, test_loader, criterion, task2_perm, device)
    print(f"\nFinal Task 2 Accuracy (after Task 2): {acc2_after_2:.2f}%")
    
    # --- Test for Forgetting ---
    print("\n--- FORGETTING TEST (Testing Task 1 again) ---")
    acc1_after_2 = test_epoch(model, test_loader, criterion, task1_perm, device)
    
    print("\n--- 🔥 RESULTS 🔥 ---")
    print(f"Task 1 (Original):   {acc1_after_1:.2f}%")
    print(f"Task 2 (Permuted):   {acc2_after_2:.2f}%")
    print(f"Task 1 (After T2): {acc1_after_2:.2f}%")
    
    forgetting = acc1_after_1 - acc1_after_2
    print(f"Catastrophic Forgetting: {forgetting:.2f}%")
    
    if forgetting < 5.0 and acc1_after_1 > 90:
        print("\n🎉 SUCCESS! High accuracy and minimal forgetting observed!")
    elif forgetting < 0:
        print("\n🎉🎉🎉 POSITIVE TRANSFER! The model got *better* at Task 1!")
    else:
        print(f"\n🧠 Note: Model accuracy is high, but forgetting ({forgetting:.2f}%) is also significant.")
    
    if acc1_after_1 < 20:
        print("\n🚨 WARNING: The model failed to learn. Accuracy is at random-guess level.")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    run_continual_learning_test()

