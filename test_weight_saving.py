#!/usr/bin/env python3
"""
Test script to verify weight saving functionality.
This script creates a simple federated learning setup and checks if weights are saved correctly.
"""

import os
import time
from pathlib import Path
import shutil

# Clean up any existing weights
if Path("/tmp/local_model").exists():
    shutil.rmtree("/tmp/local_model")
    print("🧹 Cleaned up existing weights directory")

# Import p2pfl components
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.node import Node
import torch
import torch.nn as nn

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

print("🚀 Starting weight saving test...")

# Create two nodes
print("📦 Creating nodes...")

# Node 1
node1 = Node(
    model=LightningModel(SimpleMLP()),
    data=P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    addr="127.0.0.1:6666"
)

# Node 2  
node2 = Node(
    model=LightningModel(SimpleMLP()),
    data=P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    addr="127.0.0.1:6667"
)

# Start nodes
print("🏃 Starting nodes...")
node1.start()
node2.start()

# Connect node2 to node1
print("🔗 Connecting nodes...")
time.sleep(2)
node2.connect("127.0.0.1:6666")

# Start learning on node2 (which will trigger learning on both)
print("📚 Starting federated learning (2 rounds, 1 epoch per round)...")
time.sleep(2)
node2.set_start_learning(rounds=2, epochs=1)

# Wait for learning to complete
print("⏳ Waiting for learning to complete...")
time.sleep(30)  # Adjust based on your system speed

# Check if weights were saved
weights_dir = Path("/tmp/local_model")
if weights_dir.exists():
    saved_files = list(weights_dir.glob("*.pth"))
    print(f"\n✅ Weight saving successful! Found {len(saved_files)} saved weight files:")
    for file in sorted(saved_files):
        size_kb = file.stat().st_size / 1024
        print(f"   📄 {file.name} ({size_kb:.2f} KB)")
else:
    print("\n❌ Weight directory not found!")

# Stop nodes
print("\n🛑 Stopping nodes...")
node1.stop()
node2.stop()

print("\n✨ Test completed!")
print("💡 Tip: Check /tmp/local_model/ directory for saved weights")