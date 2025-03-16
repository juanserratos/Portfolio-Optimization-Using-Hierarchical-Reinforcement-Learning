#!/usr/bin/env python
"""
Script to set up the project directory structure.
"""
import os
import sys

# Define the directory structure
directories = [
    "deep_rl_trading",
    "deep_rl_trading/data",
    "deep_rl_trading/models",
    "deep_rl_trading/env",
    "deep_rl_trading/training",
    "deep_rl_trading/evaluation",
    "deep_rl_trading/utils",
    "deep_rl_trading/config",
    "scripts",
    "data",
    "models",
    "results",
    "logs"
]

# Create the directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Create empty __init__.py files
init_files = [os.path.join(d, "__init__.py") for d in directories if "deep_rl_trading" in d]

for init_file in init_files:
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Auto-generated __init__.py file\n")
        print(f"Created init file: {init_file}")

print("\nDirectory structure setup complete!")
print("\nTo complete setup, run:")
print("pip install -e .")