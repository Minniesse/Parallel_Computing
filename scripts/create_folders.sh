#!/bin/bash

# Create required directories for the project
mkdir -p data/matrices
mkdir -p data/results
mkdir -p docs
mkdir -p showcase/demo
mkdir -p showcase/visualizations
mkdir -p showcase/analysis
mkdir -p showcase/presentation

echo "Created project directories"

# Make visualization script executable
if [ -f "visualize_benchmarks.py" ]; then
    chmod +x visualize_benchmarks.py
    echo "Made visualization script executable"
fi
