# Gensemhypergraph
# Hypergraph-Based Zero-Shot Multi-Label Classification

## Overview

This project implements a hypergraph-based framework enhanced by generative and contrastive learning mechanisms to perform zero-shot multi-label text classification.

## Project Structure

- `main.py`: Main script to run the experiment.
- `hypergraph.py`: Module to construct hypergraphs.
- `transformer_encoder.py`: Transformer-based encoder for embedding nodes and hyperedges.
- `vae.py`: Variational Autoencoder for generating augmented hypergraph views.
- `utils.py`: Utility functions for data loading and label retrieval.
- `requirements.txt`: List of dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
pip install -r requirements.txt
Usage:
python main.py
Results
The script will output the total loss for each document and the predicted labels for unseen documents.
