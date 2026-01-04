# Project Context

## Purpose
**ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs**

Research implementation for the paper "ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs" (arXiv:2402.11235).

### Core Goals
- Investigate the ability of graph neural networks to transfer knowledge across different graph datasets in a zero-shot manner
- Implement Text-Based Blueprints (TextBP) approach for zero-shot node classification
- Use natural language descriptions of datasets and labels to enable cross-dataset transfer without fine-tuning
- Support both standard fine-tuning and LoRA-based parameter-efficient training approaches
- Enable text-enhanced graph neural networks with label descriptions and virtual nodes

## Tech Stack

### Machine Learning & Graph Libraries
- PyTorch 2.1.2 (CUDA 11.8/12.1 support required)
- PyTorch Geometric 2.5.3 (graph neural networks)
- DGL 1.1.3 (graph deep learning)
- Transformers 4.31.0 (BERT, RoBERTa, T5, Llama 2)
- Lightning 2.1.2 (training framework)
- Sentence-Transformers 2.2.2 (text embeddings)
- PEFT 0.7.1 (LoRA implementation for parameter-efficient fine-tuning)

### Data & Scientific Computing
- NumPy 1.23.5
- Pandas 1.5.3
- SciPy 1.11.4
- Scikit-learn 1.3.2
- OGB 1.3.6 (Open Graph Benchmark datasets)

### Development & Tools
- Python 3.10 (recommended)
- Jupyter Lab 4.0.11
- pytest 7.4.4
- Weights & Biases 0.16.0 (experiment tracking)
- faiss-gpu 1.7.2 (similarity search)

### Hardware Requirements
- CUDA-enabled GPUs (required)
- Torch-Geometric modules (torch-scatter, torch-sparse, torch-spline-conv)

## Project Conventions

### Code Style
- **Naming Conventions:**
  - snake_case for Python files and functions
  - PascalCase for class names
  - ALL_CAPS for constants
  - Dataset names use PascalCase: 'Cora', 'Citeseer', 'Arxiv', etc.

- **Documentation:**
  - Extensive inline comments and docstrings
  - Type hints mostly implicit (Python 3.10 compatible)
  - Paper references in code where relevant

- **Code Organization:**
  - Utility functions abstracted (e.g., `obtain_act()`, `obtain_norm()`)
  - Configuration via command-line arguments
  - Sparse tensor operations preferred for scalability

### Architecture Patterns

**Core Components:**

1. **Model Architecture** (model.py):
   - `LinearPred` - Multi-layer MLP with configurable activation/initialization
   - `Encoder` - Graph convolutional encoder (GCN, GIN kernels with normalization)
   - `GraphCL`, `GraphInfoMax`, `GraphMAE` - Self-supervised learning models
   - `TextModel` - Text encoder wrapper supporting multiple transformer models

2. **Text-Enhanced Models** (st_model.py):
   - `TextBP` - Text-Based Blueprint using pre-trained text encoders and virtual nodes
   - `Text_Lora` - LoRA-tuned version for parameter-efficient training
   - Zero-shot evaluation via adjacency matrix normalization and multi-hop propagation

3. **Data Pipeline**:
   - `DataWrapper` - Dataset metadata wrapper
   - `kHopSubgraphDataset` - K-hop subgraph extraction with class diversity constraints
   - `kHopSubgraphDataset_Arxiv` - Optimized version with caching for large datasets

**Design Patterns:**
- Module composition - Models built from composable layers
- Argument-based configuration - Extensive argparse for hyperparameter control
- Text-graph fusion - Virtual node approach connecting text embeddings to graph structure
- Lazy evaluation - K-hop subgraph caching for memory optimization
- Zero-shot transfer - No fine-tuning on target dataset; uses learned text embeddings

**Key Hyperparameters:**
- K-hop subgraph depth: default 2
- Text encoders: SentenceBert, BERT, RoBERTa, T5, Llama 2
- R (propagation rounds): 10
- Learning rate: 0.0001
- Hidden dimension: 768
- LoRA rank: 4, alpha: 16

### Testing Strategy
- pytest 7.4.4 for unit tests
- Zero-shot evaluation without training on test datasets
- Primary metrics: Test accuracy and F1 score
- Multi-hop adjacency matrix propagation for embedding refinement
- L2 normalization of embeddings before cosine similarity matching

### Git Workflow
- Current branch: master
- Standard commit history with descriptive messages
- Research-focused workflow (experimental iterations)

## Domain Context

### Graph Neural Networks & Zero-Shot Learning
This project operates at the intersection of graph neural networks (GNNs) and zero-shot learning. Key domain concepts:

- **Zero-shot Transfer**: Ability to classify nodes on unseen graph datasets without any training on those datasets
- **Text-Based Blueprints**: Using natural language descriptions of labels/datasets to guide GNN predictions
- **Virtual Nodes**: Special nodes that connect to all nodes in the graph to propagate text-based information
- **K-hop Subgraphs**: Local neighborhoods around nodes used for training and inference
- **Graph Datasets**: Cora, Citeseer, PubMed, Arxiv, WikiCS, Facebook, Reddit, Instagram, Tech, Home

### Text-Graph Fusion
The core innovation is connecting text encoders (transformer models) to graph structure:
- Text embeddings of label descriptions serve as "blueprints" for node classification
- Virtual nodes bridge the text and graph modalities
- Cosine similarity between node embeddings and label text embeddings determines predictions

### LoRA for Parameter-Efficient Training
Low-Rank Adaptation (LoRA) reduces the number of trainable parameters while maintaining performance, making it practical to fine-tune large language models on graph data.

## Important Constraints

### Technical Constraints
- **GPU Required**: CUDA-enabled GPU is mandatory for training
- **Memory**: Large datasets (Arxiv) require careful memory management via caching
- **Python Version**: Python 3.10 recommended for compatibility
- **CUDA Version**: PyTorch 2.1.2 built for CUDA 11.8 or 12.1

### Research Constraints
- **Dataset Descriptions**: Fixed natural language descriptions embedded in code (model.py)
- **Virtual Node Handling**: Different strategies for directed vs. undirected graphs
- **K_over_2 Thresholds**: Dataset-specific values (Arxiv: 10, Citeseer: 2, others: ceil(num_classes/2))
- **Reproducibility**: Random seeds and deterministic operations needed for consistent results

### Computational Constraints
- K-hop subgraph extraction is computationally expensive for large graphs
- Text encoder inference can be slow without GPU acceleration
- Full dataset processing may require substantial disk space for cached .pt files

## External Dependencies

### Model Repositories
- **Hugging Face Model Hub**: Pre-trained BERT, RoBERTa, T5, Llama 2 models
- Models downloaded automatically on first use

### Dataset Sources
- **Open Graph Benchmark (OGB)**: Arxiv dataset
- **DGL Library**: Cora, Citeseer, PubMed datasets
- **Google Drive**: Custom datasets (Tech, Home) via gdown
- Datasets stored in `datasets/` directory as .pt files

### External Services
- **Weights & Biases**: Optional experiment tracking and logging
- **CUDA/GPU Drivers**: Required for GPU acceleration

### Key External APIs
- PyTorch Hub for model loading
- Hugging Face `transformers.AutoModel` and `AutoTokenizer`
- DGL and PyG dataset loaders
