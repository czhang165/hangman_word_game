# Hangman Game - Machine Learning Model

A machine learning-based Hangman game solver that combines **rule-based heuristics**, **n-gram models**, and **deep learning** to predict the most likely letters to guess next based on masked word patterns.

## Table of Contents
- [Project Overview](#project-overview)
- [Core Algorithm Design](#core-algorithm-design)
- [Files and Directories](#files-and-directories)
- [Guess Strategies](#guess-strategies)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Performance](#performance)

## Project Overview

This project develops a hybrid approach to play Hangman by predicting which letters are most likely to appear in a masked word. The system combines multiple strategies:
- **Rule-based n-gram models** with probabilistic fallback mechanisms
- **Positional encoding** using forward/backward conditional probabilities
- **Deep learning** with auxiliary feature engineering for fine-tuned predictions

### Key Design Philosophy

The project evolved through multiple iterations, balancing **computational efficiency** with **prediction accuracy**:

1. **Pure Heuristics** (baseline): Global letter frequency analysis
2. **N-gram Models** (v1-v2): Context-aware letter prediction with fuzzy matching fallback
3. **Positional Encoding** (v1-v2): Leveraging first/last letter statistics and middle-word patterns
4. **Hybrid NN Approach** (v3): Deep learning with engineered features combining n-gram and linguistic patterns

Each iteration improved accuracy by incorporating linguistic patterns (n-grams, positional information) before ultimately training a neural network that learns optimal feature weights.

## Core Algorithm Design

### 1. Rule-Based N-gram Models (Heuristic Approach)

**Strategy**: Build character frequency distributions from n-gram patterns in candidate words

**Key Components**:
- **Multi-scale n-grams** (1-7 grams): Different context window sizes capture both short-range and long-range dependencies
- **Softmax weighting**: Higher weights for 3-5 grams (empirically more predictive)
- **Conditional probability** (forward/backward): P(letter | n-gram context) captures sequential dependencies
  - Forward: Left-to-right context influence
  - Backward: Right-to-left context influence

**Trade-offs**:
- ✅ Fast, interpretable, no training required
- ❌ May miss complex linguistic patterns
- ❌ Sensitive to dictionary pruning when few candidates remain

**Fallback Mechanisms**:
1. **Fuzzy matching**: When no exact matches found, progressively relax pattern matching constraints
2. **Global n-gram fallback**: Revert to full vocabulary statistics when candidate set becomes empty

### 2. Positional Encoding (Linguistic Insight)

**Strategy**: Exploit structural properties of English words based on letter position

**Key Features**:
- **First/Last letter probabilities**: Beginning and ending letters have characteristic distributions
- **Middle-word patterns**: Letters in the middle of words follow different n-gram patterns than edges
- **Dynamic weighting**: Weights adapt based on current knowledge ratio (% of revealed letters)

**Algorithm Details**:
- Uses balanced sigmoid weights that increase with more revealed information
- Combines multiple n-gram sizes (2-5) with position-aware context patterns
- Applies weighted averaging: `0.8 * positional_prob + 0.2 * edge_prob` for first/last positions

### 3. Deep Learning Hybrid Approach

**Strategy**: Train a neural network to learn optimal feature combinations

**Feature Engineering** (15 input dimensions per letter):
1. **N-gram based frequency** (1 feature): From similarity-filtered word candidates
2. **First/Last letter probs** (2 features): Position-specific baseline distributions
3. **Masked middle patterns** (4 features): Context-specific n-gram probabilities for 3,4,4,5-gram patterns
4. **Forward/Backward conditionals** (4 features): For 2-5 gram window sizes
5. **Auxiliary features** (4 features):
   - Revealed letters (binary: 1 if already guessed, -1 if confirmed absent, 0 otherwise)
   - Vowel/consonant indicator
   - Forward/backward position indices for revealed letters

**Model Architecture**:
- Input: 15-dim feature vectors × 26 letters
- Hidden layers: 64 → 128 → 32 dimensions with GELU/LeakyReLU activations
- Batch normalization and dropout (0.25) for regularization
- Residual connection at output layer
- Output: Log-softmax probabilities for 26 letters

**Training Approach** (Behavioral Cloning):
- Dataset: ~250K English words with intermediate masking states
- Labels: Ground truth probability distribution of correct missing letters
- Dynamic sampling: Increases hard-negative samples (confirmed incorrect letters) over training iterations
- Learning rate decay: Exponential decay `lr = init_lr × 0.99^(batch_index/total_batches)`
- Early stopping: Monitors validation loss plateau

## Core Algorithm Design

## Files and Directories

### Main Code
- **`word_game_run_main.ipynb`** - Jupyter notebook containing the complete training and evaluation pipeline
  - **Data preparation**: Generating masked word samples from dictionary
  - **Algorithm implementations**: Multiple `guess_*` methods representing different strategies
  - **HangmanAPI class**: Orchestrates game playing and API interactions
  - **MyModel class**: Neural network combining rule-based features with learned weights
  - **Training loop**: Behavioral cloning with dynamic sampling and early stopping
  - **Evaluation**: Testing on practice games and calculating accuracy metrics

### Data Files

#### Training Data
- **`words_250000_train.txt`** - Vocabulary file containing 250,000 English words
  - Source for creating training examples and word candidate filtering
  - Used by both rule-based methods and neural network feature extraction

#### Datasets
- **`masked_word_dataset_v0108.pth`** - Pre-generated PyTorch dataset (custom pickle format)
  - Contains `MaskedWordDataset` object with pairs: `(masked_word_str, label_tensor)`
  - `masked_words`: List of strings with missing letters replaced by underscores (e.g., "a_ple")
  - `labels`: 26-dimensional tensors with probability/presence for each letter (A-Z)
    - 1.0 for already revealed letters
    - [0, 1] for possible missing letters (probability of being missing)
    - 0.0 for letters confirmed absent
  - Enables reproducible training without regenerating samples

### Model Checkpoints
- **`model_checkpoint_bc_*.pth`** - Saved PyTorch model states at different training iterations
  - Naming: `model_checkpoint_bc_<iteration>.pth`
  - Includes both model state dict and optimizer state for resumable training
  - Checkpoints at multiples of 500 iterations (1000, 1500, 2000, ..., 5500)
  - **Special checkpoints** marked `_ErlStp` indicate early stopping points:
    - `bc_3499_ErlStp.pth` - Early stop checkpoint
    - `bc_5324_ErlStp.pth` - Early stop checkpoint
    - `bc_5574_ErlStp.pth` - Final early stop checkpoint
  - **Final models**: `bc_5575.pth` (final iteration), `bc_5550.pth` (penultimate)

### Metrics and Logs

#### Training Metrics
- **`metrics_out_v0201.csv`** - Metrics from experimental run v0.2.01
- **`metrics_out_v0202.csv`** - Metrics from experimental run v0.2.02
- **`metrics_out_v0203.csv`** - Metrics from experimental run v0.2.03

Each CSV tracks per-batch metrics:
  - Training and validation loss
  - Letter-level prediction accuracy
  - Iteration/epoch information
  - Sample counts

#### Weights & Biases Integration
- **`wandb/`** - Weights & Biases experiment tracking logs
  - Contains run directories for each training session
  - One run per day to track experiment progression and hyperparameter changes
  - Useful for analyzing training dynamics over time

## Guess Strategies

The notebook implements **multiple guess methods** representing different approaches:

| Method | Type | Speed | Accuracy | Use Case |
|--------|------|-------|----------|----------|
| `guess()` | Rule-based | ⚡ Fast | ⭐⭐ | Baseline: global letter frequency |
| `guess_ngram()` | Rule-based | ⚡ Fast | ⭐⭐⭐ | Context-aware with exact matching |
| `guess_ngram_sftmx()` | Rule-based | ⚡ Fast | ⭐⭐⭐⭐ | Softmax-weighted n-grams + fuzzy fallback |
| `guess_with_positional_encoding()` | Heuristic | ⚡ Fast | ⭐⭐⭐⭐⭐ | Position-aware + linguistic patterns |
| `model.forward()` | Deep Learning | 🔸 Moderate | ⭐⭐⭐⭐⭐ | Learned feature fusion (production) |

**Progression**: Each version improved on the previous by:
- Expanding n-gram context windows (1→7)
- Adding fallback mechanisms (exact → fuzzy → global)
- Incorporating positional/linguistic constraints
- Learning optimal feature weights via neural network

## Model Architecture

### Input Features (15 dimensions per letter)
1. **N-gram frequency** (1): From similarity-filtered candidates
2. **First letter prob** (1): P(letter | word_start)
3. **Last letter prob** (1): P(letter | word_end)
4-7. **Middle-word patterns** (4): Context-specific n-grams at different positions
8-11. **Forward/backward conditionals** (4): Multi-scale context (2-5 grams)
12. **Revealed status** (1): {1=guessed, -1=absent, 0=unknown}
13. **Vowel indicator** (1): Binary vowel/consonant feature
14-15. **Position indices** (2): Forward/backward position of revealed letters

### Network Layers
```
Input [batch, 26, 15]
  ↓
FC1 + BatchNorm + GELU + Dropout  →  [batch, 26, 64]
  ↓
FC1_5 + BatchNorm + LeakyReLU + Dropout  →  [batch, 26, 128]
  ↓
FC2 + BatchNorm + Tanh + Dropout + Residual  →  [batch, 26, 32]
  ↓
FC3  →  [batch, 26, 1]
  ↓
LogSoftmax  →  [batch, 26] (probabilities)
```

### Key Design Choices
- **Residual connections**: Enable training deeper networks despite small hidden dimensions
- **Multiple activations**: GELU (early layers) + LeakyReLU (middle) + Tanh (late) for diverse representations
- **Per-letter processing**: Architecture processes all 26 letters independently yet jointly
- **Log-softmax output**: Enables KLDivLoss for soft label matching



## Getting Started

### Requirements
- PyTorch >= 1.9
- Python 3.7+
- NumPy, Pandas
- Jupyter Notebook
- requests (for external API)

### Installation & Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Hangman.git
cd Hangman

# Create conda environment (if using conda)
conda create -n hangman python=3.10
conda activate hangman

# Install dependencies
pip install torch numpy pandas jupyter requests

# Optionally, set up Weights & Biases for experiment tracking
pip install wandb
wandb login
```

### Usage

1. **Load and run the main notebook:**
   ```bash
   jupyter notebook word_game_run_main.ipynb
   ```

2. **Play practice games:**
   - The notebook includes cells to play against the external Hangman API
   - Test different guess strategies and compare accuracy
   - Track metrics in CSV files

3. **Load a pre-trained model:**
   ```python
   import torch
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   checkpoint = torch.load('model_checkpoint_bc_5575.pth', map_location=device)
   
   # Initialize model and load state
   from word_game_run_main import MyModel
   model = MyModel()
   model.load_state_dict(checkpoint["model_state_dict"])
   model.eval()
   ```

4. **Use the dataset:**
   ```python
   import torch
   
   # Load the pre-generated dataset
   dataset = torch.load('masked_word_dataset_v0108.pth')
   print(f"Dataset size: {len(dataset)}")
   
   # Access samples
   masked_word, label = dataset[0]
   ```

5. **Continue training from checkpoint:**
   ```python
   # The notebook includes cells that resume training from specific checkpoints
   # Just update the checkpoint path and global_batch_index in the training loop
   checkpoint_path = 'model_checkpoint_bc_5325.pth'
   global_batch_index = 5326  # Update based on checkpoint
   ```

## Performance

### Empirical Results

| Strategy | Accuracy | Notes |
|----------|----------|-------|
| Global frequency baseline | 13.6% | Simple letter frequency |
| Pure n-gram (v1) | 19.0% | 1-gram to 7-gram |
| N-gram + softmax (v2) | 19.2% | Weighted n-gram averaging |
| + Positional encoding (v1) | 30.6% | Position-aware heuristics |
| + Full positional (v2) | 38.4% | With BOW/EOW/middle patterns |
| Deep learning hybrid | ~45%* | Learned feature fusion |

*Latest model accuracy estimated from validation metrics; exact number depends on test set configuration

### Training Dynamics

The model was trained using **Behavioral Cloning**:
- **Dataset**: ~250K word vocabulary with intermediate masking states
- **Training iterations**: 5000+ batches across multiple epochs
- **Validation strategy**: Out-of-sample (OOS) words not seen during training
- **Early stopping**: Triggered when validation loss plateaus (patience=1, factor=0.11)
- **Best checkpoint**: bc_5574_ErlStp.pth (early stop) and bc_5575.pth (final)

### Key Insights

1. **N-gram scale matters**: 3-5 grams most predictive; beyond 7-grams adds noise
2. **Position encoding crucial**: First/last letter statistics improve accuracy by 10%
3. **Fallback design essential**: Fuzzy matching prevents catastrophic failures with rare words
4. **Feature engineering > architecture**: Handcrafted linguistic features more important than deep network depth
5. **Early stopping critical**: Prevents overfitting to training word patterns; final model trained for ~5500 iterations

## Notebook Structure

The `word_game_run_main.ipynb` is organized into the following sections:

### 1. **Setup & Initialization** (Cells 1-3)
- Import libraries and utility functions
- Define softmax and sigmoid weight functions
- Initialize HangmanAPI with game parameters

### 2. **API Integration** (Cells 4-18)
- HangmanAPI class implementation
- Basic guess logic and state management
- Game playing loop and result tracking

### 3. **Rule-Based Strategies** (Cells 19-30)
- `guess()`: Global frequency baseline
- `guess_ngram()`: Basic n-gram with exact matching
- `guess_ngram_sftmx()`: Softmax-weighted n-gram with fuzzy fallback
- `guess_with_positional_encoding()`: Full linguistic heuristics
- Helper functions: `fallback_fuzzy_ngram()`, `fallback_global_ngram()`
- `apply_positional_encoding()`: Positional feature calculation

### 4. **Practice Game Sessions** (Cells 31-40)
- Benchmark runs for each strategy
- Cumulative accuracy tracking
- Comparison of different heuristic approaches

### 5. **Neural Network Training Setup** (Cells 41-55)
- Dataset generation from word vocabulary
- `MaskedWordDataset` class definition
- Train/val/test split and DataLoader setup
- Auxiliary feature tensor construction
- Wandb integration for experiment tracking

### 6. **Model Definition** (Cells 56-65)
- `MyModel` class with full architecture
- Feature engineering pipeline
- Forward pass with residual connections

### 7. **Training Loop** (Cells 66-75)
- `train_model()` function with checkpoint resumption
- Loss function (KLDivLoss for soft labels)
- Learning rate scheduling
- Model evaluation on validation set

### 8. **Checkpoint Management & Inference** (Cells 76-94)
- Loading and saving checkpoints
- Inference on test sets
- Accuracy calculation
- Results visualization

## Technical Considerations

### Design Trade-offs

**Computational Efficiency vs. Accuracy**:
- Rule-based methods (n-gram, positional encoding) are <10ms per guess
- Neural network adds ~50-100ms but improves accuracy by ~7-10%
- Choice depends on application requirements (real-time game vs. batch processing)

**Generalization vs. Specificity**:
- Training on full word dataset ensures coverage of 250K+ words
- Subsampling (1/100) for training speed reduces convergence guarantees
- Early stopping prevents overfitting to training distribution

**Feature Engineered vs. End-to-End**:
- Hybrid approach preserves interpretability of linguistic features
- Neural network learns optimal weighting rather than raw pattern recognition
- Enables debugging and understanding model failures

### Known Limitations

1. **Rare words**: Performance drops on words outside training distribution
2. **Domain specificity**: Model trained on English; performance unknown on other languages
3. **Masked vocabulary**: Only tested on words in 250K word list
4. **Computational constraints**: GPU training was slow; uses subset of data

## Contributing

Contributions welcome! Potential improvements:
- Character n-gram embeddings (learned vs. rule-based)
- Transformer-based architecture for contextual understanding
- Ensemble methods combining multiple strategies
- Domain adaptation for specialized vocabularies

## Citation

If you use this project, please cite:

```bibtex
@project{hangman_ml_2025,
  title={Hangman Game Solver: A Hybrid Heuristic and Deep Learning Approach},
  author={[czhang165]},
  year={2025},
  url={https://github.com/czhang165/hangman_word_game}
}
```

## License

[MIT]

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.



