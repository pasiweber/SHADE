# SHADE

SHADE (Structure-preserving High-dimensional Analysis with Density-based Exploration) is a deep clustering algorithm that combines neural network-based dimensionality reduction with density-based clustering. It trains an autoencoder with reconstruction loss and a custom d_dc loss, followed by initial clustering by using the DCTree.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SHADE
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy torch clustpy scikit-learn tqdm matplotlib
   ```

   Note: Ensure you have PyTorch installed compatible with your CUDA version if using GPU.

## Usage

```python
from shade import SHADE
import numpy as np

# Your data
X = np.random.rand(1000, 10)

# Initialize and fit SHADE
shade = SHADE(embedding_size=5, random_state=42)
shade.fit(X)

# Get cluster labels
labels = shade.labels_
print("Cluster labels:", labels)
```

## Parameters

- `batch_size`: Size of the data batches (default: 500)
- `embedding_size`: Size of the embedding (default: 10)
- `neural_network`: Custom neural network (default: None, uses FeedforwardAutoencoder)
- `optimizer_params`: Optimizer parameters (default: {"lr": 1e-3})
- `random_state`: Random state for reproducibility (default: None)
- `device`: Device to run on (default: auto-detect)

## Example

An example usage can be found in `experiments/motivation/Motivation.ipynb`. This notebook demonstrates SHADE on synthetic data, comparing it with other deep clustering methods.

To run the example:
1. Install Jupyter: `pip install jupyter`
2. Run: `jupyter lab experiments/motivation/Motivation.ipynb`

## License

BSD 3-Clause License

