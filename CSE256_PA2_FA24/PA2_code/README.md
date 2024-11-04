# README

## Getting Started

### Requirements

- Python 3.x
- PyTorch
- Matplotlib (for visualization)

### Running the Model

To train the model, run:

```bash
python main.py --part <part> --sanity_check
```

- Use `--part part1` for training the Transformer encoder model (classification task).
- Use `--part part2` for training the Transformer decoder model (generation task).
- Use `--part part3` for training the Transformer encoder classifier (with positional encoding).
- set  `--sanity check` if you want to do sanity check.

Additional options can be specified in `main.py` for further configurations.

## Model Architecture (transformer.py)

The main architectural components in `transformer.py` are:

- **MultiHeadSelfAttention**: Implements multi-head self-attention with optional positional encoding (AliBi).
- **TransformerEncoderLayer**: A single encoder layer containing self-attention and feedforward layers.
- **TransformerEncoder**: The encoder stack that processes input sequences.
- **FeedForwardClassifier**: A classifier used on top of the encoder for classification tasks.
- **TransformerClassifier**: Combines the encoder and classifier for text classification.
- **MaskedMultiHeadSelfAttention**: Self-attention for the decoder with causal masking.
- **TransformerDecoderLayer**: A single decoder layer with self-attention and feedforward layers.
- **TransformerDecoder**: The decoder stack for language modeling, predicting the next word in a sequence.

Each component is modular and can be customized for different tasks by adjusting the parameters in `main.py`.