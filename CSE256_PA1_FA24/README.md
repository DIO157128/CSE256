# README

## How to run the code

```
python main.py --model DAN
python main.py --model SUBWORDDAN
```

## Code Structure

### DANmodels.py

In DANmodels.py, I gave a implementation of DAN model and the dataset class.

Below is the signatures for these classes.

```python
class SentimentDatasetDAN(Dataset)
class DAN(nn.Module)
class SentimentDatasetDANBPE(Dataset)
class DANBPE(nn.Module)
```

The first 2 classes are for DAN model wit word-level tokenizer and the last 2 classes are for sub-word-level. 

```python
def encode_sentences(sentences, tokenizer)
```

This method signature is for BPE tokenizer to tokenize sentences.

### main.py

In main.py, I added the training code for training DAN models.

```python
elif args.model == "DAN":
```

In the branch above, DAN models are trained in 4 different configurations. Namely, DAN with glove.6B.50d embedding, DAN with glove.6B.300d embedding, DAN with random embedding of 50 dimensions and DAN with random embedding of 300 dimensions.

```
elif args.model == "SUBWORDDAN":
```

In the branch above, DAN models are trained in 7 different configurations. Namely, DAN with embedding layer vocab size of 2000,4000,6000,8000,10000,12000,14000.

### data

This folder stores the trained BPE tokenizers and train& dev data.