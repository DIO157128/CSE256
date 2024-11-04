import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset


# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_indexer):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.indexer = word_indexer
        self.tokenized_examples = [[self.indexer.index_of('UNK') if self.indexer.index_of(word)==-1 else self.indexer.index_of(word)  for word in example.words]for example in self.examples]
        pad_examples = [sentence[:50] + [self.indexer.index_of("PAD")] * (50- len(sentence)) for sentence in self.tokenized_examples]
        self.tokenized_examples = torch.tensor(pad_examples, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.tokenized_examples[idx], self.labels[idx]


class DAN(nn.Module):
    def __init__(self, embedding ,hidden_size, output_size,random = False):
        super(DAN, self).__init__()
        if random:
            self.embedding = embedding
            self.fc1 = nn.Linear(embedding.weight.shape[1], hidden_size)
        else:
            self.embedding = embedding.get_initialized_embedding_layer(frozen=False)
            self.fc1 = nn.Linear(embedding.get_embedding_length(), hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        embeds = self.embedding(x)  # the shape of embeds: (batch_size, seq_len, embedding_dim)
        avg_embeds = torch.mean(embeds, dim=1)  # (batch_size, embedding_dim)
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.log_softmax(out)
        return out
# Dataset class for handling sentiment analysis data
def encode_sentences(sentences, tokenizer):
    encoded_data = [tokenizer.encode(sentence).ids for sentence in sentences]
    max_len = max(len(seq) for seq in encoded_data)
    padded_data = [seq + [0] * (max_len - len(seq)) for seq in encoded_data]
    return torch.tensor(padded_data,dtype=torch.long)
class SentimentDatasetDANBPE(Dataset):
    def __init__(self, infile, tokenizer):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.tokenizer = tokenizer
        self.tokenized_examples = encode_sentences(self.sentences,self.tokenizer)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.tokenized_examples[idx], self.labels[idx]
class DANBPE(nn.Module):
    def __init__(self,vocab_size, hidden_size, output_size):
        super(DANBPE, self).__init__()
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=50)
        nn.init.uniform_(embedding.weight, a=-0.1, b=0.1)
        self.embedding = embedding
        self.fc1 = nn.Linear(embedding.weight.shape[1], hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        embeds = self.embedding(x)  # the shape of embeds: (batch_size, seq_len, embedding_dim)
        avg_embeds = torch.mean(embeds, dim=1)  # (batch_size, embedding_dim)
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.log_softmax(out)
        return out