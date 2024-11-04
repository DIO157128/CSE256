import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class AliBiPositionalEncoding:
    def __init__(self, num_heads):
        self.num_heads = num_heads
        # Define slope values for each head (based on empirical rules)
        self.slopes = self._get_slopes(num_heads)

    def _get_slopes(self, num_heads):
        # Method to compute slopes as per AliBi paper
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes += slopes[-1:] * (num_heads - closest_power_of_2)

        return torch.tensor(slopes).unsqueeze(1).unsqueeze(1)  # Reshape for broadcasting

    def apply_bias(self, seq_length, device='cpu'):
        # Create a distance matrix for positional bias
        position = torch.arange(seq_length, device=device)
        distance = position.view(1, -1) - position.view(-1, 1)  # Distance matrix
        distance = distance.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)

        # Apply slopes to the distance matrix to get bias
        alibi = distance * self.slopes.to(device)
        return alibi  # Shape: (1, num_heads, seq_len, seq_len)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, position):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear layers for query, key, and value projections
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.postion = position
        if position:
            self.alibi = AliBiPositionalEncoding(num_heads)
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        # Project to query, key, value vectors and reshape for multi-head attention
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute scaled dot-product attention
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if self.postion:
            alibi_bias = self.alibi.apply_bias(seq_length, device=scores.device)
            scores += alibi_bias
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, v)

        # Concatenate multiple heads and pass through final linear layer
        attention_output = attention_output.reshape(batch_size, seq_length, embed_dim)
        return self.fc_out(attention_output), attention_weights  # Returning weights for visualization


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1, position = False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads,position)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward layer with ReLU activation
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output, attn_weights = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward layer with residual connection and layer normalization
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(feedforward_output))

        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, feedforward_dim, max_seq_length, dropout=0.1, position = False):
        super(TransformerEncoder, self).__init__()

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout, position) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Embedding layer
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # Pass through each encoder layer
        attn_weights_all_layers = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all_layers.append(attn_weights)

        # Mean pooling over the sequence dimension
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        return x, attn_weights_all_layers


class FeedForwardClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FeedForwardClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, feedforward_dim, max_seq_length, hidden_dim,
                 num_classes, dropout=0.1, postion = False):
        super(TransformerClassifier, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, num_layers, feedforward_dim, max_seq_length,
                                          dropout, postion)
        self.classifier = FeedForwardClassifier(embed_dim, hidden_dim, num_classes)

    def forward(self, x):
        x, attn_weights = self.encoder(x)
        logits = self.classifier(x)
        return logits, attn_weights


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, position):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Linear projections for query, key, and value
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.position = position
        if position:
            self.alibi = AliBiPositionalEncoding(num_heads)
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        # Project to query, key, value vectors
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute scaled dot-product attention with mask
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if self.position:
            alibi_bias = self.alibi.apply_bias(seq_length, device=scores.device)
            scores += alibi_bias
        mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, v)

        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.reshape(batch_size, seq_length, embed_dim)
        return self.fc_out(attention_output), attention_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1,position = False):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MaskedMultiHeadSelfAttention(embed_dim, num_heads,position)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output, attn_weights = self.self_attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward layer with residual connection and layer normalization
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(feedforward_output))

        return x, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, feedforward_dim, max_seq_length, dropout=0.1, position = False):
        super(TransformerDecoder, self).__init__()

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, feedforward_dim, dropout,position) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)  # Final projection to vocabulary size
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        batch_size, seq_length = x.size()
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)

        # Apply token and positional embeddings
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        attn_weights_all_layers = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all_layers.append(attn_weights)

        logits = self.fc_out(x)  # Predict next word
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss = self.criterion(logits, y)
        return loss, attn_weights_all_layers
