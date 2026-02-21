import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MaskedSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, valid_lens): # X.shape=[batch_size, seq_len]
        def seq_mask(X, valid_len, value=-1e6):
            max_len = X.size(1)
            mask = torch.arange((max_len), dtype=torch.float32, 
                                device=X.device)[None, :]<valid_len[:, None]
            X[~mask] = value
            return X

        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
            # valid_lens is 1d so same lenght for a batch 
            # eg ([2,3] means batch 0 has valid_len 2 and batch 1 has valid_len 3)
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
            # different length for each query
                valid_lens = valid_lens.reshape(-1)
            X = seq_mask(X.reshape(-1, shape[-1]), valid_lens)
            return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.masked_softmax = MaskedSoftmax()

    def forward(self, Q, K, V, valid_lens=None):
        d = Q.shape[-1]
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), V)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, dropout=0, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hidden, bias=bias)
        self.W_k = nn.LazyLinear(num_hidden, bias=bias)
        self.W_v = nn.LazyLinear(num_hidden, bias=bias)
        self.W_o = nn.LazyLinear(num_hidden, bias=bias)
    
    # for parallel computation
    def transpose_qkv(self, X):
        # Shape of input X [batch_size, num_queries/kv, num_hidden]
        # Shape of output X [batch_size, num_queries/kv, num_heads, num_hidden/num_heads]
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)

        # Shape of output X [batch_size, num_heads, num_queries/kv, num_hidden/num_heads]
        X = X.permute(0, 2, 1, 3)
        # Shape of output X [batch_size*num_heads, num_queries/kv, num_hidden/num_heads]
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        # reverese the transpose_qkv
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, Q, K, V, valid_lens):
        # Shape of Q, K, V [batch_size, num_queries/kv, num_hidden]
        # After transposing shape of Q, K, V [batch_size * num_heads, num_queries/kv, num_hiddens / num_heads]
        Q = self.transpose_qkv(self.W_q(Q))
        K = self.transpose_qkv(self.W_k(K))
        V = self.transpose_qkv(self.W_v(V))

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)

        # Shape of output [batch_size * num_heads, num_queries, num_hiddens / num_heads]
        output = self.attention(Q, K, V, valid_lens)
        # Shape of output_concat [batch_size, num_queries, num_hidden]
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, max_len=1000):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hidden))
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        indices = torch.arange(0, num_hidden, 2, dtype=torch.float32)
        term = torch.pow(10000, indices / num_hidden)
        X = pos / term
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X

class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_hidden, ffn_num_outputs):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LazyLinear(ffn_num_hidden),
            nn.ReLU(),
            nn.LazyLinear(ffn_num_outputs),
        )
    def forward(self, X):
        return self.ffn(X)

class LayerNorm(nn.Module):
    def __init__(self, features,epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, X):
        mean = X.mean(dim=-1, keepdim=True)
        var = X.var(dim=-1, keepdim=True)
        x_norm = (X - mean) / torch.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta

# We use Post Normalization (residual connection -> layer norm) as per the original paper
# Pre Normalization is more stable tho and it's the default nowadays
class AddNorm(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.ln = LayerNorm(features)
    def forward(self, X, Y):
        return self.ln(X + Y)

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(vocab_size, embed_size))
    def forward(self, X):
        return self.embedding[X]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_hidden, ffn_num_hidden, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hidden, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hidden)
        self.ffn = PositionWiseFFN(ffn_num_hidden, num_hidden)
        self.addnorm2 = AddNorm(num_hidden)
    
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_hidden, num_heads, ffn_num_hidden,
                num_layers, dropout=0, use_bias=False):
        super().__init__()
        self.num_hidden = num_hidden
        self.embedding = Embedding(vocab_size, num_hidden)
        self.pos_encoding = PositionalEncoding(num_hidden)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(f'layer{i}',TransformerEncoderBlock(
                num_hidden, ffn_num_hidden, num_heads, dropout, use_bias
            ))
    
    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X)) * math.sqrt(self.num_hidden)
        for layer in self.layers:
            X = layer(X, valid_lens)
        return X

class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_hidden, num_heads, ffn_num_hidden, i, dropout=0):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hidden, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hidden)
        self.attention2 = MultiHeadAttention(num_hidden, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hidden)
        self.ffn = PositionWiseFFN(ffn_num_hidden, num_hidden)
        self.addnorm3 = AddNorm(num_hidden)

    def forward(self, X, state):
        enc_output, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens [batch_size, num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, 
                    dtype=torch.float32, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)  
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_output, enc_output, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hidden, num_heads, ffn_num_hidden, dropout, num_layers):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, num_hidden)
        self.pos_encoding = PositionalEncoding(num_hidden)
        self.layers = nn.Sequential()
        for layer in range(num_layers):
            self.layers.add_module(f"layer{layer}", TransformerDecoderBlock(num_hidden, num_heads,
                                    ffn_num_hidden, layer, dropout))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_output, enc_valid_lens):
        return [enc_output, enc_valid_lens, [None]*self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hidden))
        for layer in self.layers:
            X, state = layer(X, state)
        return self.dense(X), state        

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_hidden, num_heads, ffn_num_hidden, 
                num_layers, dropout, use_bias=False):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, num_hidden, num_heads,
                                        ffn_num_hidden, num_layers, use_bias)
        self.decoder = TransformerDecoder(vocab_size, num_hidden, num_heads,
                                        ffn_num_hidden, dropout, num_layers)
    
    def forward(self, enc_X, dec_X, src_valid_lens):
        enc_output = self.encoder(enc_X, src_valid_lens)
        state = self.decoder.init_state(enc_output, src_valid_lens)
        output, _ = self.decoder(dec_X, state)
        return output