# RNN - Recurrent Neural Networks

Custom implementations of three RNN architectures: **Vanilla RNN**, **LSTM** (Long Short-Term Memory), and **GRU** (Gated Recurrent Unit). These networks process sequential data by maintaining hidden states across time steps.

## Why RNNs?

Traditional neural networks assume inputs are independent. RNNs maintain **memory** of previous inputs through hidden states, making them ideal for:

- Text generation
- Time series prediction
- Speech recognition
- Machine translation
- Any sequential data!

## Three Architectures

### 1. Vanilla RNN (RNN_base)

The simplest recurrent architecture.

**Hidden State Update:**

$$h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)$$

**Output:**

$$o_t = h_t W_{hq} + b_q$$

Where:
- $x_t \in \mathbb{R}^{d}$ is input at time $t$
- $h_t \in \mathbb{R}^{h}$ is hidden state at time $t$
- $W_{xh} \in \mathbb{R}^{d \times h}$ maps input to hidden
- $W_{hh} \in \mathbb{R}^{h \times h}$ maps previous hidden to current hidden
- $W_{hq} \in \mathbb{R}^{h \times o}$ maps hidden to output
- $o_t \in \mathbb{R}^{o}$ is output at time $t$

**Problem**: **Vanishing/Exploding Gradients**

During backpropagation through time (BPTT), gradients get multiplied by $W_{hh}$ at each time step:

$$\frac{\partial h_t}{\partial h_{t-k}} = \prod_{i=0}^{k-1} \frac{\partial h_{t-i}}{\partial h_{t-i-1}} \approx (W_{hh})^k$$

If eigenvalues of $W_{hh}$:
- $< 1$: Gradients **vanish** (can't learn long-term dependencies)
- $> 1$: Gradients **explode** (unstable training)

**Core Implementation** (`RNN.py:4-30`):

The forward pass loops through time steps, updating the hidden state:

```python
for Xt in X:  # X: (seq_len, batch_size, input_size)
    state = activation(Xt @ W_xh + state @ W_hh + b_h)
    output = state @ W_hq + b_q
```

Simple but prone to vanishing/exploding gradients!

---

### 2. LSTM - Long Short-Term Memory

Solves vanishing gradient problem using **gates** and a **cell state**.

**Architecture Components:**

1. **Input Gate** $i_t$: What new information to store
2. **Forget Gate** $f_t$: What old information to discard
3. **Output Gate** $o_t$: What information to output
4. **Cell Candidate** $\tilde{C}_t$: New candidate values
5. **Cell State** $C_t$: Long-term memory (uninterrupted gradient flow!)
6. **Hidden State** $h_t$: Short-term memory (output)

**Equations:**

$$i_t = \sigma(x_t W_{xi} + h_{t-1} W_{hi} + b_i)$$

$$f_t = \sigma(x_t W_{xf} + h_{t-1} W_{hf} + b_f)$$

$$o_t = \sigma(x_t W_{xo} + h_{t-1} W_{ho} + b_o)$$

$$\tilde{C}_t = \tanh(x_t W_{xg} + h_{t-1} W_{hg} + b_g)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$h_t = o_t \odot \tanh(C_t)$$

Where:
- $\sigma$ is sigmoid function (gates open/close with values in [0, 1])
- $\odot$ is element-wise multiplication
- $\tanh$ squashes values to [-1, 1]

**Why LSTM Works:**

The cell state $C_t$ has an **uninterrupted gradient path**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

Since $f_t \in [0, 1]$ (controlled by forget gate), gradients don't explode, and the network learns when to "forget" vs. "remember" information.

**Core Implementation** (`RNN.py:32-79`):

```python
for Xt in X:
    H, C = state
    # Compute gates
    I = sigmoid(Xt @ W_xi + H @ W_hi + b_i)  # Input gate
    F = sigmoid(Xt @ W_xf + H @ W_hf + b_f)  # Forget gate
    O = sigmoid(Xt @ W_xo + H @ W_ho + b_o)  # Output gate
    G = tanh(Xt @ W_xg + H @ W_hg + b_g)     # Cell candidate
    
    # Update states
    C = F * C + I * G                         # Cell state
    H = O * tanh(C)                           # Hidden state
    output = H @ W_hq + b_q
```

The key is the cell state update: controlled gradient flow prevents vanishing gradients.

---

### 3. GRU - Gated Recurrent Unit

Simplified LSTM with fewer parameters (no separate cell state).

**Architecture Components:**

1. **Reset Gate** $r_t$: How much past information to forget
2. **Update Gate** $z_t$: How much past vs. new information to keep
3. **Candidate Hidden State** $\tilde{h}_t$: New candidate values
4. **Hidden State** $h_t$: Memory and output (combined)

**Equations:**

$$r_t = \sigma(x_t W_{xr} + h_{t-1} W_{hr} + b_r)$$

$$z_t = \sigma(x_t W_{xz} + h_{t-1} W_{hz} + b_z)$$

$$\tilde{h}_t = \tanh(x_t W_{xh} + (r_t \odot h_{t-1}) W_{hh} + b_h)$$

$$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$

**Key Insight:**

Update gate $z_t$ controls the **linear interpolation** between old and new:
- $z_t \approx 1$: Keep old hidden state (long-term dependency)
- $z_t \approx 0$: Use new candidate (recent information)

**GRU vs LSTM:**

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (input, forget, output) | 2 (reset, update) |
| States | 2 (hidden, cell) | 1 (hidden) |
| Parameters | More | Fewer (~25% less) |
| Speed | Slower | Faster |
| Performance | Often better | Often comparable |

**Core Implementation** (`RNN.py:81-117`):

```python
for Xt in X:
    R = sigmoid(Xt @ W_xr + state @ W_hr + b_r)      # Reset gate
    Z = sigmoid(Xt @ W_xz + state @ W_hz + b_z)      # Update gate
    H_tilde = tanh(Xt @ W_xh + (state * R) @ W_hh + b_h)  # Candidate
    state = Z * state + (1 - Z) * H_tilde            # Linear interpolation
```

Fewer parameters than LSTM, comparable performance.

---

## Example: Shakespeare Text Generation

File: `shakespeare_gen.py`

### Data Preparation

**Character-level modeling** with one-hot encoding:

1. Load text and convert to lowercase
2. Build vocabulary: sorted list of unique characters
3. Create character ↔ index mappings
4. Encode text as one-hot vectors

Each character becomes a sparse vector of size `vocab_size`.

### Training Configuration

```python
num_hiddens = 512      # Hidden state size
seq_length = 64        # Sequence length for BPTT
batch_size = 128       # Batch size
lr = 0.5               # Learning rate (SGD)
epochs = 10
```

### Model Comparison

Benchmark all three architectures on the same dataset:

- **Vanilla RNN** with tanh activation
- **LSTM** with 4 gates
- **GRU** with 2 gates

Train each for 10 epochs and compare loss curves and generated text quality.

### Text Generation

Autoregressive sampling from the trained model:

1. Start with a prefix (e.g., "the ")
2. Feed prefix through RNN to get initial hidden state
3. Sample next character from softmax distribution
4. Feed sampled character back as input
5. Repeat for desired length

This generates coherent text by maintaining context in the hidden state.

### Expected Results

**Vanilla RNN**: Struggles with long-term dependencies, generates mostly nonsense

**LSTM**: Best performance, coherent Shakespeare-like text

**GRU**: Comparable to LSTM, slightly faster training

## Key Learning Points

### 1. Sequential Processing

RNNs process sequences **one element at a time**, maintaining state:

```
h_1 → h_2 → h_3 → ... → h_T
 ↑     ↑     ↑           ↑
x_1   x_2   x_3         x_T
```

### 2. Parameter Sharing

Same weights $W_{xh}, W_{hh}$ used at every time step (like CNNs sharing kernels spatially, RNNs share temporally).

### 3. Backpropagation Through Time (BPTT)

"Unroll" the network through time and backpropagate:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

### 4. Vanishing Gradients

Vanilla RNN: $\frac{\partial h_t}{\partial h_1} \approx (W_{hh})^t$ → exponentially small

LSTM/GRU: Gates control gradient flow → stable learning

### 5. One-Hot Encoding

Character-level modeling represents each character as a sparse vector:

```
'a' → [1, 0, 0, ..., 0]
'b' → [0, 1, 0, ..., 0]
```

Vocabulary size = input/output dimension

## Files

- `RNN.py` - Three RNN architectures (Vanilla, LSTM, GRU)
- `engine.py` - Training loop, data loader, text generation
- `shakespeare_gen.py` - Shakespeare text generation + benchmarking
- `tinyshakespeare.txt` - Training corpus (~1MB Shakespeare text)
- `main.ipynb` - Jupyter notebook experiments

## Usage

```python
from RNN import RNN_lstm
from engine import Training, DataLoader, generate_text

# Prepare data (see shakespeare_gen.py)
corpus_indices, encoded_text, vocab, char_to_idx, idx_to_char = data_prep()

# Build model
model = RNN_lstm(input_size=len(vocab), hidden_size=512, output_size=len(vocab))

# Train
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
trainer = Training(model, data_loader, optimizer, epochs=10, ...)
losses = trainer.train()

# Generate text
sample = generate_text(model, 'the ', 100, vocab, char_to_idx, idx_to_char)
```

## Training Tips

- **Gradient Clipping**: Prevent exploding gradients with `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)`
- **Learning Rate**: 0.1-0.5 for character-level models
- **Hidden Size**: 256-512 for Shakespeare dataset
- **Sequence Length**: 32-128 for BPTT (longer = more context but slower)
- **Batch Size**: 128-256 for stability

---

## Dataset Credit

The `tinyshakespeare.txt` file is the **Tiny Shakespeare** dataset, a concatenation of all works by William Shakespeare (~1MB), widely used as a character-level language modeling benchmark.

> **Compiled by**: [Andrej Karpathy](https://github.com/karpathy/char-rnn) as part of the `char-rnn` project.
> The original texts are in the public domain.
