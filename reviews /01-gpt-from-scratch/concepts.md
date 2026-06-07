## Self Attention

Three learned projections of the input: Query, Key, Value.

**Why three:**
- Q and K determine where to look
- V determines what to take from where you look

**Core snippet:**
```python
k = self.key(x)
q = self.query(x)
v = self.value(x)
scores = q @ k.transpose(-2, -1) * head_size**-0.5
weights = F.softmax(scores, dim=-1)
out = weights @ v
```

**Key insight:**
Q @ K^T measures similarity between tokens.
Softmax normalizes to probabilities.
weights @ V is the weighted sum of information collected.

---

## Bigram Model

Simplest language model. Predicts next character from current character only.
No context. No history.

**Core snippet:**
```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
logits = self.token_embedding_table(idx)
```

**Why it matters:**
Introduces three things used throughout:
- Token embeddings
- Cross entropy loss
- The generate loop

## Logits

Raw unnormalized scores output by the model before applying softmax.

**In bigram model:**
```python
logits = self.token_embedding_table(idx)  # (B, T, C)
```
Each token produces one score per vocabulary character.
Highest score = most likely next character.

**Logits → Probabilities:**
```python
probs = F.softmax(logits, dim=-1)
```

**Key insight:**
Logits can be any real number.
Softmax converts them to probabilities summing to 1.


## Self Attention — Core Concepts

### 1. Attention as Communication
Tokens communicate with each other through Q, K, V.
Each token sends a query, receives values from tokens 
whose keys match its query best.
Can be visualized as nodes in a directed graph.

```python
weights = F.softmax(q @ k.transpose(-2, -1), dim=-1)
out = weights @ v  # each token aggregates info from others
```

---

### 2. Attention Has No Notion of Space
Self attention treats input as an unordered set.
Position is not inherently captured.
Positional encodings must be added explicitly.

```python
# Without this, "cat sat" == "sat cat" to the model
x = token_embeddings + positional_embeddings
```

---

### 3. No Communication Across Batch Dimension
Each sequence in a batch is processed independently.
Tokens never attend across different sequences in the batch.

```python
# B = batch, T = time/sequence, C = channels
x.shape  # (B, T, C) — B sequences fully independent
```

---

### 4. Encoder vs Decoder Blocks
**Encoder:** Tokens attend to all other tokens — past and future.
**Decoder:** Tokens attend only to past tokens via masking.
Masking preserves the autoregressive property in generation.

```python
# Decoder masking — future tokens blocked
tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

---

### 5. Attention vs Self Attention vs Cross Attention
- **Attention:** General mechanism to weigh parts of input
- **Self Attention:** Q, K, V all come from same sequence
- **Cross Attention:** Q from one sequence, K and V from another
  (used in encoder-decoder models like original Transformer)

```python
# Self attention — same source
q, k, v = W_q(x), W_k(x), W_v(x)

# Cross attention — different sources
q = W_q(decoder_input)
k, v = W_k(encoder_output), W_v(encoder_output)
```

---

### 6. Scaled Dot Product — Why Divide by sqrt(head_size)
Without scaling dot products grow large with head size.
Large values push softmax into flat regions.
Flat softmax = very small gradients = slow or broken training.

```python
scores = q @ k.transpose(-2, -1) * head_size**-0.5
```
