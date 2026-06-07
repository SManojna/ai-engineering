# Terms

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
