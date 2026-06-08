# Q&A — GPT from Scratch

## Q1: What is a bigram model and what are its limitations?

**Intuition:** Considers two consecutive words/characters.

**Precise answer:**
Bigram model predicts the next token based on only the current token.
"Bi" refers to the pair — current and next. Not two previous tokens.

**Limitation:**
No long range context. Cannot understand dependencies beyond
the immediate previous token.
Example: In "The cat sat on the..." bigram only sees "the" 
and guesses next word. It has no idea about "cat" or "sat".

---

## Q2: What problem does self attention solve that bigram cannot?

**Intuition:** Self attention understands the full context.

**Precise answer:**
Bigram sees only 1 previous token.
Self attention lets every token look at every other token
in the sequence simultaneously.

Example: In "The bank by the river" — self attention lets
"bank" look at "river" and understand it means riverbank
not financial bank. Bigram cannot do this.

---

## Q3: What are Q, K, V and why do we need three separate projections?

**Intuition:** Three different views of the same input.

**Precise answer:**
- Q (Query): What this token is looking for
- K (Key): What this token contains
- V (Value): The actual information this token carries

Q and K have one job — find relevance between tokens.
V has a different job — carry information forward.
Two different jobs need two different representations.
That is why three projections and not two.

They are all nn.Linear layers with different random
initializations. They specialize during training because
they occupy different positions in the attention equation.

---

## Q4: What does Q @ K^T compute and why?

**Intuition:** Measures how similar each token is to every other token.

**Precise answer:**
Dot product measures similarity between two vectors.
Q @ K^T computes dot product between every Q vector
and every K vector simultaneously.
Result is a (T, T) matrix — every token scored against
every other token.
High score = highly relevant. Low score = not relevant.

---

## Q5: Why do we transpose K?

**Intuition:** Mathematical requirement.

**Precise answer:**
Q shape: (T, head_size)
K shape: (T, head_size)
Q @ K does not work — dimensions don't align.
Q @ K^T works — gives (T, T) which is exactly what we need.
The transpose is not a design decision. It is a mathematical requirement.

---

## Q6: Why do we scale by sqrt(head_size)?

**Intuition:** Some kind of normalization.

**Precise answer:**
As head_size grows larger the dot products get very large.
Large values push softmax into flat regions where gradients
become very small — vanishing gradients.
When gradients vanish training slows down or breaks entirely.
Dividing by sqrt(head_size) keeps values in a stable range
before softmax so gradients flow properly.

---

## Q7: What does softmax do to attention scores?

**Intuition:** Normalization technique.

**Precise answer:**
Converts raw scores (any real number) to probabilities
that sum to 1.
Example: [-1.2, 0.5, 2.1] → [0.04, 0.19, 0.77]
Now each score represents how much attention to pay
to each token as a proportion.

---

## Q8: Why do we multiply by V at the end?

**Intuition:** V is the answer part.

**Precise answer:**
After softmax we have attention weights — how much to
attend to each token.
Multiplying by V computes a weighted sum of information.

Example: For word "sat" with weights [0.1, 0.4, 0.5]:
output = 0.1*V("the") + 0.4*V("cat") + 0.5*V("sat")

"sat" collects 50% of its own information, 40% from "cat",
10% from "the". This is how context gets mixed in.
Q and K decide WHERE to look. V decides WHAT to take.

---

## Q9: What is the difference between encoder and decoder attention?

**Intuition:** Encoder sees everything, decoder is restricted.

**Precise answer:**
**Encoder:** Tokens attend to ALL other tokens — past and future.
Full bidirectional context.
Used when you need to understand the full input
(example: translation, understanding a sentence).

**Decoder:** Tokens attend only to PAST tokens.
Future tokens are masked.
Used for text generation — you cannot look at words
you haven't generated yet.
This preserves the autoregressive property.

---

## Q10: What is masked attention and why does a language model need it?

**Intuition:** We use the tril lower triangular matrix to block future tokens.

**Precise answer:**
Future tokens are set to negative infinity before softmax.
Negative infinity through softmax becomes zero.
Zero attention weight = that token is completely blocked.

```python
tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

Why needed: Language model generates one token at a time.
If it could see future tokens during training it would
cheat — just copy them. Masking forces it to actually
learn to predict.

---

## Q11: What is cross attention and when is it used?

**Intuition:** Some tokens get their information from a different sequence.

**Precise answer:**
In self attention Q, K, V all come from the same sequence.
In cross attention:
- Q comes from one sequence (decoder)
- K and V come from a different sequence (encoder output)

Used in encoder-decoder models like original Transformer
for translation. Decoder queries the encoded source sentence
to decide which source words to focus on while generating
each target word.

---

## Q12: Why do transformers need positional encodings?

**Intuition:** Transformer needs to know word positions.

**Precise answer:**
Self attention treats input as an unordered set.
"Cat sat on mat" and "mat on sat cat" look identical
to self attention without positional information.
Positional encodings inject position information
by adding a position vector to each token embedding.
Without them the model has no sense of word order.

---

## Q13: What is gradient descent and how does it work?

**Intuition:** Tells us which direction reduces the error.

**Precise answer:**
Gradient descent finds the minimum of the loss function
by iteratively adjusting weights.

Think of loss surface as a hilly landscape.
Every combination of weights is a point on that landscape.
Height = loss value.

Steps:
1. Start anywhere — initialize weights randomly
2. Compute gradient — which direction is downhill?
3. Take a small step downhill — subtract lr * gradient
4. Repeat until loss converges to minimum

Note: Vanishing and exploding gradients are separate problems
that can occur during gradient descent but are not
the definition of gradient descent itself.

---

## Q14: What is MSE loss and why do we use it?

**Intuition:** Penalizes large errors more.

**Precise answer:**
MSE = (1/n) * sum((y_pred - y_true)²)

Squaring the errors does two things:
1. Makes all errors positive — negative errors don't cancel positive ones
2. Penalizes large errors disproportionately — being very wrong is much worse than being slightly wrong

Used for regression because output is continuous.
Gives gradient descent a smooth surface to minimize.

---

## Q15: What is learning rate and what happens if too high or too low?

**Intuition:** Controls how fast the model learns.

**Precise answer:**
Learning rate controls the step size during weight update.
w = w - lr * gradient

**Too high:**
Steps are too large. Overshoots the minimum.
Loss bounces around or diverges instead of converging.
Not about skipping details — about missing the target entirely.

**Too low:**
Steps are tiny. Takes thousands of extra iterations.
Converges eventually but wastes compute.
Not about overfitting — about efficiency.

Overfitting and underfitting are model complexity problems.
Learning rate is an optimization speed problem.
They are separate concepts.
