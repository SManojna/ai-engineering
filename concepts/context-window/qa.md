# Q&A — Context Window

## Q1: What is a context window and why does it matter?

**Precise answer:**
Maximum number of tokens an LLM can process in a single forward pass.
Everything inside = model can see and use.
Everything outside = model cannot see at all.

**Analogy:**
Think of the LLM as a person solving a puzzle on a desk.
The context window is the size of the desk.
Only puzzle pieces on the desk can be used.
Pieces on the floor do not exist for that person right now.

**Why it matters:**
- Limits how much conversation history model remembers
- Determines how much document you can feed for RAG
- Larger context = more compute = higher cost and latency

---

## Q2: What happens when input exceeds the context window?

**Precise answer:**
Tokens that don't fit are truncated — cut off entirely.
Model has no awareness anything was cut.

**How truncation happens:**
- Most common: earliest tokens dropped, most recent kept
- Some implementations: middle content dropped
- Advanced systems: compress and summarize before truncating

**Analogy:**
Same desk analogy. If you add more puzzle pieces than the desk fits
some pieces fall off the edge onto the floor.
The person working at the desk never knew those pieces existed.

---

## Q3: What is the difference between context window and memory?

**Precise answer:**
**Context window:** How much the model can see at once.
Fixed by architecture. Temporary. Cleared after conversation ends.
A property of the model itself.

**Memory:** External storage retrieved and injected into context window.
Persistent across conversations. Optional — built by the engineer.
A system built around the model, not part of the model.

**Analogy:**
LLM is a person working in a room.
- Context window = size of their desk. Fixed. Cannot change.
- Memory = filing cabinet outside the room. Someone retrieves
  relevant files and places them on the desk when needed.

The desk size never changes.
But you can be smart about what files you bring to the desk.

**Example:**
ChatGPT memory saves "user prefers Python" to a database.
Next conversation that fact gets retrieved and placed into
the context window. The model itself has not changed.

---

## Q4: What counts towards the context window?

**Precise answer:**
Everything counts:
- System prompt
- Conversation history
- User input
- Assistant responses
- Retrieved RAG documents
- Tool call results

**Analogy:**
Everything on the desk takes up space.
Your question, the assistant's previous answers, the documents
you retrieved, the instructions you gave — all on the same desk.
RAG documents eat into the space available for conversation history.

**Critical implication:**
In RAG systems retrieved documents consume context budget.
You must balance how much document context vs conversation
history fits on the desk.

---

## Q5: How is context window size measured?

**Precise answer:**
Tokens. Not words or characters.

1 token ≈ 0.75 words in English
1000 tokens ≈ 750 words ≈ 1.5 pages of text

**Why tokens and not words:**
Different models use different tokenizers.
"tokenization" might be 3 tokens in one model, 2 in another.
Token count is the universal unit the model actually processes.
Word count is ambiguous. Token count is precise.

**Analogy:**
The desk is measured in tiles not in objects.
Each tile holds roughly one token.
A long word takes more tiles than a short word.
The desk has a fixed number of tiles regardless of what you put on it.

**Common context window sizes:**
- GPT-3.5: 16k tokens
- GPT-4: 128k tokens
- Claude 3: 200k tokens
- Gemini 1.5: 1M tokens
