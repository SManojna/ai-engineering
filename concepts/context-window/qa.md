# Q&A — Context Window

---

## Q1: What is a context window and why does it matter?

**Definition:**
Maximum number of tokens an LLM can process in a single forward pass.

- Inside context window → model can see and use
- Outside context window → model cannot see at all

**Why it matters:**
- Conversation history is limited by context size
- RAG documents must fit within context
- Larger context = more expensive
- Larger context = slower response

**Analogy:**
LLM is a person solving a puzzle on a desk.
Context window = size of the desk.
Pieces on the floor do not exist for that person.

---

## Q2: What happens when input exceeds the context window?

**Answer:**
Tokens that don't fit are truncated — cut off entirely.
Model has no awareness anything was cut.

**How truncation happens:**
- Most common: Earliest tokens dropped, most recent kept
- Some systems: Middle content dropped
- Advanced systems: Compress and summarize before truncating

**Analogy:**
If you add more puzzle pieces than the desk fits,
pieces fall off the edge onto the floor.
The person never knew those pieces existed.

---

## Q3: What is the difference between context window and memory?

**Quick distinction:**

| | Context Window | Memory |
|--|----------------|--------|
| What | Tokens model sees right now | Persistent external storage |
| Where | Inside the model | Outside the model |
| Duration | Temporary — cleared after conversation | Persistent across conversations |
| Who builds it | Model architecture | Engineer builds it |
| Fixed? | Yes — property of model | No — optional system |

**Analogy:**
LLM is a person working in a room.
Context window = size of their desk. Fixed. Cannot change.
Memory = filing cabinet outside the room.
Someone retrieves relevant files and places them on the desk when needed.

**Real world example:**
ChatGPT memory saves "user prefers Python" to a database.
Next conversation that fact gets retrieved and injected into the context window.
The model itself has not changed — the desk just has a new file on it.

---

## Q4: What counts towards the context window?

**Everything counts:**
- System prompt
- Conversation history
- User input
- Assistant responses
- Retrieved RAG documents
- Tool call results

**Critical implication for RAG:**
Retrieved documents eat into the space available for conversation history.
You must balance document context vs conversation history.

**Analogy:**
Everything on the desk takes up space.
Your question, assistant answers, retrieved documents, system instructions — all on the same desk.
More documents = less room for conversation history.

---

## Q5: How is context window size measured?

**Unit: Tokens**

- 1 token = approximately 0.75 words
- 1,000 tokens = approximately 750 words = approximately 1.5 pages

**Why tokens and not words:**
Different models use different tokenizers.
"tokenization" = 3 tokens in one model, 2 in another.
Token count is the universal unit the model actually processes.

**Common context window sizes:**

| Model | Context Window |
|-------|----------------|
| GPT-3.5 | 16k tokens |
| GPT-4 | 128k tokens |
| Claude 3 | 200k tokens |
| Gemini 1.5 | 1M tokens |

**Analogy:**
The desk is measured in tiles not in objects.
Each tile holds roughly one token.
A long word takes more tiles than a short word.
Desk has a fixed number of tiles regardless of what you put on it.

---

## Q6: Why can't we make the context window infinitely large?

**Three fundamental constraints:**

**1. Quadratic Scaling — most important**

Attention computes every token pair:
- 10 tokens → 10 x 10 = 100 computations
- 100 tokens → 100 x 100 = 10,000 computations
- 1000 tokens → 1000 x 1000 = 1,000,000 computations

Double context = 4x compute. Not 2x.
At infinite context window compute becomes infinite.

**2. GPU Memory**
KV cache stores K and V matrices for every token.
Larger context = more KV cache = more GPU memory consumed.
GPU memory is finite and expensive.

**3. Quality Degrades**
Models lose focus across very large contexts.
Information in the middle gets ignored.
Known as the lost in the middle problem — covered in Q14.

**Real world example:**
Claude has 200k token context window on Anthropic's cloud GPUs.
This entire conversation sits in the context window right now.
Beyond 200k Anthropic's system starts dropping oldest messages.
The constraint exists regardless of cloud or local —
it is a property of the model architecture not the machine.

**Analogy:**
Making the desk infinitely large has three problems:
1. Takes forever to look across an infinite desk
2. Costs infinite money to build
3. Person loses focus and misses things in the middle anyway

---

## Q7: How does context window size affect cost and latency?

**Latency:**
Larger context = more compute per request = longer time to generate first token = slower response for user.

**Cost:**
Larger context = more GPU time = more input tokens = higher API bill.

**The core tradeoff:**

| | Larger Context | Smaller Context |
|--|----------------|-----------------|
| Information available | More | Less |
| Answer quality | Better | Worse |
| Cost | Higher | Lower |
| Latency | Higher | Lower |

**Engineer's job:**
Balance context size against cost and latency requirements for your specific use case.

**Analogy:**
Bigger desk = more reference material available.
But bigger desk = more expensive office + longer time to find what you need.
