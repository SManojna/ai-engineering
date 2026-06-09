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

---
## Q8: What is the quadratic scaling problem in attention?

**Precise answer:**
Attention computes Q @ K^T which produces a (T, T) matrix.
Every token is scored against every other token.

T = 10   → 100 computations
T = 100  → 10,000 computations
T = 1000 → 1,000,000 computations

Double context = 4x compute. This is O(n²) — quadratic scaling.

**How it relates to context window:**
Context window defines maximum T.
Larger context window = larger T = larger attention matrix
= quadratically more compute and memory.
This is the fundamental reason context windows cannot grow freely.

**Note:**
Quadratic scaling comes from the (T, T) attention matrix.
Not from having three projections Q, K, V.
Q, K, V are fixed at three regardless of context size.

---
## Q9: Techniques to handle long contexts beyond context window

**1. Truncation**
Cut off tokens that don't fit.
- Drop oldest tokens
- Drop middle tokens
- Drop least relevant tokens
Simple but loses information permanently.

**2. Summarization/Compression**
Summarize older parts of conversation using an LLM.
Compressed summary takes fewer tokens than original.
Inject summary back into context window.
Loses some detail but preserves key information.

**3. RAG — Retrieval Augmented Generation**
Store documents outside context window in a vector database.
Retrieve only the most relevant chunks when needed.
Inject retrieved chunks into context window.
Most widely used technique in production systems.
Covered deeply in Repo 3 of THE PLAN.

**4. Sliding Window Attention**
Instead of every token attending to every other token,
each token only attends to a fixed window of nearby tokens.
Reduces O(n²) to O(n).
Loses long range dependencies.

**5. Hierarchical Processing**
Break long document into chunks.
Summarize each chunk separately.
Feed summaries into final context.
Used for very long documents like books.

**6. External Memory Systems**
Persistent database outside the model.
Retrieve relevant memories based on current query.
Inject into context window.
What you described as pulling from memory.

**Key insight for interviews:**
RAG is the most practical and widely deployed solution.
It does not increase context window size.
It makes better use of the existing context window
by only injecting what is relevant.


---
## Q11: You have a 100 page document and 8k context window. How do you handle it?

**The math:**
8k tokens ≈ 12 pages maximum in one context window.
100 pages cannot fit. Need a strategy.

**Approach 1 — RAG (most common in production)**
Chunk document into ~500 token pieces.
Why 500 tokens: small enough to be specific, large enough to be meaningful.
Too small = chunks lose meaning. Too large = irrelevant content retrieved.
Store chunks in vector database.
Retrieve only 3-5 most relevant chunks per query.
Inject retrieved chunks into context window.
Best for: Question answering over large documents.

**Approach 2 — Hierarchical Summarization**
Split into 12 page chunks — not one summary per page.
100 pages = roughly 8 chunks of 12 pages each.
Summarize each chunk separately using LLM.
All 8 summaries fit into one final context window.
Final LLM call synthesizes all summaries.
Best for: Questions needing overview of entire document.

**Approach 3 — Map Reduce**
Send each chunk to LLM with the question separately.
Collect all partial answers.
Final LLM call synthesizes all partial answers.
Best for: Finding all mentions of something across entire document.
Example: Find every reference to "climate change" across 100 pages.

**Approach 4 — Sliding Window over Chunks**
Process document in overlapping windows.
Window 1: pages 1-12
Window 2: pages 6-18
Window 3: pages 12-24
Pages 6-12 appear in both Window 1 and Window 2.
Overlap ensures boundary content gets processed twice — nothing missed.
Best for: Sequential documents where context flows across pages.

**Important note:**
Sliding window attention = model architecture change, cannot apply to existing LLMs.
Sliding window over chunks = engineering technique, works with any LLM.
These are two different things.
---

## Q12: What is lost context and how do you mitigate it?

**Definition:**
Any situation where relevant information is unavailable
or ignored when the model generates a response.

**Three types and mitigations:**

**Type 1 — Truncation Loss**
What: Context window exceeded, oldest tokens dropped.
Model forgets earlier conversation.
Mitigation:
- Summarize old conversation before dropping
- Memory: simple persistent store of key facts
  Example: save "user's name is X" or "we decided Y earlier"
  Inject these facts into every new context window
  Simpler than RAG — just key value pairs, not vector search
- Increase context window size if model allows

**Type 2 — RAG Retrieval Loss**
What: Relevant chunk exists in document but retrieval missed it.
Wrong embedding, bad chunking, or query mismatch.
Mitigation:
- Better chunking: keep related content together in one chunk
- Hybrid search: find chunks by meaning AND exact keywords
- Reranking: retrieve top 10, then score again, keep top 3
- Query expansion: rephrase question multiple ways, retrieve for each
- Retrieve more chunks: cast wider net, top 10 instead of top 3

**Type 3 — Lost in the Middle**
What: Information present in context window but model ignores it.
Models attend strongly to beginning and end.
Middle content gets relatively less attention.
Mitigation:
- Put most important information at beginning or end of context
- Reorder RAG chunks — most relevant chunk first or last
- Use smaller context windows so middle is closer to edges
- Use models trained specifically for long context tasks

**Your instinct about multiple versions:**
Similar to query expansion — run multiple versions of query,
retrieve chunks for each, take union, remove duplicates.
Increases chance of capturing relevant content.

---

## Q13: How does RAG solve the context window limitation?

**The problem:**
100 page document cannot fit in 8k context window.

**How RAG solves it:**
RAG does not increase the context window.
It makes smarter use of the existing context window.

Step 1: Chunk entire document into ~500 token pieces
Step 2: Store all chunks in vector database
Step 3: When question arrives retrieve only 3-5 relevant chunks
Step 4: Inject only those chunks into context window
Step 5: LLM answers using only relevant chunks

**Why 500 tokens per chunk:**
- Too small: chunks lose meaning
- Too large: irrelevant content retrieved alongside relevant
- 500 tokens is a common starting point, tuned per use case

**Key insight:**
RAG trades perfect recall for practical scalability.
It assumes the answer lives in a small subset of the document.
If retrieval fails the answer fails — retrieval quality is critical.

**Analogy:**
Instead of reading entire library before answering,
a smart librarian finds the 3 most relevant pages for your question.
You answer based on those 3 pages.
Fast, cheap, usually correct.

---

## Q14: What is the lost in the middle problem?

**Definition:**
When relevant information exists inside the context window
but the model ignores it because it sits in the middle.

Not about information lost during chunking or summarization.
The information is present. The model just does not attend to it.

**Why it happens:**
Attention mechanism naturally focuses more on:
- Beginning of context — primacy effect
- End of context — recency effect
- Middle receives relatively less attention weight

**Research finding:**
Models perform significantly worse when the answer
is in the middle of a long context vs beginning or end.
Performance drops as context length increases.

**Connection to needle in haystack:**
Needle in haystack = test where specific fact is hidden in long context.
Lost in the middle = reason model fails when needle is in the middle.
Same problem, different framing.

**Mitigation:**
- Put most important information at beginning or end of context
- Reorder RAG chunks — most relevant chunk first or last
- Use smaller context windows — middle is closer to edges
- Use models trained specifically for long context tasks

**Implication for RAG:**
When injecting retrieved chunks always put most relevant
chunk first or last. Never bury it in the middle.

---

## Q15: How do you decide what to put in context window when space is limited?

**Core principle:**
Context window is a limited budget. Spend it on what matters most.

**Always keep — non negotiable:**
- System prompt: defines model behavior
- User's current question: without this model cannot answer
- Top ranked RAG chunks: most relevant content by relevance score

**Cut first:**
- Old conversation history → summarize, don't keep full text
- Low relevance RAG chunks → keep top 3, drop the rest
- Redundant system prompt instructions → trim down

**Decision framework:**
- Does this change the answer? No → drop it
- Is this available in vector DB? Yes → retrieve on demand, don't keep in context
- Is this recent or old? Old → summarize or drop
- Is this large? Yes → compress first

**One critical rule:**
Most important content goes first or last — never in the middle.
Lost in the middle problem means middle content gets ignored.

**Who does this:**
AI Engineer — not the model trainer.
You build the system around the model.
You decide what goes into the context window.
Model trainer decides the context window size.
You decide what fills it.


---
Level 4 — Technical Deep Dive
Q16: What is KV cache and how does it relate to context window?
Q17: What is context compression and when would you use it?
Q18: How do models like Gemini achieve very large context windows?
Q19: What is the role of positional encoding in context window limits?
Q20: What tradeoffs come with very large context windows like 1M tokens?

Additional questions
Q21: What is the difference between context window and model knowledge?
Q22: How do you handle multi-turn conversations efficiently within a context window?
Q23: What is prompt caching and how does it reduce context window costs?
Q24: How does context window management differ between chatbot and agent systems?
Q25: If a user complains your AI app is forgetting things mid conversation what do you diagnose first?
