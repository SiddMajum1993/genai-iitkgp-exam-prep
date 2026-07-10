# Lecture 9 — Retrieval Augmented Generation (RAG) Fundamentals
**Estimated read time:** 25–35 min · **Difficulty:** Core

## What this lecture is about

Retrieval Augmented Generation (RAG) is not a buzzword or a feature. It is a fundamental architectural shift in how we build with Large Language Models. At its heart, RAG solves a critical problem: LLMs are brilliant reasoners but terrible databases. They have a knowledge cutoff. They hallucinate with confidence. They cannot cite sources. Fine-tuning them on new knowledge is expensive, slow, and gets stale the moment the world changes. RAG sidesteps all of this by injecting fresh, authoritative context at inference time—WITHOUT retraining the model.

This lecture walks you through the foundations: why RAG exists, how the core mechanism works, how retrieval actually happens (both classic lexical methods and modern dense retrieval), how to transform raw documents into searchable chunks, and how to move beyond the simplest "naive RAG" pipeline into advanced and agentic variants. You will also learn how to evaluate RAG systems and understand their failure modes. By the end, you should be able to reason about retrieval pipelines from first principles and understand the trade-offs between simplicity, accuracy, and cost. This is foundational—expect exam questions that test whether you understand the difference between fine-tuning and RAG, naive vs agentic RAG, and the role of retrieval in the overall pipeline.

## Prerequisites

- You should know what an LLM is and have used one (e.g., ChatGPT).
- Basic understanding of embeddings (vectors that represent text) is helpful but not required—we will build that intuition here.
- No deep ML or NLP background required.

---

## Concept 1: The Problem RAG Solves

### The problem it solves

LLMs like GPT-4 or Claude are trained on massive corpora of text up to a certain date. After that date, they know nothing. They cannot access your company's internal documents. They cannot read the news from this morning. When you ask them something outside their training data, they often hallucinate—they generate plausible-sounding but factually incorrect answers with confidence.

Fine-tuning the model on new data is one solution, but it has serious drawbacks:
- It is expensive (requires GPU compute and labeled data).
- It bakes knowledge into the model's weights, which become stale as soon as the world changes.
- It does not allow you to cite sources or explain WHERE the knowledge came from.

RAG solves this by treating the LLM as a reasoning engine and the knowledge base as a separate, updateable resource. At inference time, when a user asks a question, the system retrieves relevant passages from the knowledge base and injects them into the LLM's prompt. The LLM generates an answer conditioned on that context. THIS IS FUNDAMENTALLY DIFFERENT FROM FINE-TUNING (which changes the model's weights) and DIFFERENT FROM GENERATE-THEN-VERIFY (which generates first and checks later).

### The intuition

Think of RAG like an open-book exam. The LLM is the student—it knows how to reason, synthesize, and write. But it does not need to memorize everything. Instead, it gets a few pages of notes (the retrieved passages) before answering each question. The quality of the answer depends on (1) whether the right pages were retrieved and (2) whether the student can reason from those notes.

Contrast this with fine-tuning, which is like asking the student to memorize everything beforehand. If the exam changes, or new material is added, you have to retrain the student from scratch.

### Worked example

**User query:** "What is the return policy for online orders?"

**Without RAG:** The LLM generates from its parametric memory. It might say "Most companies offer 30 days" or hallucinate a specific policy that sounds plausible but is wrong.

**With RAG:**
1. The system embeds the query and searches a knowledge base (e.g., company policy documents).
2. It retrieves the top-ranked passage: "Online orders can be returned for a refund within 30 days of delivery."
3. It injects that passage into the prompt:
   ```
   Context: Online orders can be returned for a refund within 30 days of delivery.
   User question: What is the return policy for online orders?
   Answer:
   ```
4. The LLM generates: "You can return online orders within 30 days of delivery for a refund."

The answer is grounded in the retrieved context, citable, and correct.

### Common misunderstandings

- **"RAG is just putting stuff in the prompt."** Technically yes, but the retrieval step is non-trivial. If you retrieve the wrong documents, the LLM will answer based on irrelevant context.
- **"RAG replaces fine-tuning."** No. Fine-tuning teaches the model domain-specific reasoning or writing style. RAG gives it access to external facts. They are complementary.
- **"If I use a long-context LLM (1M tokens), I don't need RAG."** Wrong. (a) Cost scales with input tokens—passing a 1M-token corpus on every query is expensive. (b) Retrieval naturally supports citation and access control. (c) Needle-in-a-haystack experiments show that long-context LLMs struggle to attend to all parts of very long contexts.

### Exam angle

Expect questions that test whether you understand:
- **Why RAG exists** (knowledge cutoff, hallucination, cost of fine-tuning).
- **The three-step RAG pipeline**: retrieve → augment prompt → generate.
- **Difference between RAG and fine-tuning**: RAG gives access to external knowledge at inference time without changing model weights; fine-tuning changes the weights.
- **Difference between RAG and generate-then-verify**: RAG retrieves FIRST, then generates conditioned on retrieved context. Generate-then-verify generates FIRST, then checks factuality afterward.

---

## Concept 2: The Core RAG Mechanism

### The problem it solves

How do you actually implement "retrieve → augment → generate" in a working system?

### The intuition

At inference time:
1. **User query arrives:** "Who won IPL 2026?"
2. **Retrieve relevant passages:** The system searches a knowledge base (e.g., Wikipedia, news articles) for passages related to IPL 2026. It ranks them by relevance and returns the top-k (e.g., top-5).
3. **Augment the prompt:** The system constructs a new prompt that includes both the user query and the retrieved passages:
   ```
   Context:
   [Passage 1 from knowledge base]
   [Passage 2 from knowledge base]
   ...
   User question: Who won IPL 2026?
   Answer:
   ```
4. **LLM generates:** The LLM reads the context and generates a grounded answer.

The key insight: the LLM never "knows" the answer from its training. It reads the answer from the context and generates a coherent response.

### The formula (with meaning)

The RAG pipeline can be formalized as:

**p(answer | query) = ∑ p(answer | query, doc) · p(doc | query)**

In plain English:
- For each document in the knowledge base, compute how relevant it is to the query: **p(doc | query)** (the retrieval step).
- For each relevant document, compute the probability the LLM generates the answer conditioned on that document: **p(answer | query, doc)** (the generation step).
- Sum (or max) over all documents.

In practice, we do not compute this sum exactly. We retrieve the top-k documents (k = 5 or 10), augment the prompt with those, and generate once.

### Worked example

**Naive RAG pipeline** (the simplest version):

1. **Indexing phase** (happens once):
   - Chunk the knowledge base into passages (e.g., 512-token chunks).
   - Embed each chunk into a vector using an embedding model (e.g., text-embedding-3-small).
   - Store the vectors in a vector database (e.g., ChromaDB, FAISS).

2. **Query phase** (happens per user query):
   - Embed the user query into a vector using the same embedding model.
   - Search the vector database for the top-k most similar chunks (by cosine similarity).
   - Construct the augmented prompt: [retrieved chunks] + [user query].
   - Pass the augmented prompt to the LLM.
   - Return the LLM's response.

This is called **naive RAG** because it is a straight-line pipeline with no iteration or optimization.

### Common misunderstandings

- **"The LLM retrieves the documents."** No. The retrieval step uses a separate retrieval system (vector database + embedding model). The LLM only sees the final augmented prompt.
- **"Retrieval is perfect."** No. Retrieval can fail—the top-k documents might not contain the answer, or they might be irrelevant. This is a major source of RAG failures.
- **"More retrieved chunks = better answers."** Not necessarily. If you retrieve too many chunks, you dilute the signal and overwhelm the LLM's context window. Also, irrelevant chunks can mislead the LLM.

### Exam angle

Expect questions that test:
- **The three phases of RAG:** indexing, retrieval, generation.
- **What happens at inference time:** embed query → retrieve top-k → augment prompt → generate.
- **Difference between indexing and retrieval:** Indexing happens once (offline); retrieval happens per query (online).
- **What "grounding" means:** The LLM's response is conditioned on the retrieved context, not just its parametric memory.

---

## Concept 3: Classical Retrieval — BM25

### The problem it solves

Before neural embeddings, how did search engines work? How do you rank documents by relevance to a query using just keyword matching?

### The intuition

**TF-IDF** (Term Frequency-Inverse Document Frequency) is the foundational idea:
- **Term frequency (TF):** If a term appears many times in a document, that document is probably about that term.
- **Inverse document frequency (IDF):** If a term appears in almost every document, it is not a useful signal (e.g., "the", "is"). We downweight such terms.

**BM25** is a smarter version of TF-IDF that adds two refinements:
1. **Saturation:** After a certain point, increasing term frequency does not help much. If "cat" appears 10 times vs 100 times, the document is not 10x more relevant—it just repeats itself.
2. **Document length normalization:** Longer documents naturally have higher term frequencies. BM25 normalizes by document length to avoid bias.

BM25 is still widely used in production systems (Elasticsearch, Solr) and is the default baseline for RAG retrieval.

### The formula (with meaning)

BM25 score for a document d and query q:

**score(d, q) = ∑ (over terms t in q) IDF(t) · (tf(t,d) · (k1 + 1)) / (tf(t,d) + k1 · (1 - b + b · (|d| / avgdl)))**

Where:
- **IDF(t) = log(N / df(t))**: Inverse document frequency. N = total docs, df(t) = docs containing term t. Rare terms get high IDF; common terms get low IDF.
- **tf(t,d)**: Number of times term t appears in document d.
- **k1**: Controls term frequency saturation (typical value: 1.5). Higher k1 means term frequency matters more.
- **b**: Controls document length normalization (typical value: 0.75). b=1 means full length normalization; b=0 means no normalization.
- **|d| / avgdl**: Document length relative to average document length.

### Worked example

Suppose we have three documents:
- d1: "The cat sat on the mat"
- d2: "The dog sat on the dog"
- d3: "The cat chased the dog"

Query: "cat"

**Step 1:** Compute IDF for "cat":
- df(cat) = 2 (appears in d1 and d3)
- IDF(cat) = log(3 / 2) ≈ 0.176

**Step 2:** Compute BM25 score for d1:
- tf(cat, d1) = 1
- |d1| = 6 tokens
- avgdl = (6 + 6 + 6) / 3 = 6
- Assume k1=1.5, b=0.75
- BM25(d1) = 0.176 · (1 · 2.5) / (1 + 1.5 · (1 - 0.75 + 0.75 · (6/6))) = 0.176 · 2.5 / 2.5 = 0.176

**Step 3:** Compute for d2 (cat does not appear):
- BM25(d2) = 0

**Step 4:** Compute for d3:
- tf(cat, d3) = 1
- BM25(d3) ≈ 0.176 (same length, same tf)

Ranking: d1 ≈ d3 > d2.

### Common misunderstandings

- **"BM25 is outdated."** No. BM25 is still the baseline retrieval method in production systems and often outperforms dense retrieval on keyword-heavy queries (e.g., product names, IDs).
- **"Dense retrieval (embeddings) always beats BM25."** Not true. BM25 is better for exact keyword matches and rare terms. Dense retrieval is better for semantic similarity and paraphrases.
- **"I don't need to understand BM25 if I'm using embeddings."** Wrong. Hybrid retrieval (BM25 + dense) is the production default. You need to understand both.

### Exam angle

Expect questions that test:
- **What TF-IDF measures:** Term frequency × inverse document frequency.
- **Why IDF matters:** Rare terms are more informative than common terms.
- **What BM25 adds:** Saturation and document length normalization.
- **When BM25 outperforms dense retrieval:** Exact keyword matches, rare terms, product IDs.

---

## Concept 4: Dense Retrieval and Embeddings

### The problem it solves

BM25 only matches keywords. If the query says "cat" and the document says "feline," BM25 misses it. How do we retrieve based on meaning, not just keywords?

### The intuition

**Embeddings** are dense, fixed-dimensional vectors (e.g., 768 dimensions) where geometric proximity (cosine similarity or dot product) corresponds to semantic similarity. If two sentences mean the same thing, their embeddings should be close in vector space.

Example:
- Query: "What is the capital of France?"
- Document 1: "Paris is the capital of France."
- Document 2: "The capital of Germany is Berlin."

BM25 would rank both documents equally (both contain "capital"). Dense retrieval would rank Document 1 higher because "France" and "Paris" co-occur in the embedding space.

**How embeddings are created:**
- Train a neural network (e.g., BERT) to map text to vectors such that similar meanings map to similar vectors.
- This is done via contrastive learning: push together embeddings of similar sentences and push apart embeddings of dissimilar sentences.

### The formula (with meaning)

**Cosine similarity** between query embedding q and document embedding d:

**sim(q, d) = (q · d) / (||q|| · ||d||)**

Where:
- **q · d** is the dot product (sum of element-wise products).
- **||q||** and **||d||** are the L2 norms (lengths of the vectors).

If both q and d are L2-normalized (||q|| = ||d|| = 1), then cosine similarity = dot product.

**Retrieval:** Compute sim(q, d) for all documents d, return the top-k.

### Worked example

Suppose:
- Query embedding: q = [0.7, 0.2, 0.1]
- Doc 1 embedding: d1 = [0.6, 0.3, 0.1]
- Doc 2 embedding: d2 = [0.1, 0.1, 0.8]

**Cosine similarity (assuming L2-normalized):**
- sim(q, d1) = 0.7·0.6 + 0.2·0.3 + 0.1·0.1 = 0.42 + 0.06 + 0.01 = 0.49
- sim(q, d2) = 0.7·0.1 + 0.2·0.1 + 0.1·0.8 = 0.07 + 0.02 + 0.08 = 0.17

Ranking: d1 > d2. Document 1 is semantically closer to the query.

### Common misunderstandings

- **"Embeddings are magic and always work."** No. Embeddings trained on web text may fail on domain-specific jargon (legal, medical). You may need domain-specific embedders or fine-tuning.
- **"Bigger embeddings = better retrieval."** Not always. A 1536-dimensional embedding is not necessarily better than 768-dimensional. What matters is the training data and objective.
- **"Cosine similarity is always the right metric."** Mostly true for modern embedders (they are trained to be cosine-comparable), but older models may expect dot product or Euclidean distance. Read the model card.

### Exam angle

Expect questions that test:
- **What an embedding is:** A fixed-dimensional dense vector representing text.
- **How retrieval works:** Compute similarity between query embedding and document embeddings, return top-k.
- **Difference between sparse (BM25) and dense (embeddings) retrieval:** Sparse matches keywords; dense matches meaning.
- **Why embeddings solve the synonym problem:** "cat" and "feline" have similar embeddings.

---

## Concept 5: Bi-Encoder vs Cross-Encoder

### The problem it solves

There are two ways to compute similarity between a query and a document using neural networks. Which is better, and when?

### The intuition

**Bi-encoder (dual-encoder):**
- Encode the query INDEPENDENTLY: E(query) → q
- Encode the document INDEPENDENTLY: E(doc) → d
- Compute similarity: sim(q, d) = q · d

Advantage: document embeddings can be precomputed and indexed. At query time, you only encode the query and search the index. This scales to millions of documents.

Disadvantage: the query and document never "see" each other during encoding, so the model cannot capture fine-grained interactions.

**Cross-encoder:**
- Encode the query and document JOINTLY: E([query; SEP; doc]) → score

Advantage: the model can attend to interactions between query tokens and document tokens, producing a more accurate relevance score.

Disadvantage: you must run the model N times for N documents (no precomputation). This is too slow for large-scale retrieval.

**Standard production pattern:**
1. **Bi-encoder retrieves top-100** (fast, scalable).
2. **Cross-encoder reranks top-100 to top-5** (accurate, expensive).

This is called **two-stage retrieval** or **retrieve-then-rerank**.

### Worked example

Query: "How do I reset my password?"

**Stage 1 (Bi-encoder):**
- Embed query: E(query) → q
- Search 10,000 documents in the vector database.
- Return top-100 by cosine similarity.

**Stage 2 (Cross-encoder):**
- For each of the top-100 documents, compute: score = CrossEncoder([query; SEP; doc])
- Rerank the 100 documents by score.
- Return top-5.

**Why this works:** The bi-encoder is fast but coarse. The cross-encoder is slow but precise. By narrowing from 10,000 to 100 with the bi-encoder, we make the cross-encoder tractable.

### Common misunderstandings

- **"Cross-encoders are always better."** Not at scale. They are too slow for initial retrieval over millions of documents.
- **"I can use a cross-encoder as a bi-encoder."** No. Cross-encoders produce a single score for [query; doc] pairs, not separate embeddings for query and doc.
- **"Reranking is optional."** It is optional for prototypes, but almost universal in production systems because it significantly improves top-k precision.

### Exam angle

Expect questions that test:
- **Bi-encoder:** Independent encoding, precomputable, scalable, less accurate.
- **Cross-encoder:** Joint encoding, N forward passes for N docs, slower, more accurate.
- **Two-stage retrieval:** Bi-encoder (top-100) → cross-encoder (rerank to top-5).
- **Why reranking helps:** Cross-encoder captures fine-grained query-document interactions.

---

## Concept 6: Chunking Strategies

### The problem it solves

Documents are long. Embedding models have a maximum input length (e.g., 512 tokens). LLMs have a maximum context window (e.g., 4K tokens). How do you break documents into chunks that are (1) small enough to embed, (2) semantically coherent, and (3) useful for retrieval?

### The intuition

**Why chunk?**
- Embedding models work best on short, focused passages (e.g., 300-800 tokens).
- Retrieval is more precise when each chunk represents ONE idea or fact, not multiple unrelated facts.
- LLMs generate better answers when the retrieved context is focused, not a 10-page document.

**Trade-offs:**
- **Small chunks (100-300 tokens):** High precision (each chunk is a single fact), but high fragmentation (the answer might be split across chunks).
- **Large chunks (800-1500 tokens):** More context for the LLM, but blurrier embeddings (each chunk represents multiple ideas).

**Defaults:** 300-800 tokens with 10-20% overlap is the industry standard.

### Chunking strategies

1. **Fixed-size chunking:**
   - Split every N tokens (e.g., 512 tokens).
   - Add overlap (e.g., 50 tokens) to avoid splitting mid-sentence.
   - Simple, fast, but may split in awkward places.

2. **Sentence / paragraph chunking:**
   - Split at natural boundaries (sentence or paragraph).
   - Better coherence, but requires good sentence detection.

3. **Semantic chunking:**
   - Embed each sentence.
   - Group consecutive sentences whose embeddings are similar.
   - Split when embedding similarity drops (indicates a topic shift).

4. **Hierarchical / parent-child chunking:**
   - Store small chunks for retrieval (e.g., 200 tokens).
   - Store larger "parent" chunks for LLM context (e.g., 1000 tokens).
   - Retrieve the small chunk, but pass the large chunk to the LLM.
   - Best of both worlds: precise retrieval + rich context for generation.

### Worked example

Suppose you have a document:
```
[Section 1: Introduction to cats]
Cats are mammals. They are carnivorous. They sleep 12-16 hours per day.

[Section 2: Introduction to dogs]
Dogs are also mammals. They are omnivorous. They are highly social animals.
```

**Fixed-size chunking (3 sentences per chunk):**
- Chunk 1: "Cats are mammals. They are carnivorous. They sleep 12-16 hours per day."
- Chunk 2: "Dogs are also mammals. They are omnivorous. They are highly social animals."

**Semantic chunking:**
- Chunk 1: All sentences about cats.
- Chunk 2: All sentences about dogs.

**Parent-child:**
- Small chunks (for retrieval): Each sentence is a chunk.
- Large chunks (for LLM): Entire section.
- If the query is "How long do cats sleep?", retrieve the sentence "They sleep 12-16 hours per day", but pass the entire Section 1 to the LLM.

### Common misunderstandings

- **"There is a universal best chunk size."** No. Chunk size depends on corpus structure, query patterns, and LLM context window.
- **"More overlap is always better."** Not true. Overlap increases redundancy and index size. 10-20% is typically enough.
- **"I can chunk once and forget about it."** No. Chunking interacts with the embedder, reranker, and LLM. It should be a tuning parameter.

### Exam angle

Expect questions that test:
- **Why we chunk:** Documents are too long for embedders and LLMs.
- **Trade-offs:** Small chunks = high precision, high fragmentation. Large chunks = more context, blurrier embeddings.
- **Common strategies:** Fixed-size, sentence/paragraph, semantic, hierarchical.
- **Default recommendation:** 300-800 tokens with 10-20% overlap.

---

## Concept 7: Advanced Query Transformations

### The problem it solves

User queries are often vague, short, or poorly worded. If you embed the query as-is, retrieval may fail. How do you transform the query to improve retrieval?

### Query transformation techniques

1. **Query rewriting:**
   - Use an LLM to rewrite the user query into a more explicit form.
   - Example: "How do I reset?" → "How do I reset my password?"

2. **Multi-query:**
   - Generate multiple paraphrased versions of the query.
   - Retrieve for each paraphrase.
   - Merge the results (union or intersection).
   - Handles ambiguity and improves recall.

3. **Step-back prompting:**
   - Abstract the query to a higher-level concept.
   - Example: "What is the boiling point of water at 5000 feet elevation?" → "What factors affect boiling point?"
   - Retrieve documents about the general concept, then generate a specific answer.

4. **HyDE (Hypothetical Document Embedding):**
   - Ask the LLM to generate a hypothetical answer to the query.
   - Embed the hypothetical answer (not the query).
   - Retrieve documents similar to the hypothetical answer.
   - Works because the hypothetical answer is in "document space," not "query space."

### Worked example: HyDE

Query: "How do I reduce my electricity bill?"

**Naive approach:**
- Embed the query as-is: E("How do I reduce my electricity bill?")
- Retrieve documents with similar embeddings.

**HyDE approach:**
1. Ask the LLM to generate a hypothetical answer:
   - LLM: "You can reduce your electricity bill by switching to LED bulbs, unplugging idle devices, using energy-efficient appliances, and adjusting your thermostat."
2. Embed the hypothetical answer:
   - E("You can reduce your electricity bill by switching to LED bulbs, unplugging idle devices, using energy-efficient appliances, and adjusting your thermostat.")
3. Retrieve documents similar to this embedding.

**Why it works:** The hypothetical answer is rich, detailed, and in "document space." Documents about reducing electricity bills will be written in a similar style.

### Common misunderstandings

- **"Query transformations always help."** Not necessarily. On well-formed queries, they add latency without improving retrieval. Use them when queries are short or ambiguous.
- **"HyDE is a replacement for retrieval."** No. HyDE is a pre-retrieval step. You still need to retrieve and rerank.

### Exam angle

Expect questions that test:
- **Why query transformations help:** User queries are often vague or poorly worded.
- **Multi-query:** Generate multiple paraphrases, retrieve for each, merge results.
- **Step-back prompting:** Abstract to a higher-level concept.
- **HyDE:** Generate a hypothetical answer, embed it, retrieve similar documents.

---

## Concept 8: Naive vs Advanced vs Agentic RAG

### The problem it solves

The simplest RAG pipeline (naive RAG) retrieves once and generates once. But what if the retrieved documents are not relevant? What if the query requires multiple retrieval steps? How do we iterate?

### Naive RAG

**Pipeline:** embed query → retrieve top-k → augment prompt → generate.

**Characteristics:**
- Fixed, one-shot pipeline.
- No iteration.
- No query transformation or reranking.
- Fast, simple, but brittle.

**Failure modes:**
- Retrieval fails (wrong documents).
- Query is too vague.
- Answer requires multi-hop reasoning (information from multiple documents).

### Advanced RAG

**Pipeline:** pre-retrieval → retrieve → post-retrieval → generate.

**Pre-retrieval:**
- Query rewriting, multi-query, step-back prompting, HyDE.

**Post-retrieval:**
- Reranking (cross-encoder).
- Filtering (remove low-relevance chunks).
- Context compression (summarize retrieved chunks to fit in LLM context).

**Characteristics:**
- Fixed pipeline, but with optimization steps.
- One retrieval step, but improved quality.
- Better than naive RAG, but still no iteration.

### Agentic RAG

**Pipeline:** The LLM is the CONTROLLER in a loop. It decides:
- Whether to retrieve.
- What to retrieve.
- When to stop.
- What tools to use.

**Characteristics:**
- Iterative, multi-step retrieval.
- LLM-driven decision-making.
- Can use multiple tools (retrieval, calculator, web search).
- Best for complex, multi-hop queries.

**Example:**
Query: "How many more people live in Paris's metropolitan area than in the city proper?"

**Agentic RAG steps:**
1. Plan: "I need two numbers: city proper population and metro area population. Then subtract."
2. Retrieve: Search for "Paris city proper population" → 2.1M.
3. Retrieve: Search for "Paris metropolitan area population" → 11.3M.
4. Reflect: "Do I have both numbers? Yes."
5. Use calculator tool: 11.3M - 2.1M = 9.2M.
6. Generate final answer: "About 9.2 million more people live in the Paris metropolitan area than in the city proper."

**Key insight:** Agentic RAG does not retrieve once and hope. It iterates, checks, and adapts.

### Common misunderstandings

- **"Agentic RAG is always better."** Not necessarily. It is slower, more expensive, and more complex. Use it for multi-hop queries. Use naive RAG for simple lookups.
- **"Agentic RAG is a replacement for advanced RAG."** No. Agentic RAG can INCLUDE advanced RAG techniques (reranking, query rewriting) as tools in the loop.
- **"Advanced RAG = pre-retrieval + post-retrieval."** Correct. But it is still a fixed pipeline, not iterative.

### Exam angle

**This is high-priority.** Sample paper Q18 tests the naive-vs-agentic distinction.

Expect questions that test:
- **Naive RAG:** One-shot retrieval, fixed pipeline, no iteration.
- **Advanced RAG:** Pre-retrieval (query rewriting) + post-retrieval (reranking), still one-shot.
- **Agentic RAG:** LLM-driven loop, iterative retrieval, tool use, multi-step reasoning.
- **When to use each:** Naive for simple lookups, advanced for most production systems, agentic for complex multi-hop queries.

---

## Concept 9: Chain-of-RAG (CoRAG)

### The problem it solves

Some questions require retrieving information, reasoning about it, then retrieving MORE information based on the intermediate reasoning. How do we chain retrieval steps together?

### The intuition

**Traditional RAG:** retrieve → generate.
**CoRAG:** retrieve → reason → retrieve → reason → ... → generate.

This is especially useful for **multi-hop questions** where the answer depends on a chain of facts.

### Worked example

Query: "What is the nationality of the director of the film that won Best Picture the year Titanic was released?"

**Naive RAG:**
- Retrieve documents about "Best Picture Titanic director nationality."
- Likely fails because no single document contains all this information.

**CoRAG:**
1. Sub-query: "When was Titanic released?" → 1997.
2. Sub-query: "What won Best Picture for 1997 films?" → Titanic itself (at the 1998 ceremony).
3. Sub-query: "Who directed Titanic?" → James Cameron.
4. Sub-query: "What is James Cameron's nationality?" → Canadian.
5. Final answer: "Canadian."

**Key insight:** Each retrieval step informs the next. The system does not try to retrieve everything at once.

### Common misunderstandings

- **"CoRAG is the same as agentic RAG."** They are related but distinct. CoRAG focuses on CHAINED retrieval steps. Agentic RAG focuses on LLM-driven CONTROL. CoRAG can be implemented as a form of agentic RAG.
- **"CoRAG works for all queries."** No. It is overkill for single-hop queries. Use it only for multi-hop questions.

### Exam angle

Expect questions that test:
- **What multi-hop questions are:** Questions that require chaining multiple facts.
- **How CoRAG works:** Iterative retrieval with intermediate reasoning.
- **Example:** The "Best Picture director nationality" example above.

---

## Concept 10: RAG Failure Modes

### The problem it solves

RAG can fail in multiple ways. If you understand the failure modes, you can diagnose and fix problems.

### Three categories of failure

1. **Retrieval failures:**
   - Wrong documents retrieved (semantic similarity ≠ relevance).
   - Incomplete retrieval (multi-hop questions require multiple documents).
   - Stale knowledge base (documents are outdated).

2. **Context failures:**
   - Lost in the middle (LLMs ignore chunks in the middle of long contexts).
   - Context overload (too many chunks degrade generation quality).
   - Irrelevant context (noisy chunks mislead the LLM).

3. **Generation failures:**
   - Hallucination despite retrieval (LLM generates facts not in retrieved docs).
   - Knowledge conflict (retrieved doc contradicts LLM's parametric memory).
   - Attribution errors (LLM cites the wrong source).

### Worked example: Retrieval failure

Query: "What is the return window for online orders?"

**Retrieved chunk (wrong):** "Our in-store exchange policy allows swaps within 14 days."

**LLM answer:** "You have 14 days to return online orders."

**Problem:** Semantic similarity matched "return" and "policy," but the chunk is about IN-STORE returns, not online returns.

**Fix:** Use a reranker (cross-encoder) to filter out irrelevant chunks. Or improve the chunking strategy to separate in-store and online policies.

### Common misunderstandings

- **"If retrieval similarity is high, the answer is correct."** No. High similarity does not guarantee correctness. The LLM may still hallucinate.
- **"More retrieved chunks = fewer failures."** Not true. Irrelevant chunks can mislead the LLM (context failure).

### Exam angle

Expect questions that test:
- **Three categories of failure:** Retrieval, context, generation.
- **Examples of each:** Wrong docs retrieved, lost in the middle, hallucination despite retrieval.
- **How to diagnose:** Check retrieval metrics (recall, precision), context metrics (relevance), and generation metrics (faithfulness).

---

## Concept 11: RAGAS Evaluation Framework

### The problem it solves

How do you measure whether a RAG system is working? You need metrics for retrieval quality AND generation quality.

### The four RAGAS metrics

1. **Context Precision:**
   - Are the retrieved chunks relevant to the query?
   - High precision = most retrieved chunks are on-topic.

2. **Context Recall:**
   - Do the retrieved chunks contain all the information needed to answer the query?
   - High recall = the knowledge base had the answer, and we retrieved it.

3. **Answer Relevancy:**
   - Does the generated answer address the user's query?
   - Measures alignment between answer and query intent.

4. **Faithfulness:**
   - Is the generated answer grounded in the retrieved context?
   - Low faithfulness = hallucination (answer contains claims not in the context).

### Worked example

**Query:** "Who wrote Romeo and Juliet?"

**Retrieved context:** "Romeo and Juliet is a tragedy written by William Shakespeare."

**LLM answer:** "Romeo and Juliet was written by William Shakespeare in 1597."

**Ground truth:** "William Shakespeare wrote Romeo and Juliet."

**Evaluation:**
- **Context Precision:** ~1.0 (the retrieved chunk is fully on-topic).
- **Context Recall:** ~1.0 (the context contains the needed information).
- **Answer Relevancy:** ~0.95 (the answer directly addresses "who wrote it").
- **Faithfulness:** ~0.5 (the answer claims "in 1597," which is NOT in the context—hallucinated detail).

**Key insight:** Faithfulness catches hallucinations even when the answer is mostly correct.

### Common misunderstandings

- **"High retrieval similarity = high faithfulness."** No. Faithfulness measures whether the answer is grounded in the context, not whether retrieval was good.
- **"I can skip evaluation if the answer looks good."** No. Silent hallucinations are common. You need systematic evaluation.

### Exam angle

Expect questions that test:
- **Four RAGAS metrics:** Context precision, context recall, answer relevancy, faithfulness.
- **What each measures:** Precision/recall for retrieval, relevancy/faithfulness for generation.
- **Example:** The "Romeo and Juliet" example above.

---

## Putting it together

Here is the full RAG stack:

**Indexing (offline):**
1. Chunk documents (300-800 tokens, 10-20% overlap).
2. Embed chunks (text-embedding-3-small or bge-large-en-v1.5).
3. Store embeddings in a vector database (ChromaDB, FAISS).
4. Optionally: build a BM25 index for hybrid retrieval.

**Query (online):**
1. **Pre-retrieval:** Rewrite query (LLM-based), or generate multi-query, or use HyDE.
2. **Retrieval:** Embed query, search vector DB (top-100 by cosine similarity).
3. Optionally: Hybrid retrieval (combine BM25 + dense using RRF).
4. **Post-retrieval:** Rerank (cross-encoder) to top-5.
5. **Augmentation:** Construct prompt: [retrieved chunks] + [user query].
6. **Generation:** LLM generates answer.
7. **Evaluation:** Measure context precision, context recall, answer relevancy, faithfulness.

**When to use each variant:**
- **Naive RAG:** Prototypes, simple lookups.
- **Advanced RAG:** Production systems (add pre-retrieval + post-retrieval).
- **Agentic RAG:** Complex, multi-hop queries that require iterative retrieval.
- **CoRAG:** Multi-hop questions where each retrieval step informs the next.

---

## Check yourself

1. **What is the core difference between RAG and fine-tuning?**
   - RAG injects external knowledge at inference time without changing model weights. Fine-tuning changes the model's weights to learn domain-specific reasoning or facts.

2. **Why does BM25 fail on semantic queries?**
   - BM25 matches keywords, not meaning. If the query says "cat" and the document says "feline," BM25 misses it.

3. **What is the production pattern for combining bi-encoder and cross-encoder?**
   - Bi-encoder retrieves top-100 (fast, scalable). Cross-encoder reranks to top-5 (accurate, expensive).

4. **Why do we chunk documents?**
   - Documents are too long for embedding models (max input length) and LLMs (max context window). Chunking improves retrieval precision.

5. **What is the difference between naive RAG and agentic RAG?**
   - Naive RAG: one-shot retrieval, fixed pipeline. Agentic RAG: LLM-driven loop, iterative retrieval, tool use, multi-step reasoning.

---

## Quick reference (for revision)

| Concept | Key idea |
|---------|----------|
| **RAG** | Retrieve relevant docs at inference time → augment prompt → generate grounded answer |
| **Why RAG** | Knowledge cutoff, hallucination, cost of fine-tuning |
| **Naive RAG** | Embed → retrieve top-k → augment → generate (one-shot, fixed pipeline) |
| **BM25** | Keyword-based retrieval with TF-IDF + saturation + length normalization |
| **Dense retrieval** | Embed query and docs, retrieve by cosine similarity |
| **Bi-encoder** | Independent encoding, precomputable, scalable |
| **Cross-encoder** | Joint encoding, N forward passes, more accurate |
| **Chunking** | Split docs into 300-800 tokens with 10-20% overlap |
| **Query transformations** | Rewrite, multi-query, step-back, HyDE |
| **Advanced RAG** | Pre-retrieval (query rewriting) + post-retrieval (reranking) |
| **Agentic RAG** | LLM-driven loop, iterative retrieval, tool use |
| **CoRAG** | Chained retrieval with intermediate reasoning |
| **RAG failures** | Retrieval (wrong docs), context (lost in middle), generation (hallucination) |
| **RAGAS** | Context precision/recall, answer relevancy, faithfulness |
