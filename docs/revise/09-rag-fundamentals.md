# Lecture 9 — Retrieval Augmented Generation (RAG)
**Instructor:** Prof. Koustav Rudra, Assistant Professor, AI, IIT Kharagpur

## TL;DR (5 bullets)

- RAG fetches external knowledge at inference time and supplies it to the LLM to condition generation, addressing hallucinations, knowledge cutoffs, and verifiability issues without retraining.
- Naive RAG is a stateless, linear pipeline: index → retrieve once → augment prompt → generate. Advanced RAG adds pre-retrieval (query rewriting) and post-retrieval (reranking, filtering).
- Agentic RAG uses an LLM-driven controller in a loop; plans multi-step retrieval, routes across tools (web search, calculators, multiple collections), reflects, and iterates until the question is answered.
- Chain-of-RAG (CoRAG) breaks complex multi-hop questions into sub-queries, retrieves iteratively, and refines intermediate answers (e.g., "What's the nationality of the director of the 1994 Best Picture winner?" → three retrieval steps).
- RAG failures fall into three categories: retrieval failures (wrong/incomplete docs, stale knowledge), context failures (lost in the middle, overload, irrelevant noise), and generation failures (hallucination despite retrieval, knowledge conflict, attribution errors).

## Exam-relevant concepts

### RAG Core Mechanism
- **Definition:** Retrieval Augmented Generation (RAG) is a pattern that retrieves relevant documents from an external knowledge base at inference time, concatenates them into the prompt, and conditions the LLM's generation on that retrieved context. It does NOT fine-tune the model; it is NOT a post-generation verification step; it is NOT replacing generation with search.
- **Formula / numbers / defaults:** User query → retrieve relevant docs → augmented prompt = query + documents → LLM generates grounded response.
- **When it matters (exam trap):** Q: "How does RAG improve factual accuracy?" A: By supplying external context at inference time. NOT by retraining the model on new data. NOT by verifying the answer after generation. NOT by replacing the LLM with a search engine.
- **Common confusion:** RAG is not fine-tuning. Fine-tuning changes model weights; RAG leaves weights unchanged and supplies dynamic context per query.

### Naive RAG vs Advanced RAG vs Agentic RAG
- **Definition:**
  - **Naive RAG:** Single-pass, stateless pipeline. Indexing → retrieval (one step) → augmentation → generation. No query rewriting, no reranking. Each query is independent.
  - **Advanced RAG:** Adds pre-retrieval (query manipulation, expanding domain terms) and post-retrieval (reranking top-k, filtering low-value chunks). Still a fixed pipeline, no iteration.
  - **Agentic RAG:** LLM acts as a controller/agent. Plans, selects tools (vector search engine A, web search, calculator), retrieves, reflects ("Do I have enough info?"), and may issue multiple retrieval steps in a loop until the question is answered. Multi-turn, dynamic, tool-using.
- **When it matters (exam trap):** Sample paper Q18 tests exactly this distinction. If the question describes "LLM plans its approach, decides what to investigate, takes action using tools, breaks task into smaller steps, checks whether what it found answers the question, and if not, keeps searching," that is Agentic RAG. If it describes "retrieve once, then generate," that is Naive RAG.
- **Common confusion:** Agentic RAG is NOT just adding a reranker. Reranking is post-retrieval filtering (Advanced RAG). Agentic RAG is characterized by the LLM controlling the retrieval loop, issuing multiple queries, and using external tools beyond retrieval (e.g., calculator, web search).

### Chain-of-RAG (CoRAG)
- **Definition:** CoRAG extends RAG by enabling dynamic, multi-step retrieval. For multi-hop questions (e.g., "What's the nationality of the director of the film that won Best Picture the year Titanic was released?"), it iteratively refines sub-queries based on intermediate states. Sub-query 1 → answer 1 → sub-query 2 → answer 2 → ... → final answer.
- **When it matters (exam trap):** Multi-hop questions require information from multiple documents that are not individually similar to the original query. CoRAG handles this by iterative refinement, whereas traditional single-pass RAG fails.
- **Common confusion:** CoRAG is a specific form of Agentic RAG focused on multi-hop reasoning. Not all Agentic RAG is CoRAG, but CoRAG is agentic.

### Sparse vs Dense Retrieval
- **Definition:**
  - **Sparse retrieval:** Keyword-focused. Represents documents and queries as sparse vectors (high-dimensional, mostly zeros). BM25, TF-IDF. Fast, interpretable, relies on exact term matches. Weakness: vocabulary mismatch (synonyms, paraphrasing).
  - **Dense retrieval:** Semantic-focused. Uses deep learning (BERT, T5) to encode queries and documents as dense fixed-dim vectors (embeddings). Matching is cosine similarity or dot product in embedding space. Captures semantic meaning, not just keywords. Weakness: computationally more expensive.
- **Formula / numbers / defaults:** BM25 scoring (see slide 502-516). Dense: cosine similarity between query_embedding and doc_embedding, return top-k.
- **When it matters (exam trap):** Sparse retrieval (BM25) is fast, interpretable, and strong on exact keyword matches. Dense retrieval is better for semantic/nuanced queries where synonyms and paraphrasing matter. Hybrid (BM25 + dense) is the production default.
- **Common confusion:** Sparse vectors are high-dimensional but mostly zeros (only non-zero for terms that appear). Dense vectors are fixed low-dimensional (e.g., 768, 1024) and all entries non-zero.

### BM25 (Best Matching 25)
- **Definition:** BM25 is a nonbinary probabilistic ranking function based on TF-IDF. It scores documents by summing IDF-weighted term frequencies, with scaling by term frequency saturation and document length normalization.
- **Formula / numbers / defaults:**
  - RSV (Retrieval Status Value) = sum over query terms of: IDF(t) * [tf(t,d) * (k1 + 1)] / [tf(t,d) + k1 * (1 - b + b * (Ld / Lave))]
  - k1: tuning parameter controlling term frequency scaling (default ~1.2-2.0).
  - b: tuning parameter controlling document length normalization (default ~0.75).
  - tf(t,d): term frequency in document d. Ld: length of document d. Lave: average document length in collection.
  - IDF(t) = log(N / df(t)), where N = total documents, df(t) = documents containing term t.
- **When it matters (exam trap):** BM25 is the standard sparse retrieval baseline. It penalizes very common terms (low IDF) and normalizes for document length. It does NOT capture semantic similarity, only keyword overlap.
- **Common confusion:** BM25 is not TF-IDF. BM25 applies non-linear saturation to TF and document-length scaling; TF-IDF is linear.

### TF-IDF
- **Definition:** TF-IDF (term frequency-inverse document frequency) measures word importance in a document. TF = frequency of term in document (higher TF = more aligned with topic). IDF = inverse of number of documents containing the term (terms in many docs are not significant).
- **Formula / numbers / defaults:**
  - tf(t,d) = count(t in d) / total terms in d
  - idf(t) = log(N / df(t))
  - tf-idf(t,d) = tf(t,d) * idf(t)
- **When it matters (exam trap):** TF-IDF is foundational for sparse retrieval. High TF-IDF means the term is frequent in this document but rare across the corpus. Low IDF (e.g., "the") means the term is common and not distinctive.
- **Common confusion:** TF is per-document. IDF is corpus-wide. TF-IDF is per-term per-document.

### Grounding
- **Definition:** Grounding involves providing LLMs with use-case-specific data and knowledge sources that are not part of their original training data. RAG framework acts as grounding for LLMs by supplying external context at inference time.
- **When it matters (exam trap):** Grounding is the mechanism by which RAG reduces hallucinations. The LLM is conditioned on retrieved, verifiable documents, not just its parametric memory.
- **Common confusion:** Grounding is not the same as citing sources. Grounding is supplying context to the LLM. Citation is attributing the generated answer to a source.

### LLM Limitations Addressed by RAG
- **Definition:** LLMs suffer from:
  1. Hallucinations: generating plausible but false information (fabricated citations, incompatible recipe ingredients, fake engineering parameters).
  2. Verifiability issues: no way to check if the answer is grounded in real sources (distorted historical narratives, fabricated expertise).
  3. Knowledge cutoffs: training data is static; LLMs don't know about events after the cutoff date (outdated medical info, security vulnerabilities, IPL 2026 winner).
  4. Inaccessible information: private databases, hard-to-access data, real-time data not in training.
- **When it matters (exam trap):** RAG solves all four by retrieving external, up-to-date, verifiable documents and conditioning generation on them. It does NOT solve these by retraining.
- **Common confusion:** RAG does not eliminate hallucinations completely. Even with retrieved context, LLMs can still hallucinate (generation failure mode).

## Pipelines / architectures described

### Naive RAG Pipeline
1. **Indexing (offline):** Ingest documents (PDFs, HTML, databases) → chunk into segments (e.g., 256-512 tokens, 10% overlap) → encode chunks into embeddings (dense) or build inverted index (sparse) → store in vector database / index.
2. **Retrieval (online):** User query arrives → encode query into same representation → score candidate chunks (cosine similarity for dense, BM25 for sparse) → return top-k most relevant chunks.
3. **Augmentation:** Concatenate retrieved chunks into prompt template: "Context: [chunk1, chunk2, ...] \n\n Question: [query] \n\n Answer:"
4. **Generation:** LLM processes augmented prompt and generates grounded response.

Characteristics: Stateless. Each query is independent. No query rewriting, no reranking, no iteration. Fast, simple, but limited.

### Advanced RAG Pipeline
1. **Pre-retrieval:** Query manipulation. Rewrite user query to match domain-specific terminologies, expand abbreviations, add context. E.g., "How do I reduce my electricity bill?" → "Ways to lower home energy costs; Tips to save on power consumption; How to make appliances more energy efficient."
2. **Retrieval:** Same as Naive RAG.
3. **Post-retrieval:** Reranking (cross-encoder reranks top-k results to top-5) and filtering (remove low-value or irrelevant chunks).
4. **Augmentation + Generation:** Same as Naive RAG, but with higher-quality retrieved chunks.

Characteristics: Still a fixed pipeline, but improved retrieval quality. No dynamic iteration.

### Agentic RAG Pipeline
1. **Query arrives.**
2. **Agent (LLM) plans:** Decompose the question. "To answer this, I need X and Y. Let me retrieve X first."
3. **Agent selects tool:** Vector search engine A (Collection A), Web search, Calculator, etc.
4. **Agent retrieves:** Issue retrieval query to selected tool.
5. **Agent reflects:** "Do I have enough information? Is this relevant?" If no, repeat step 2-4 with a different query or tool.
6. **Agent generates:** Once sufficient information is gathered, generate the final answer.

Characteristics: Multi-turn, dynamic, tool-using. LLM controls the loop. Can issue multiple retrieval steps. Can route across different sources (vector DB, web, SQL DB). Use cases: legal research, financial analysis (combining market data + regulatory info), multi-hop QA.

Example: "How many more people live in Paris's metropolitan area than in the city proper?"
- Plan: need two figures (city proper, metro), then subtract.
- Retrieve: city proper = 2.1M, metro = 11.3M.
- Reflect: have both numbers? Yes.
- Use calculator tool: 11.3M - 2.1M = 9.2M.
- Answer: "About 9.2 million more."

### Chain-of-RAG (CoRAG) Pipeline
1. **User asks complex multi-hop question:** "What's the nationality of the director of the film that won Best Picture the year Titanic was released?"
2. **System generates sub-query 1:** "When was Titanic released?" → retrieve → 1997.
3. **System generates sub-query 2:** "What won Best Picture for 1997 films?" → retrieve → Titanic itself (at 1998 ceremony).
4. **System generates sub-query 3:** "Who directed Titanic?" → retrieve → James Cameron.
5. **System generates sub-query 4:** "What is James Cameron's nationality?" → retrieve → Canadian.
6. **System synthesizes final answer:** Canadian.

Characteristics: Iterative refinement. Intermediate states guide next retrieval. Training uses rejection sampling for intermediate chains. Inference uses greedy decoding, best-of-N, or tree search.

### Standard RAG with Memory
- Standard RAG is stateless: each query is independent.
- RAG with memory adds a memory of past turns (past questions, answers, retrieved documents).
- Use cases: personal AI agents, conversational chatbots, customer support, tutoring.
- Example:
  - Q1: "What is the capital of France?" → A1: "Paris."
  - Q2: "What about its population?" → System recalls context (Paris), rewrites query to "What is the population of Paris?" → retrieves → A2: "About 2.05 million."

Characteristics: Conversational. Memory is used to rewrite queries and maintain context across turns.

## Retrieval pipeline details

### Document Ingestion
- Load PDFs, HTML, databases. Handle multiple formats. Tools: PyMuPDF, unstructured, Docling, Document AI.
- Challenge: PDFs, OCR'd scans, multi-column layouts, tables are the largest source of RAG quality regression. Parsing failures > embedding failures.

### Chunking Strategies
1. **Fixed-size chunking:** Sliding window (e.g., 512 tokens, 10% overlap). Fast but may split mid-sentence.
2. **Sentence / paragraph splitting:** Split at natural sentence boundaries. Better coherence; relies on good sentence detection.
3. **Semantic chunking:** Group semantically related sentences using embedding similarity shifts.
4. **Hierarchical / parent-child:** Store small chunks for retrieval (precise), retrieve larger parent chunk to send to LLM (more context).

Example: "Cats are mammals. They sleep all day. Dogs are mammals too. They love long walks."
- Fixed: "Cats are mammals. They sleep all day. Dogs are mam..." (may cut mid-word).
- Sentence: ["Cats are mammals.", "They sleep all day.", "Dogs are mammals too.", "They love long walks."]
- Semantic: ["Cats are mammals. They sleep all day.", "Dogs are mammals too. They love long walks."]

### Query Formulation Strategies
1. **Naive:** Send raw user query as-is. Works for simple, well-formed queries.
2. **HyDE (Hypothetical Document Embeddings):** Ask LLM to generate a hypothetical answer first; embed that answer and use it as the retrieval query. Improves recall when the user query is short or ambiguous.
3. **Multi-query:** Generate multiple paraphrased queries (e.g., 3-5 variants); retrieve for each; union the results. Handles ambiguity.
4. **Step-back prompting:** Abstract the query to a higher-level concept before retrieval. Example: "How do I reduce my electricity bill?" → "What factors affect household electricity usage?" Improves coverage for complex questions.

### Retrieval & Ranking
- **Sparse (BM25):** Score by term overlap and IDF weighting. Fast, interpretable. Return top-k.
- **Dense (embeddings):** Cosine similarity between query_embedding and doc_embeddings. Return top-k.
- **Hybrid:** Run both BM25 and dense, combine with Reciprocal Rank Fusion (RRF). Production default.

### Re-ranking (optional but recommended)
- After retrieval (top-k = 50-100), use a cross-encoder to rerank to top-5 or top-10.
- Cross-encoder: encodes query and doc together (not independently), computes relevance score. Much more accurate than bi-encoder, but too slow for full corpus search.
- Tools: Cohere Rerank, bge-reranker-large.

## RAG failure modes

### Retrieval Failures
1. **Wrong chunk retrieved:** Semantic similarity does not equal relevance. Vocabulary mismatch (user says "return window," doc says "refund period").
   - Example: Query: "What is the return window for online orders?" Retrieved: "Our in-store exchange policy allows swaps within 14 days." (Wrong doc.) Answer: 14 days (wrong).
2. **Incomplete retrieval:** Multi-hop questions need facts from multiple docs. Single-pass retrieval may miss one.
   - Example: Query same as above. Retrieved: "Online orders can be returned for a refund." (Cut off, missing "within 30 days.") Answer: "I don't know."
3. **Stale knowledge base:** Docs not updated, retriever returns outdated information.
   - Example: Query same. Retrieved: "Online return window: 15 days." (Last year's policy.) Answer: 15 days (wrong).

### Context Failures
1. **Lost in the middle:** LLMs ignore chunks in the middle of long contexts. Relevant information buried in middle positions is missed.
   - Example: Query: "When was the company founded?" Retrieved: [10 chunks, correct answer in chunk 5]. Answer: "I don't know."
2. **Context overload:** Too many retrieved chunks degrade generation quality. LLM gets confused or mixes up information.
   - Example: 20 chunks about history, HR policies, blog posts, FAQs. Answer: vague / mixed-up.
3. **Irrelevant context:** Noisy chunks mislead the LLM even when good chunks are also present.
   - Example: Query: "When was the company founded?" Retrieved: "Our partner company was founded in 1999." (Irrelevant.) Answer: 1999 (wrong).

### Generation Failures
1. **Hallucination despite retrieval:** LLM still generates facts not in retrieved docs.
   - Example: Query: "What is the refund window for online orders?" Retrieved: "Refundable within 30 days." Answer: "You get 30 days, and a free shipping label plus a 10% voucher." (Voucher is hallucinated.)
2. **Knowledge conflict:** Retrieved doc contradicts model's parametric memory; model may ignore retrieval and rely on parametric memory.
   - Example: Context: "30 days." Parametric memory: "most stores use 15 days." Answer: 15 days (ignores retrieval).
3. **Attribution errors:** Model cites wrong source or ignores the provided context.
   - Example: Context: "Refundable within 30 days [Policy.pdf]." Answer: "30 days. [Source: Shipping-FAQ.pdf]" (Wrong citation.)

## RAG assessment framework

### RAGAS (Retrieval-Augmented Generation Assessment Suite)
- Open-source RAG evaluation framework.
- Evaluates the entire RAG pipeline: context retrieval → answer generation.
- Four key metrics:
  1. **Answer Relevancy:** How closely the generated response matches the user's query intent. (Does the answer address the question?)
  2. **Faithfulness:** Whether the response stays true to the retrieved evidence. (Are the claims in the answer supported by the retrieved context?)
  3. **Context Precision:** How relevant the retrieved chunks are to the query. (Are the retrieved docs on-topic?)
  4. **Context Recall:** Whether all necessary information was retrieved. (Did we retrieve everything needed to answer the question?)

### RAGAS Example
- Question: "Who wrote Romeo and Juliet?"
- Contexts (retrieved): "Romeo and Juliet is a tragedy written by William Shakespeare."
- Answer (LLM generated): "Romeo and Juliet was written by William Shakespeare in 1597."
- Ground truth (reference): "William Shakespeare wrote Romeo and Juliet."

Scores:
- **Context Precision:** ~1.0 (retrieved chunk is fully on-topic).
- **Context Recall:** ~1.0 (context contains everything needed: the author).
- **Answer Relevancy:** ~0.95 (answer directly addresses "who wrote it").
- **Faithfulness:** ~0.5 (Two claims: "written by Shakespeare" (supported) and "in 1597" (NOT in context). Half the claims are grounded, so score drops.)

Key insight: Only Faithfulness catches hallucinated details ("in 1597") not present in retrieved context.

### TruLens
- Another leading RAG evaluation framework (mentioned but not detailed in lecture).

## Likely MCQ angles

1. **RAG vs fine-tuning:** RAG supplies external context at inference time; fine-tuning updates model weights. RAG does NOT retrain the model.
2. **Naive vs Agentic RAG:** Naive = single-pass, stateless. Agentic = LLM-controlled loop, multiple retrieval steps, tool-using. If the question says "plans, reflects, iterates," it's Agentic.
3. **Sparse vs Dense retrieval:** Sparse (BM25) = keyword overlap, fast, interpretable. Dense (embeddings) = semantic similarity, better for synonyms/paraphrasing. Hybrid = both.
4. **BM25 formula components:** tf (term frequency in doc), idf (inverse doc frequency), k1 (TF scaling), b (doc length normalization). BM25 is NOT linear TF-IDF.
5. **RAG failure modes:** Retrieval failures (wrong/incomplete/stale docs), context failures (lost in middle, overload, noise), generation failures (hallucination, knowledge conflict, attribution errors).
6. **Chunking strategies:** Fixed-size (fast, may split mid-sentence), sentence (coherent, needs good detection), semantic (groups related sentences), hierarchical (small for retrieval, large for LLM).
7. **Query formulation:** Naive (raw query), HyDE (generate hypothetical answer, embed it), multi-query (paraphrase multiple times), step-back (abstract to higher-level concept).
8. **RAGAS metrics:** Answer Relevancy (addresses query?), Faithfulness (grounded in context?), Context Precision (retrieved docs relevant?), Context Recall (all needed info retrieved?). Only Faithfulness catches hallucinated details not in context.
9. **Grounding:** Providing LLMs with use-case-specific external data. RAG is grounding. Reduces hallucinations by conditioning on retrieved docs.
10. **Chain-of-RAG (CoRAG):** Multi-step retrieval for multi-hop questions. Iteratively refines sub-queries based on intermediate answers. NOT single-pass.
11. **Re-ranking:** Post-retrieval step. Cross-encoder reranks top-k (e.g., 50) to top-n (e.g., 5). More accurate than bi-encoder but too slow for full corpus.
12. **Context overload vs lost in the middle:** Overload = too many chunks confuse LLM. Lost in middle = LLM ignores chunks in middle positions of long contexts.
13. **Hallucination despite retrieval:** LLM can still generate unsupported facts even when correct context is provided (generation failure).
14. **Knowledge cutoff:** LLMs don't know events after training cutoff. RAG solves this by retrieving up-to-date external documents.
15. **Stateless vs memory-augmented RAG:** Standard RAG treats each query independently. RAG with memory recalls past turns to rewrite ambiguous follow-up queries (e.g., "What about its population?" → "What is the population of Paris?").

## One-line flashcards (20)

- Q: RAG supplies context to the LLM at what stage? → A: Inference time (NOT training time).
- Q: Naive RAG is characterized by how many retrieval steps? → A: One. Single-pass, stateless.
- Q: Agentic RAG differs from Naive RAG by what capability? → A: LLM-controlled loop, multiple retrieval steps, tool selection, reflection, iteration.
- Q: What does BM25 stand for? → A: Best Matching 25 (a probabilistic ranking function for sparse retrieval).
- Q: BM25 formula includes which two tuning parameters? → A: k1 (TF scaling) and b (document length normalization).
- Q: TF-IDF: TF is per-document; IDF is? → A: Corpus-wide (inverse of number of docs containing the term).
- Q: Sparse retrieval (BM25) matches on what? → A: Keyword overlap (exact term matches).
- Q: Dense retrieval matches on what? → A: Semantic similarity (cosine/dot product of embeddings).
- Q: Hybrid retrieval combines which two methods? → A: BM25 (sparse) + dense embeddings.
- Q: Grounding in RAG means? → A: Supplying external, verifiable context to the LLM at inference time.
- Q: RAGAS Faithfulness metric measures what? → A: Whether the generated answer's claims are supported by the retrieved context.
- Q: Which RAGAS metric would catch "in 1597" hallucination if "1597" is not in context? → A: Faithfulness.
- Q: Lost in the middle means? → A: LLMs ignore chunks in the middle of long contexts.
- Q: Context overload means? → A: Too many retrieved chunks degrade generation quality (confuse the LLM).
- Q: Chain-of-RAG (CoRAG) is designed for what type of question? → A: Multi-hop questions requiring multiple retrieval steps.
- Q: HyDE (Hypothetical Document Embeddings) query strategy does what? → A: LLM generates a hypothetical answer first; that answer is embedded and used as the retrieval query.
- Q: Re-ranking is done with which model type? → A: Cross-encoder (encodes query and doc together, slower but more accurate).
- Q: RAG with memory differs from standard RAG by? → A: Keeping track of past turns to rewrite ambiguous follow-up queries.
- Q: Knowledge cutoff problem: LLM doesn't know events after? → A: Its training data cutoff date. RAG solves this by retrieving current info.
- Q: Retrieval failure mode: "stale knowledge base" means? → A: Retrieved documents are outdated; knowledge base not updated.
