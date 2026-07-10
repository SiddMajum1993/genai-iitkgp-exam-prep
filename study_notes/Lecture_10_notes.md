# Lecture 10 — RAG: Dense Retrieval & Semantic Search (Post-Read Material)
**Instructor:** Post-read material (no specific instructor listed; follows Lecture 9 by Prof. Koustav Rudra)

## TL;DR (5 bullets)

- Embeddings are dense, fixed-dimensional vectors (e.g., 768, 1024 dims) where geometric proximity (cosine/dot product) approximates semantic similarity. Produced by bi-encoders (query and doc encoded independently) so doc embeddings can be precomputed and indexed.
- Document processing (PDFs, OCR, multi-column, tables) is the largest source of RAG quality regression, more than embedding model choice. Tooling matters: PyMuPDF, unstructured, Docling, Document AI.
- Chunking determines retrieval granularity. Recommended: 300-800 tokens with 10-20% overlap, recursive splitting. Smaller chunks = higher precision but fragmentation; larger chunks = more context but blurrier semantics. No universal right size.
- Hybrid retrieval (BM25 + dense) with Reciprocal Rank Fusion (RRF) is the industrial default. Reranking (Cohere Rerank, bge-reranker-large) on top-50 → top-5 is standard in serious systems. Semantic search alone is rarely production-ready.
- Vector databases = storage + ANN index + metadata + persistence + (sometimes) hybrid search. ChromaDB for dev ergonomics; pgvector for SQL integration; Qdrant/Milvus for billion-scale. ANN speedup comes from quantization (int8, PQ) and index structure (HNSW, IVF).

## Exam-relevant concepts

### Embeddings (Dense Vectors)
- **Definition:** Fixed-dimensional vectors (typically 384, 768, 1024, or 1536 dimensions) produced by neural encoder models (BERT, T5, E5, BGE, text-embedding-3-small/large). All entries are non-zero (dense). Semantic similarity is approximated by cosine similarity or dot product.
- **Formula / numbers / defaults:**
  - Cosine similarity: cos(q, d) = (q · d) / (||q|| ||d||). Range: [-1, 1]. Higher = more similar.
  - Dot product: q · d. For L2-normalized vectors, dot product and cosine give same ranking.
  - L2 distance: ||q - d||. Lower = more similar. For normalized vectors, L2 distance and cosine are monotonically related.
  - Practical systems normalize vectors aggressively because dot product is the cheapest primitive.
- **When it matters (exam trap):** Dense embeddings capture semantic meaning, not just keywords. They handle synonyms, paraphrasing, and conceptual similarity. But they are computationally more expensive than sparse methods (BM25). For normalized vectors, cosine, dot product, and L2 distance give the same ranking.
- **Common confusion:** Embeddings are NOT sparse. Sparse vectors (BM25, TF-IDF) are high-dimensional with mostly zeros. Dense vectors are low-dimensional with all non-zero entries.

### Bi-encoder vs Cross-encoder
- **Definition:**
  - **Bi-encoder:** Encodes query and document independently into two vectors; similarity is computed by dot product or cosine. Fast at retrieval time because document vectors can be precomputed and indexed. Used for first-stage retrieval.
  - **Cross-encoder:** Encodes query and document together into a single representation; outputs a relevance score. Much more accurate (attends across query-doc pairs) but too slow for full corpus search. Used for reranking top-k results.
- **When it matters (exam trap):** Bi-encoders are used for retrieval (scalable, precompute doc embeddings). Cross-encoders are used for reranking (accurate, but compute query-doc pair on-the-fly, so expensive). If a question asks "which is used for initial retrieval from a large corpus?" → Bi-encoder. "Which is used to rerank top-50 to top-5?" → Cross-encoder.
- **Common confusion:** Cross-encoders are more accurate than bi-encoders but cannot be used for full-corpus retrieval because they require encoding every (query, doc) pair at query time.

### Document Processing & Chunking
- **Definition:**
  - **Document processing:** Parsing raw files (PDFs, HTML, Word, scanned images) into clean text. Handles multi-column layouts, tables, headers, footers, OCR noise.
  - **Chunking:** Splitting processed text into segments (chunks) for embedding and retrieval. Determines retrieval granularity.
- **Formula / numbers / defaults:**
  - Recommended chunk size: 300-800 tokens with 10-20% overlap.
  - Overlap: adjacent chunks share ~50-150 tokens at boundaries to avoid splitting semantic units.
  - Chunking strategies: fixed-size (simple, may split mid-sentence), sentence/paragraph (coherent boundaries), semantic (group by embedding similarity shifts), hierarchical (small for retrieval, large for LLM).
- **When it matters (exam trap):** Chunking is a critical tuning parameter, not a one-time decision. Smaller chunks = higher precision (more focused match) but risk fragmentation (answer spans multiple chunks). Larger chunks = more context but blurrier semantics (harder to match precisely). There is no universal right size; it depends on corpus structure and query patterns.
- **Common confusion:** Bigger chunks do NOT always mean better context. Larger chunks dilute the embedding signal and reduce retrieval precision. Use the smallest chunk that retains a coherent unit of meaning, with parent-document retrieval to expand context for the LLM.

### Recursive Splitting
- **Definition:** A chunking strategy that splits text hierarchically: first by paragraphs, then by sentences, then by tokens, recursively, until chunks fit within the target size. Preserves natural boundaries better than fixed-size sliding window.
- **When it matters (exam trap):** Recursive splitting is the recommended default for most corpora because it respects document structure (paragraphs, sentences) while ensuring chunks stay within size limits.
- **Common confusion:** Recursive splitting is NOT the same as fixed-size chunking. Fixed-size uses a sliding window and may cut mid-sentence. Recursive tries natural boundaries first.

### Hybrid Retrieval (BM25 + Dense)
- **Definition:** Run both sparse retrieval (BM25) and dense retrieval (embeddings) in parallel, then combine results. Combines keyword precision (BM25) with semantic understanding (dense).
- **Formula / numbers / defaults:**
  - Reciprocal Rank Fusion (RRF): For each result, RRF_score = sum over all rankers of [1 / (k + rank)], where k is a constant (default 60). Higher RRF score = better.
  - Example: Doc A is rank 1 in BM25 and rank 5 in dense. RRF(A) = 1/(60+1) + 1/(60+5) ≈ 0.0164 + 0.0154 = 0.0318.
- **When it matters (exam trap):** Hybrid retrieval is the industrial default for serious RAG systems. BM25 catches exact keyword matches; dense catches semantic matches. RRF combines them without needing to tune weights. If a question asks "what is the recommended retrieval strategy for production?" → Hybrid (BM25 + dense) with RRF.
- **Common confusion:** Hybrid does NOT mean "use BM25 OR dense." It means "use both and fuse the results." RRF is a rank-based fusion method; it does NOT require normalizing or calibrating scores from different rankers.

### Reranking
- **Definition:** A post-retrieval step. After retrieving top-k candidates (e.g., k=50-100), use a more expensive, more accurate model (cross-encoder) to rerank them to a smaller set (e.g., top-5 or top-10) that is passed to the LLM.
- **Formula / numbers / defaults:**
  - Typical pipeline: Retrieve top-50 with bi-encoder (fast) → rerank to top-5 with cross-encoder (accurate) → pass top-5 to LLM.
  - Rerankers: Cohere Rerank API, bge-reranker-large, bge-reranker-base.
- **When it matters (exam trap):** Reranking improves precision by filtering out irrelevant chunks that passed the initial retrieval. It is standard in production systems. If a question asks "how to improve retrieval quality without changing the embedding model?" → Add a reranker.
- **Common confusion:** Reranking is NOT retrieval. Retrieval is top-k selection from the full corpus (fast, approximate). Reranking is top-n selection from top-k (slow, accurate). You cannot rerank the full corpus because cross-encoders are too expensive.

### Vector Databases
- **Definition:** Specialized databases for storing, indexing, and querying high-dimensional vectors. Core components: storage (persist vectors), ANN index (fast approximate nearest neighbor search), metadata (filtering by date/tenant/ACL), and (sometimes) hybrid search (BM25 + dense in one query).
- **Formula / numbers / defaults:**
  - Common DBs: ChromaDB (in-memory, dev-friendly), pgvector (PostgreSQL extension), Qdrant (production Rust-based), Milvus (distributed, billion-scale), FAISS (library, not a DB).
  - ANN (Approximate Nearest Neighbor) index: HNSW (hierarchical navigable small world), IVF (inverted file index), PQ (product quantization).
  - Quantization: int8 (8-bit integers), PQ (product quantization). Reduces memory and speeds up similarity computation at the cost of slight accuracy loss.
- **When it matters (exam trap):** Vector databases are NOT traditional relational databases. They solve nearest-neighbor search; relational DBs solve everything else. In practice, many systems use both (e.g., pgvector for vector search + PostgreSQL for metadata/ACLs). If a question asks "do vector databases replace relational databases?" → No. They are complementary.
- **Common confusion:** FAISS is a library, not a database. It provides ANN index structures but no persistence, metadata, or server. ChromaDB, Qdrant, Milvus are full databases.

### ANN (Approximate Nearest Neighbor) Indexes
- **Definition:** Data structures that trade exact nearest-neighbor guarantees for speed. Instead of checking all N documents, ANN indexes (HNSW, IVF) prune the search space and return approximate top-k in sub-linear time.
- **Formula / numbers / defaults:**
  - Exact search: O(N) comparisons. ANN search: O(log N) or O(sqrt(N)) with high recall (e.g., 95%+).
  - HNSW (Hierarchical Navigable Small World): graph-based, very fast queries, high memory. Default for ChromaDB.
  - IVF (Inverted File): cluster-based, slower than HNSW but lower memory. Used in FAISS.
- **When it matters (exam trap):** A 100x speedup at the embedding/index layer typically comes from quantization (int8, PQ) and ANN structure, NOT from a faster embedding model. If a question asks "how to speed up retrieval?" → Use ANN index (HNSW, IVF) and quantization.
- **Common confusion:** ANN is approximate, not exact. It may miss some true top-k results, but with good tuning (e.g., recall > 95%), the loss is negligible and the speedup is massive.

### Quantization (int8, Product Quantization)
- **Definition:** Reducing precision of vector components to save memory and speed up similarity computation.
  - **int8 quantization:** Store each float32 component as an 8-bit integer. 4x memory reduction.
  - **Product Quantization (PQ):** Split vector into sub-vectors, quantize each sub-vector to a codebook entry. Much higher compression (e.g., 16x-32x).
- **When it matters (exam trap):** Quantization is the primary lever for scaling to billion-vector corpora. It reduces memory and speeds up dot-product computation. The cost is a small accuracy loss (typically <1% recall drop with int8).
- **Common confusion:** Quantization is NOT the same as dimensionality reduction. Quantization reduces precision per component (float32 → int8). Dimensionality reduction reduces the number of components (e.g., 1536 → 768 via PCA or Matryoshka).

### Matryoshka Embeddings
- **Definition:** Variable-dimensional embeddings. A single model produces embeddings where the first d dimensions (d=128, 256, 512, 768, 1024, 1536) are valid truncations. You can use 128-dim for cheap retrieval and 1536-dim for precise reranking.
- **When it matters (exam trap):** Matryoshka embeddings let you trade off cost vs. accuracy by choosing the dimensionality. If a question asks "how to reduce embedding storage without retraining?" and Matryoshka is an option, it's the answer.
- **Common confusion:** Matryoshka is NOT the same as PCA. Matryoshka models are trained to make truncated vectors meaningful. PCA is a post-hoc projection that may lose more information.

### Late-Interaction Retrievers (ColBERT)
- **Definition:** Store per-token embeddings instead of per-passage embeddings. At query time, compute token-level similarity (MaxSim: for each query token, max similarity to any doc token). Allows fine-grained matching without full cross-encoder cost.
- **When it matters (exam trap):** Late-interaction is a middle ground between bi-encoders (cheap, coarse) and cross-encoders (expensive, fine-grained). If a question describes "storing per-token embeddings and computing token-level similarity at query time," that is late-interaction (ColBERT).
- **Common confusion:** Late-interaction is NOT a cross-encoder. ColBERT still encodes query and doc separately (like bi-encoder), but it keeps per-token embeddings and interacts at token level, not passage level.

### Instruction-Tuned Embedders
- **Definition:** Embedding models trained to accept an instruction prefix (e.g., "Represent this question for retrieving relevant passages: [query]" or "Represent this document for retrieval: [doc]"). Narrows the gap to task-specific fine-tuning.
- **Formula / numbers / defaults:**
  - Models: E5-instruct, BGE-instruct, NV-Embed.
  - Usage: Prefix query with instruction, embed. Prefix doc with instruction, embed. Compute similarity as usual.
- **When it matters (exam trap):** Instruction-tuned embedders are more flexible than generic embedders. They can be adapted to different tasks (QA retrieval, code search, biomedical search) without fine-tuning, just by changing the instruction.
- **Common confusion:** Instruction-tuning is NOT the same as fine-tuning. Instruction-tuning is done once by the model provider. You use it by changing the prompt, not by retraining.

### BGE (BAAI General Embedding) Family
- **Definition:** Open-source embedding models from BAAI (Beijing Academy of AI). bge-large-en-v1.5 and bge-base-en-v1.5 are current open SOTA for dense retrieval. bge-reranker-large is a popular cross-encoder for reranking.
- **When it matters (exam trap):** BGE models are the recommended open-source default for production. If a question asks "which open-source embedding model?" and BGE is an option, it's likely the answer.
- **Common confusion:** BGE-large is a bi-encoder (for retrieval). bge-reranker-large is a cross-encoder (for reranking). They are different models for different stages.

### MTEB (Massive Text Embedding Benchmark)
- **Definition:** Aggregates 50+ embedding tasks (retrieval, classification, clustering, semantic similarity, etc.) into a single leaderboard. The de facto ranking for embedding models. Hosted at huggingface.co/spaces/mteb/leaderboard.
- **When it matters (exam trap):** MTEB averages across 50+ tasks. Your use case may correlate with only a few. A model scoring 65 on MTEB is not necessarily better than one scoring 64 for YOUR specific domain. Always evaluate on your own data.
- **Common confusion:** MTEB is a benchmark, not a model. It ranks models; it does NOT produce embeddings.

### BEIR Benchmark
- **Definition:** A zero-shot retrieval benchmark across 18 diverse datasets (fact-checking, biomedical, legal, QA, etc.). Tests generalization of embedding models to new domains without fine-tuning.
- **When it matters (exam trap):** BEIR tests zero-shot generalization. If a model scores high on BEIR, it is robust across domains. If it scores low, it may need domain-specific fine-tuning.
- **Common confusion:** BEIR is for retrieval only (not classification, clustering, etc.). MTEB is broader.

### GraphRAG
- **Definition:** A RAG variant that builds a knowledge graph from the corpus and uses graph traversal alongside vector search. Particularly useful for global, multi-hop questions that require synthesizing information from many documents.
- **When it matters (exam trap):** GraphRAG is a recent (2024) research direction from Microsoft. It is NOT the same as standard RAG. If a question describes "building a knowledge graph and traversing it during retrieval," that is GraphRAG.
- **Common confusion:** GraphRAG is NOT just RAG with a graph database. It is a specific architecture that combines graph traversal and vector search.

### Self-RAG / FLARE / Adaptive Retrieval
- **Definition:** RAG variants where the LLM decides when and what to retrieve during generation. Instead of retrieving once before generation, the LLM can issue follow-up retrieval queries mid-generation if it detects uncertainty.
- **When it matters (exam trap):** Adaptive retrieval is a form of Agentic RAG. If a question describes "LLM decides when to retrieve during generation," that is adaptive/self-RAG.
- **Common confusion:** Self-RAG is NOT the same as multi-query RAG. Multi-query retrieves multiple paraphrased queries upfront. Self-RAG retrieves dynamically during generation.

### Reasoning-Augmented Retrieval (HyDE, Step-Back)
- **Definition:** Replace or augment the user query with an LLM-generated query before retrieval.
  - **HyDE (Hypothetical Document Embeddings):** LLM generates a hypothetical answer; that answer is embedded and used as the retrieval query. Works because answers are semantically closer to relevant docs than questions are.
  - **Step-back prompting:** Abstract the query to a higher-level concept before retrieval. Example: "How do I reduce my electricity bill?" → "What factors affect household electricity usage?"
- **When it matters (exam trap):** HyDE and step-back are pre-retrieval query rewriting strategies. They improve retrieval for short, ambiguous, or complex queries. If a question asks "how to improve retrieval for ambiguous queries?" → Query rewriting (HyDE, step-back, multi-query).
- **Common confusion:** HyDE generates a hypothetical answer, NOT a paraphrased question. It embeds the answer and retrieves docs similar to that answer.

## Pipelines / architectures described

### Standard Dense Retrieval Pipeline
1. **Offline indexing:**
   - Ingest documents (PDFs, HTML, etc.) → parse with document processing tools (PyMuPDF, unstructured, Docling).
   - Chunk documents (300-800 tokens, 10-20% overlap, recursive splitting).
   - Encode each chunk with a bi-encoder (e.g., bge-large-en-v1.5, text-embedding-3-small) → get dense vectors (768 or 1536 dims).
   - Store vectors in a vector database (ChromaDB, Qdrant, pgvector) with metadata (doc_id, date, tenant, ACL).
   - Build ANN index (HNSW, IVF) for fast search. Optionally apply quantization (int8, PQ) to reduce memory.

2. **Online retrieval:**
   - User query arrives → encode query with same bi-encoder → get query vector.
   - Query vector database: find top-k approximate nearest neighbors (cosine similarity or dot product) via ANN index.
   - Apply metadata filters (e.g., tenant=X, date>2024-01-01) at query time.
   - Return top-k chunks.

3. **Optional reranking:**
   - Pass top-50 chunks to a cross-encoder (bge-reranker-large, Cohere Rerank).
   - Cross-encoder scores each (query, chunk) pair.
   - Rerank to top-5 or top-10.

4. **Augmentation + generation:**
   - Concatenate top-5 chunks into prompt.
   - LLM generates answer conditioned on chunks.

### Hybrid Retrieval Pipeline (BM25 + Dense + RRF)
1. **Offline indexing:**
   - Same as dense retrieval, but ALSO build an inverted index (BM25) for sparse retrieval.
   - Tools: Elasticsearch, pgvector (supports both), Qdrant (supports hybrid).

2. **Online retrieval:**
   - Run BM25 query → get top-k1 results (ranked by BM25 score).
   - Run dense query → get top-k2 results (ranked by cosine similarity).
   - Fuse the two result lists with Reciprocal Rank Fusion (RRF):
     - For each doc, RRF_score = sum over rankers of [1 / (k + rank)], k=60.
     - Sort by RRF_score descending.
   - Return top-k combined results.

3. **Optional reranking:** Same as above (cross-encoder on top-50 → top-5).

4. **Augmentation + generation:** Same as above.

### Production Best Practices (Lecture 10 Recommendations)
- **Embedding model:** text-embedding-3-small for prototype; bge-large-en-v1.5 or text-embedding-3-large for production.
- **Vector DB:** ChromaDB or pgvector for small/medium scale; Qdrant, Milvus, or managed service (Pinecone, Weaviate) for billion-scale.
- **Chunk size:** 300-800 tokens with 10-20% overlap, recursive splitting.
- **Retrieval:** Hybrid (BM25 + dense) with RRF.
- **Reranker:** Cohere Rerank API or bge-reranker-large on top-50 → top-5.
- **ANN index:** HNSW for speed (ChromaDB, Qdrant default). IVF for lower memory (FAISS).
- **Quantization:** int8 for 4x memory reduction with <1% recall loss.
- **Eval:** Build a domain-specific eval set (100 queries from real users) and measure retrieval recall@k, MRR, nDCG. Also measure RAGAS metrics (faithfulness, answer relevancy).

## Likely MCQ angles

1. **Bi-encoder vs cross-encoder:** Bi-encoder for retrieval (scalable, precompute docs). Cross-encoder for reranking (accurate, too slow for full corpus).
2. **Embedding similarity metrics:** Cosine, dot product, L2 distance. For L2-normalized vectors, all three give the same ranking. Dot product is cheapest.
3. **Chunking size trade-off:** Smaller chunks = higher precision but fragmentation. Larger chunks = more context but blurrier semantics. No universal right size.
4. **Recursive splitting vs fixed-size:** Recursive preserves natural boundaries (paragraphs, sentences). Fixed-size is simpler but may split mid-sentence.
5. **Hybrid retrieval (BM25 + dense):** Industrial default. Combines keyword precision (BM25) and semantic understanding (dense). RRF fuses results without tuning weights.
6. **Reciprocal Rank Fusion (RRF):** Rank-based fusion. RRF_score = sum over rankers of [1 / (k + rank)], k=60. Higher score = better.
7. **Reranking pipeline:** Retrieve top-50 with bi-encoder → rerank to top-5 with cross-encoder. Improves precision.
8. **Vector database components:** Storage + ANN index + metadata + persistence + (sometimes) hybrid search. NOT a replacement for relational DBs.
9. **ANN speedup sources:** Quantization (int8, PQ) + ANN index structure (HNSW, IVF). NOT faster embedding models.
10. **Quantization trade-off:** Reduces memory and speeds up similarity computation. Cost: slight accuracy loss (<1% recall drop with int8).
11. **Matryoshka embeddings:** Variable-dim embeddings. First d dimensions are valid truncations. Use 128-dim for cheap retrieval, 1536-dim for precise reranking.
12. **Late-interaction (ColBERT):** Stores per-token embeddings. Computes token-level similarity at query time. Middle ground between bi-encoder (cheap, coarse) and cross-encoder (expensive, fine).
13. **Instruction-tuned embedders:** Accept instruction prefix. More flexible than generic embedders. No fine-tuning needed, just change the instruction.
14. **BGE family:** bge-large-en-v1.5 for retrieval (bi-encoder). bge-reranker-large for reranking (cross-encoder). Open-source SOTA.
15. **MTEB caveat:** Averages 50+ tasks. Your domain may correlate with only a few. Always evaluate on your own data.
16. **BEIR benchmark:** Zero-shot retrieval across 18 diverse datasets. Tests generalization.
17. **GraphRAG:** Builds knowledge graph + vector search. For global, multi-hop questions. Not the same as standard RAG.
18. **Self-RAG / adaptive retrieval:** LLM decides when to retrieve during generation. Not the same as multi-query RAG (which retrieves upfront).
19. **HyDE:** LLM generates hypothetical answer; embed answer and retrieve docs similar to it. Works because answers are closer to docs than questions.
20. **Step-back prompting:** Abstract query to higher-level concept before retrieval. Improves coverage for complex questions.
21. **Document processing failures:** PDFs, OCR, multi-column layouts, tables are largest source of RAG quality regression. Tooling choice matters more than embedding choice for many corpora.
22. **Metadata filtering:** Per-document ACLs, tenant, date range. Frequent source of bugs (leaked vector = leaked document). Mandatory for enterprise search.
23. **Long-context LLMs + RAG:** Complementary, not replacement. RAG provides citation, freshness, access control, and cost efficiency. Long-context does NOT replace RAG.
24. **FAISS vs ChromaDB:** FAISS is a library (ANN index, no persistence). ChromaDB is a full database (storage, index, metadata, server).
25. **ColBERTv2 / PLAID:** Late-interaction retrievers. Higher quality, higher storage cost. Increasingly viable with PQ compression.
26. **Reranking-as-a-service:** Cohere Rerank, Voyage Rerank, Jina Rerank. Low-friction API call for reranking.

## One-line flashcards (20)

- Q: Bi-encoder encodes query and doc how? → A: Independently (so doc embeddings can be precomputed and indexed).
- Q: Cross-encoder encodes query and doc how? → A: Together (into a single representation; outputs relevance score).
- Q: Which is used for retrieval from a large corpus: bi-encoder or cross-encoder? → A: Bi-encoder (cross-encoder is too slow).
- Q: Which is used for reranking top-k results: bi-encoder or cross-encoder? → A: Cross-encoder (more accurate).
- Q: Recommended chunk size for RAG? → A: 300-800 tokens with 10-20% overlap, recursive splitting.
- Q: Smaller chunks give higher precision or recall? → A: Higher precision (but risk fragmentation).
- Q: Larger chunks give more context or less? → A: More context (but blurrier semantics, lower precision).
- Q: Hybrid retrieval combines which two methods? → A: BM25 (sparse) + dense embeddings.
- Q: RRF (Reciprocal Rank Fusion) formula? → A: RRF_score = sum over rankers of [1 / (k + rank)], k=60.
- Q: Reranking pipeline: retrieve how many with bi-encoder, rerank to how many with cross-encoder? → A: Retrieve top-50, rerank to top-5.
- Q: Vector database core components? → A: Storage + ANN index + metadata + persistence.
- Q: ANN speedup comes from? → A: Quantization (int8, PQ) + ANN index structure (HNSW, IVF).
- Q: int8 quantization gives how much memory reduction? → A: 4x (float32 → int8).
- Q: Matryoshka embeddings allow what? → A: Variable dimensionality (first d dims are valid truncations).
- Q: Late-interaction (ColBERT) stores what? → A: Per-token embeddings (not per-passage).
- Q: Instruction-tuned embedders accept what? → A: Instruction prefix (e.g., "Represent this question for retrieval: ...").
- Q: BGE-large-en-v1.5 is used for? → A: Retrieval (bi-encoder).
- Q: bge-reranker-large is used for? → A: Reranking (cross-encoder).
- Q: MTEB is a benchmark for? → A: Ranking embedding models (50+ tasks).
- Q: BEIR tests what? → A: Zero-shot retrieval generalization across 18 diverse datasets.
- Q: GraphRAG combines what? → A: Knowledge graph traversal + vector search.
- Q: HyDE generates what before retrieval? → A: Hypothetical answer (embed answer, retrieve docs similar to it).
- Q: Step-back prompting does what? → A: Abstracts query to higher-level concept before retrieval.
- Q: Document processing (PDFs, OCR) is the largest source of? → A: RAG quality regression (more than embedding model choice).
- Q: FAISS is a? → A: Library (ANN index), not a full database (no persistence).
