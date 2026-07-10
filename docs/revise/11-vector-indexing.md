# Lecture 11 — Vector DB Indexing, Routing, and RAG Security
**Instructor:** Prof. Pawan Goyal

## TL;DR (5 bullets)
- Product Quantization (PQ) compresses vectors by 32x+ using codebooks and byte codes; IVF prunes search scope by clustering
- HNSW builds skip-list graphs for O(log N) search; efSearch is the recall-latency dial
- Query routing (rule/embedding/LLM/hybrid) picks data source or model before retrieval; embedding-based = cosine-match to 20-50 sample profiles, no LLM
- Retrieval Gateway enforces RBAC by injecting metadata filters at DB query time (NOT post-filter, NOT system prompt)
- Semantic caching uses high cosine threshold (tau) because false hit (wrong answer) >> false miss (extra cost)

## Exam-relevant concepts

### Product Quantization (PQ)
- **Definition:** Compression technique that splits D-dim vectors into m sub-vectors, replaces each with nearest centroid ID (0-255), storing 1 byte per sub-vector instead of original floats.
- **Formula / numbers:**
  - Raw storage = D dimensions × 4 bytes (float32) × N vectors
  - Post-PQ storage = m sub-vectors × 1 byte × N vectors
  - Per-query lookup table = m × k entries (m sub-vectors, k=256 centroids per codebook)
  - Distance calculation per DB vector = m lookups + (m-1) additions
- **When it matters (exam trap):** Q19-Q22 sample questions test exact calculation. Setup: 10M vectors, 1024-dim, m=16, k=256.
- **Common confusion:** Thinking PQ reduces search scope (NO — that's IVF). PQ only compresses memory footprint.

### IVF (Inverted File)
- **Definition:** Clustering technique that groups vectors into k buckets (coarse centroids), searches only nprobe nearest buckets.
- **Formula:** Search scope reduced from O(N) to O(N/k) when nprobe=1; O(nprobe·N/k) generally.
- **When it matters (exam trap):** Combined with PQ (IVF+PQ) for 50M+ vector datasets under strict RAM constraints.
- **Common confusion:** nprobe is IVF's recall dial (like efSearch for HNSW), NOT a compression parameter.

### HNSW (Hierarchical Navigable Small World)
- **Definition:** Skip-list-inspired graph index with layered proximity connections; greedy descent from top (few nodes, long hops) to layer-0 (all nodes).
- **Formula:** Expected search = O(log N) hops. Memory = O(N·M) where M = edges per node.
- **Knobs:**
  - M = build-time edges per node (baseline quality + memory)
  - efSearch = query-time candidate list size (recall-speed trade-off)
- **When it matters (exam trap):** Compare IVF+PQ (RAM-constrained, massive scale) vs HNSW (graph-based, higher memory, faster for moderate scale).
- **Common confusion:** efSearch is query-time tunable; M requires rebuild.

### ColBERT (Late Interaction)
- **Definition:** Token-level embeddings (not doc-level); query-time MaxSim operator scores each query token's best match across doc tokens.
- **Formula:** score = Σᵢ max_j (eᵢ · eⱼ)
- **Trade-off:** Near cross-encoder accuracy at bi-encoder precompute cost, but index size ~ tokens (not docs).
- **When it matters (exam trap):** "Accuracy close to cross-encoder, no query-time joint attention, willing to provision more storage" → ColBERT (Q sample from transcript).
- **Common confusion:** ColBERT is NOT two-stage retrieve-then-rerank; it's a single-stage retrieval with deferred interaction.

### Bi-encoder vs Cross-encoder
- **Bi-encoder:** Separate encodings, score = E(q) · E(d), precompute doc vectors offline.
- **Cross-encoder:** Joint encoding [q ; SEP ; d], every query token attends to every doc token, N forward passes for N docs.
- **Industry pattern:** Retrieve 100-200 with bi-encoder + ANN, rerank top-K with cross-encoder.
- **When it matters (exam trap):** "Cannot afford query-time latency of joint attention over all candidates" rules OUT cross-encoder retrieval stage.
- **Common confusion:** Cross-encoder is for reranking retrieved candidates, not initial retrieval at scale.

### Request Shaping (Query Transformation)
- **Rewriting:** LLM rewrites vague query into specific search statement (e.g., "OOMKilled" → "What causes OOMKilled exit code in Kubernetes?").
- **Follow-up:** Inject conversation history to make standalone query ("it" → "parental leave policy").
- **Multi-Query:** Generate multiple distinct queries for complex prompts, retrieve in parallel.
- **Step-Back Prompting:** Abstract hyper-specific query into broader foundational question to retrieve underlying rules.
- **When it matters (exam trap):** "No results on deep-in-the-weeds query, DB has architectural rules but lacks specific keywords" → Step-Back Prompting.
- **Common confusion:** These are pre-retrieval transforms, NOT routing (though related).

## PQ Cheat Sheet

### Sample-Paper Setup (memorize exact numbers)
Given: 10M vectors, 1024-dim, float32, PQ with m=16 sub-vectors, k=256 centroids per codebook.

| Metric | Formula | Calculation | Result |
|--------|---------|-------------|--------|
| Raw storage | D × 4 bytes × N | 1024 × 4 × 10M | 40.96 GB |
| Post-PQ storage | m × 1 byte × N | 16 × 1 × 10M | 160 MB |
| Compression ratio | Raw / Post-PQ | 40960 / 160 | 256x |
| Lookup table size (per query) | m × k | 16 × 256 | 4,096 entries |
| Lookups per DB vector | m | 16 | 16 |
| Additions per DB vector | m - 1 | 16 - 1 | 15 |

### General PQ Formulas
- **D** = original dimension
- **m** = number of sub-vectors
- **k** = centroids per codebook (typically 256 for 1-byte codes)
- **N** = number of vectors in DB

**Storage:**
- Raw: `D × 4 × N` bytes (float32)
- PQ: `m × 1 × N` bytes (1 byte per sub-vector code)

**Query-time:**
- Lookup table: `m × k` entries (precompute distance from query sub-vectors to all centroids)
- Per-vector distance: sum `m` table lookups (each sub-vector code indexes into its codebook's precomputed distances)

## Routing patterns

| Strategy | How Set Up | How Matched | When to Pick |
|----------|-----------|-------------|--------------|
| **Rule-Based (Logical)** | Hard-coded if/else, regex keywords (e.g., if contains "PTO" → HR) | String match against query | Zero cost, deterministic, extremely low latency; use when keywords are stable and domain is simple |
| **Embedding-Based** | Build profiles: 20-50 sample queries per DB, embed them | Embed live query, cosine-match to profiles, route to nearest cluster | Fast, semantic, no LLM, offline-capable; use when slight phrasing variations break regex |
| **LLM-Based** | Prompt LLM with route descriptions, ask for JSON {"target_database": "X", "reasoning": "Y"} | LLM reasons and returns structured route decision | Highest accuracy for ambiguous queries; use when simpler methods fail but adds latency/cost |
| **Classifier-Based** | Train lightweight ML model (Random Forest, small BERT) on historical query→label pairs | Ingest query, output probability scores per category | Balances cost and accuracy for high-traffic, stable intent categories (HR/Legal/Code) |
| **Hybrid (Waterfall)** | Chain: Rule → Embedding → LLM; escalate only on low confidence | Try cheapest first; if confidence < threshold, fall to next tier | Production best practice: pay for expensive LLM only when cheap deterministic/semantic methods ambiguous |

**Exam trap:** "Avoid expensive LLM router for every query, but pure regex too brittle" → Hybrid routing waterfall (Q from transcript).

**Key distinction for embedding-based:**
- NO LLM in the loop at query time
- Offline-capable (just embedding model + cosine similarity)
- Contrast with LLM router (calls LLM for classification) and rule-based (no semantic understanding)

## Security enforcement points

### Where each defense sits in RAG pipeline
1. **Retrieval Gateway (CORRECT for Least Privilege / RBAC)**
   - Intercepts identity token (SSO/AD)
   - Reads user's role/clearance
   - Injects metadata filter into Vector DB query BEFORE retrieval
   - Example: `where={"clearance_level": user.role}`
   - Result: DB physically cannot return unauthorized docs

2. **System Prompt (UNRELIABLE)**
   - Instructs LLM to refuse answering if context contains sensitive data
   - Trap: context already retrieved; relies on LLM following instruction (not enforcement)

3. **Post-Filter (TOO LATE)**
   - Retrieve all docs, then filter by role after retrieval
   - Trap: sensitive data already passed through system; audit trail shows unauthorized access

4. **LLM Router (WRONG LAYER)**
   - Classify query intent as "Sensitive" and block generation
   - Trap: router guesses intent, doesn't verify user permission; no document-level control

**Exam answer pattern (from transcript Q):**
"To enforce Least Privilege, implement AuthZ in the Retrieval Gateway by intercepting user's identity token and injecting dynamic metadata filter into Vector DB query BEFORE retrieval."

**Common confusion:**
- Post-filter vs pre-filter: post-filter retrieves everything first (data leak in logs), pre-filter never retrieves unauthorized docs
- System prompt vs gateway: prompt is guidance, gateway is enforcement at data layer

### Retrieval Gateway responsibilities (beyond security)
- **Access Control (AuthN/AuthZ):** RBAC metadata filtering as above
- **Quota Management (FinOps):** Track usage per user/team, throttle at budget threshold, route to cheaper models
- **Semantic Caching:** Embed query, check cosine similarity to cached queries, serve cached answer if similarity ≥ tau
- **Observability:** Log every query, retrieved chunks, citations, latency, cache hits to central dashboard

## Semantic Caching threshold (tau)

### The asymmetry
- **False hit (serve cached answer for different query):** Returns wrong answer → damages user trust >> lost API cost
- **Cache miss (no match, run full pipeline):** Pays standard LLM + retrieval cost → just money

**Decision rule:**
```
if cos(query, cached_query) ≥ tau:
    serve cached answer
else:
    run full retrieval + LLM
```

**Optimal design (exam answer from transcript Q):**
"Bias threshold HIGH, accepting lower overall cache hit rate and slightly higher baseline costs in exchange for guaranteeing accuracy on cached responses."

**Why NOT low threshold:**
- Maximizing hit rate with low tau increases false hits
- One wrong answer in production >> cost of 100 cache misses

**Savings if hit rate = h:**
```
savings = h × (retrieval_cost + LLM_cost)
```

**Common confusion:**
- Exact-match hash cache eliminates false hits but also eliminates semantic matches (slight rephrasing = miss)
- Dynamic cross-encoder verification on every cache hit defeats the purpose (still paying for expensive model)

## Design decision trees (from transcript)

### Q1: ChromaDB vs FAISS vs Managed
**Scenario: Prototype, < few million vectors, need metadata + persistence**
→ ChromaDB (stores vectors + text + metadata together, filters by metadata, 15 lines of code, HNSW underneath)

**Scenario: Huge scale (billions), raw speed, fine index control**
→ FAISS library (fastest, most tunable, scales to billions, full control over index type)
→ Trap: no persistence, no metadata, no text storage — must maintain own ID→text map

### Q2: Which FAISS index?
**Up to ~1M vectors, require 100% recall**
→ IndexFlatIP (exact brute force, always finds true nearest neighbors)

**Millions of vectors, tolerate slight recall loss, fits in RAM**
→ IVFFlat (nlist ≈ √N, nprobe = recall dial, needs training ≥ 30-40× nlist points)
→ Trap: nprobe=1 fastest/lowest-recall; nprobe=nlist degenerates to exact

**50M+ vectors, STRICT memory (RAM) constraints, tolerate recall loss**
→ IVF + PQ (IVF clusters, PQ compresses; directly addresses strict RAM limits)
→ Exam answer (Q from transcript): "Which indexing strategy for 50M chunks, strict RAM, tolerate slight recall drop?" → IVF+PQ

**Moderate scale, graph-based, higher memory available**
→ HNSW (O(log N) search, memory = O(N·M), efSearch dial)

## Retrieval pipeline failures (silent bugs)

### 1. Empty chunks to prod
**Cause:** Scanned PDF has no text layer, pypdf returns empty strings silently.
**Fix:** Always probe for text layer; route to OCR if absent.

### 2. Half the recall, no warning
**Cause:** Asymmetric embedding model used WITHOUT `query:` / `passage:` prefixes.
**Fix:** Read model card, use correct prefixes.

### 3. Euclidean on a cosine model
**Cause:** Using DB's default L2 metric on model trained for cosine similarity.
**Fix:** Match distance metric to training (IndexFlatIP for cosine = inner product on normalized vectors).

**Exam trap:** All three fail quietly — no error, no crash, just worse answers.

## Likely MCQ angles

1. **PQ calculation:** Given 10M vectors, 1024-dim, m=16, k=256 → raw storage 40.96 GB, PQ storage 160 MB, lookup table 4096 entries, 16 lookups + 15 additions per vector.

2. **IVF+PQ vs HNSW:** "50M vectors, strict RAM, tolerate recall loss" → IVF+PQ. "Moderate scale, graph-based" → HNSW.

3. **ColBERT vs Cross-encoder:** "Accuracy close to cross-encoder, no query-time joint attention, willing to provision more storage" → ColBERT (late interaction, token-level embeddings).

4. **Step-Back Prompting:** "No results on hyper-specific query, DB has architectural rules but lacks keywords" → Step-Back (abstract to broader foundational question).

5. **Hybrid Routing:** "Avoid expensive LLM router for every query, regex too brittle" → Hybrid waterfall (Rule → Embedding → LLM).

6. **Least Privilege enforcement:** "Ensure junior engineer cannot access executive salary data" → Retrieval Gateway intercepts token, injects metadata filter BEFORE retrieval (NOT system prompt, NOT post-filter).

7. **Semantic caching threshold:** "False hit damages trust more than cache miss costs" → Bias tau HIGH (lower hit rate, guarantee accuracy).

8. **Two-stage retrieval:** Bi-encoder + ANN retrieves 100-200 candidates, cross-encoder reranks top-K, LLM generates from top 5-20.

9. **Index choice branching:**
   - Up to 1M / need 100% recall → IndexFlatIP
   - Millions / tolerate loss / RAM OK → IVFFlat
   - 50M+ / strict RAM → IVF+PQ
   - Graph-based / moderate scale → HNSW

10. **Embedding-based routing:** Build 20-50 sample profiles per DB, cosine-match live query to profiles, no LLM in loop, offline-capable.

## One-line flashcards

- Q: PQ raw storage for 10M vectors, 1024-dim float32? → A: 1024 × 4 × 10M = 40.96 GB
- Q: PQ post-compression storage, m=16, 10M vectors? → A: 16 × 1 × 10M = 160 MB
- Q: PQ lookup table size per query, m=16, k=256? → A: 16 × 256 = 4,096 entries
- Q: PQ distance calculation per DB vector, m=16? → A: 16 lookups + 15 additions
- Q: IVF reduces what? → A: Search scope from O(N) to O(N/k); NOT compression
- Q: IVF recall dial? → A: nprobe (number of clusters searched)
- Q: HNSW time complexity? → A: O(log N)
- Q: HNSW recall-speed knob? → A: efSearch (query-time candidate list size)
- Q: HNSW build-time parameter? → A: M (edges per node; sets baseline quality + memory)
- Q: ColBERT MaxSim formula? → A: score = Σᵢ max_j (eᵢ · eⱼ)
- Q: ColBERT trade-off? → A: Near cross-encoder accuracy, bi-encoder cost, but index size ~ tokens
- Q: Bi-encoder score? → A: E(q) · E(d), precompute docs offline
- Q: Cross-encoder score? → A: sig(W · BERT([q;SEP;d])), N forward passes for N docs
- Q: Two-stage pattern? → A: Bi-encoder retrieves 100-200, cross-encoder reranks top-K, LLM generates
- Q: Step-Back Prompting when? → A: Hyper-specific query, no results, DB has underlying rules but lacks exact keywords
- Q: Multi-Query Retrieval when? → A: Complex prompt spans multiple microservices or entities, retrieve in parallel
- Q: Follow-up transformation when? → A: Multi-turn, user uses pronouns ("it", "they"), inject history for standalone query
- Q: Rewriting when? → A: Vague/fragmented query (e.g., "OOMKilled"), LLM rewrites to specific search statement
- Q: Embedding-based routing setup? → A: 20-50 sample queries per DB, embed them, cosine-match live query
- Q: Embedding-based routing LLM in loop? → A: NO (contrast LLM router which asks LLM for JSON route)
- Q: Hybrid routing waterfall? → A: Rule → Embedding → LLM; escalate only on low confidence
- Q: Least Privilege enforcement point? → A: Retrieval Gateway, intercept token, inject metadata filter BEFORE retrieval
- Q: Why NOT system prompt for AuthZ? → A: Guidance not enforcement, context already retrieved
- Q: Why NOT post-filter for AuthZ? → A: Already retrieved unauthorized docs (data leak in logs)
- Q: Semantic caching asymmetry? → A: False hit (wrong answer) >> false miss (extra cost)
- Q: Semantic caching optimal threshold? → A: Bias HIGH (lower hit rate, guarantee accuracy)
- Q: Index for 50M vectors, strict RAM? → A: IVF + PQ (clusters + compresses)
- Q: Index for <1M, need 100% recall? → A: IndexFlatIP (exact brute force)
- Q: Index for millions, tolerate loss, RAM OK? → A: IVFFlat (nlist ≈ √N, nprobe dial)
- Q: Silent RAG bug: scanned PDF? → A: No text layer, pypdf empty strings; probe first, OCR if needed
- Q: Silent RAG bug: asymmetric model? → A: Missing query:/passage: prefixes; read model card
- Q: Silent RAG bug: distance metric? → A: L2 on cosine model; match metric to training
