# Lecture 11 — Vector Database Indexing, Routing, and RAG Security
**Estimated read time:** 30–40 min · **Difficulty:** Core to Advanced

## What this lecture is about

This lecture solves the critical scaling problem that every production RAG system eventually faces. When you store a few thousand document vectors in a database and need to find the nearest neighbors to a query, brute-force cosine similarity works fine. But when your dataset grows to 10 million, 50 million, or a billion vectors, the compute and memory costs explode. This is where approximate nearest neighbor algorithms, compression techniques, and intelligent routing come in.

You will learn how to design vector search systems that can handle massive scale without sacrificing too much accuracy. You will understand why combining different techniques like IVF (Inverted File indexes) with Product Quantization gives you the best trade-off between memory usage, search speed, and retrieval quality. The lecture also covers how to route queries intelligently to the right data sources before retrieval happens, and most importantly, how to secure your RAG system so that unauthorized users cannot access sensitive documents. This is the highest-yield material from Sample Paper Section 4 — questions 19–24 come directly from these concepts.

## Prerequisites

Before diving in, you should understand:
- What a vector embedding is (a list of numbers representing semantic meaning)
- Cosine similarity as a distance metric
- Basic k-means clustering
- The RAG pipeline: Embed → Retrieve → Generate
- What a vector database does (stores embeddings and finds nearest neighbors)

If those concepts are fuzzy, review Lecture 10 first.

---

## Concept 1: The Scale Problem — Why Brute Force Fails

### The problem it solves

When you have 10,000 document vectors and a query comes in, your system computes cosine similarity between the query and all 10,000 vectors, sorts them, and returns the top-k. That takes maybe 10 milliseconds. But when you have 10 million vectors, that same brute-force scan takes thousands of times longer. The compute cost scales linearly: double the vectors, double the time. And the memory cost is equally punishing: storing 10 million vectors at 1024 dimensions each, with 32-bit floats, requires over 40 GB of RAM.

### The intuition (analogy first)

Imagine you are looking for a book in a massive library with 10 million books, and the only way to find it is to walk down every single aisle, pull every single book off the shelf, and check the title. That is brute-force search. Now imagine instead that the library is organized into sections (Fiction, History, Science), and within each section, books are clustered by topic. When you arrive, you first check which section your book is likely in, then walk only those aisles. That is approximate nearest neighbor search. You trade a tiny risk of missing the perfect book (maybe it was misfiled) for a massive speed boost.

### The two bottlenecks

1. **Compute bottleneck**: Comparing a query to N vectors requires N distance calculations. At N = 10M, this is prohibitively slow.
2. **Memory bottleneck**: Storing N vectors at D dimensions with 4 bytes per float requires N × D × 4 bytes. For N = 10M, D = 1024, that is 40.96 GB.

### The solution space

There are two families of techniques to solve these:
- **Pruning methods (IVF, HNSW)**: Reduce the number of vectors you compare against by clustering or building graph structures.
- **Compression methods (Product Quantization)**: Reduce the memory footprint of each vector by replacing high-precision floats with compact byte codes.

Production systems often combine both.

### Exam angle

Expect questions that describe a dataset size (e.g., 50 million vectors), mention RAM constraints, and ask which indexing strategy to use. The answer will hinge on whether you need to optimize for memory, speed, or both.

---

## Concept 2: IVF (Inverted File Index) — Clustering to Prune Search Space

### The problem it solves

IVF answers the question: how do I avoid comparing my query to all N vectors? It does so by clustering the vectors into buckets, then searching only the buckets nearest to the query.

### The intuition (analogy first)

Think of a post office sorting mail into zip-code bins. When a letter arrives for zip code 94102, the postal worker does not check every single address in the country. They go straight to the 94102 bin and search only there. IVF does the same with vectors. You run k-means clustering to create nlist clusters (the bins), assign each vector to its nearest cluster, then at query time, you find the nprobe nearest cluster centroids and search only those bins.

### How it works (step by step)

**Indexing phase (offline):**

1. You have N vectors in D-dimensional space.
2. You run k-means clustering with nlist clusters (e.g., nlist = 1024).
3. K-means finds 1024 centroids. Each centroid represents the "center" of a cluster.
4. You assign each of the N vectors to its nearest centroid. Now you have 1024 buckets, each containing roughly N / 1024 vectors.

**Query phase (online):**

1. A query vector q arrives.
2. You compute the distance from q to each of the 1024 centroids.
3. You pick the nprobe nearest centroids (e.g., nprobe = 10).
4. You search only the vectors in those 10 buckets (so you compare q to roughly 10 × N / 1024 vectors instead of N).
5. You return the top-k from those candidates.

### Worked example (with real numbers)

Suppose N = 10,000,000 vectors, nlist = 1000, nprobe = 10.

- Each cluster has roughly 10,000,000 / 1000 = 10,000 vectors.
- At query time, you search 10 clusters = 10 × 10,000 = 100,000 vectors.
- You avoided comparing against 9,900,000 vectors (99% reduction).
- If each distance calculation takes 1 microsecond, brute-force would take 10 seconds, but IVF takes 0.1 seconds.

**Trade-off:** You might miss the true nearest neighbor if it happens to live in a cluster you did not probe. This is the "approximate" part of ANN.

### The formula

Search complexity: O(nprobe × N / nlist)

When nprobe = 1, you get the fastest search but lowest recall. When nprobe = nlist, you search every cluster and recover exact brute-force (no speed gain). In practice, nprobe is tuned as the recall dial — higher nprobe = higher recall but slower.

### Common misunderstandings

- **"IVF compresses vectors."** No. IVF only prunes the search space. Each vector is still stored in full precision.
- **"nlist should always be very large."** Not true. If nlist is too large and your dataset is too small, k-means will fail because there are not enough points per cluster. A common rule: you need at least 30–40× nlist training points.
- **"nprobe is set at build time."** No. nprobe is a query-time parameter. You can adjust it per query without rebuilding the index.

### Exam angle

Sample paper question: You have 50 million vectors, strict RAM limits, can tolerate slight recall drop. Which indexing strategy? The answer will likely involve IVF or IVF+PQ. The key is recognizing that IVF reduces search scope (not memory), so it needs to be paired with compression if RAM is the bottleneck.

---

## Concept 3: Product Quantization (PQ) — Compressing Vectors by 32× or More

### The problem it solves

Even after IVF prunes the search space, you still need to store all N vectors in memory. For N = 10M, D = 1024, that is 40.96 GB. If your server has only 16 GB of RAM, you cannot fit the index. Product Quantization compresses each vector from D × 4 bytes down to m bytes (typically m = 8 to 128), achieving compression ratios of 32× to 256×.

### The intuition (analogy first)

Imagine you are a photographer with 10 million high-resolution images (each 10 MB). Storing them all requires 100 TB. Instead, you create a palette of 256 common colors for each region of the image (top-left, top-right, etc.). Then for each region, instead of storing the exact RGB values of every pixel, you store the ID of the closest color in the palette (1 byte). The image looks slightly blurry when reconstructed, but it fits in 1/100th the space. PQ does the same with vectors.

### How it works (step by step)

**Training phase (offline):**

1. Split each D-dimensional vector into m sub-vectors of size D/m. For example, if D = 1024 and m = 16, each sub-vector has 64 dimensions.
2. For each of the m sub-spaces, run k-means with k centroids (typically k = 256). This creates m "codebooks," each with 256 centroids.
3. The codebooks are your palettes.

**Encoding phase (offline):**

1. For each vector in your database, split it into m sub-vectors.
2. For each sub-vector, find the nearest centroid in its corresponding codebook.
3. Replace the sub-vector with the ID of that centroid (0–255 = 1 byte).
4. Now the vector is represented by m bytes instead of D × 4 bytes.

**Query phase (online):**

1. A query vector q arrives.
2. Split q into m sub-vectors.
3. For each sub-vector q_i, compute the distance from q_i to all k centroids in codebook i. Store these in a lookup table (LUT). The LUT has m × k entries (e.g., 16 × 256 = 4,096).
4. For each database vector x (now represented by m byte codes), compute the approximate distance from q to x by summing m lookups in the LUT.
   - d(q, x) ≈ LUT[1][code_1] + LUT[2][code_2] + ... + LUT[m][code_m]
5. This sum requires m lookups and (m-1) additions — very fast.

### Worked example 1: Sample paper setup (N=10M, D=1024, m=16, k=256)

**Raw storage:**
- Each vector: D × 4 bytes = 1024 × 4 = 4,096 bytes
- Total: N × 4,096 = 10,000,000 × 4,096 = 40,960,000,000 bytes = 40.96 GB

**PQ storage:**
- Each vector: m × 1 byte = 16 bytes
- Total: N × 16 = 10,000,000 × 16 = 160,000,000 bytes = 160 MB

**Compression ratio:** 40.96 GB / 160 MB = 256×

**Query cost:**
- Build LUT: m × k distance calculations = 16 × 256 = 4,096 (done once per query)
- Distance to one DB vector: m lookups + (m-1) additions = 16 + 15 = 31 operations (extremely cheap)

**Why this is fast:** The LUT is computed once, then each of the 10 million database vectors is scored with just 31 operations. Compare this to brute-force, where each database vector requires D multiplications and D additions (1024 muls + 1024 adds = 2048 operations per vector). PQ is roughly 64× faster per vector, on top of using 256× less memory.

### Worked example 2: Generalize with different numbers (N=5M, D=768, m=12, k=256)

**Raw storage:**
- Each vector: 768 × 4 = 3,072 bytes
- Total: 5,000,000 × 3,072 = 15,360,000,000 bytes = 15.36 GB

**PQ storage:**
- Each vector: 12 × 1 = 12 bytes
- Total: 5,000,000 × 12 = 60,000,000 bytes = 60 MB

**Compression ratio:** 15.36 GB / 60 MB = 256×

**Query cost:**
- LUT size: m × k = 12 × 256 = 3,072 entries
- Distance to one vector: 12 lookups + 11 additions = 23 operations

**Key insight:** The compression ratio stays constant (256×) because k = 256 = 2^8 = 1 byte per sub-vector. The query cost scales linearly with m (the number of sub-vectors). Larger m gives finer granularity (better recall) but slightly more query cost. Typical production values: m = 8 to 128.

### The formula (distance approximation)

d(q, x) ≈ Σ_{i=1 to m} LUT[i][code_i(x)]

Where code_i(x) is the 1-byte ID stored for the i-th sub-vector of database vector x.

**Why the approximation works:** The quantization error is small because each sub-space has 256 centroids. Within a 64-dimensional sub-space, 256 centroids can represent the distribution reasonably well. The error is further reduced because errors across sub-vectors tend to be independent and partially cancel out.

### Common misunderstandings

- **"PQ speeds up search."** Not exactly. PQ compresses memory and makes distance calculations cheaper (table lookups instead of dot products), so indirectly it speeds things up. But the main win is memory.
- **"PQ and IVF are the same."** No. IVF prunes search scope by clustering the full dataset. PQ compresses each vector by clustering sub-vectors within each sub-space. They solve different problems.
- **"If I use PQ, I do not need IVF."** Wrong. With PQ alone, you still compare the query to all N vectors (just faster and with less RAM). With IVF alone, you prune search scope but still use 40 GB of RAM. Combining IVF+PQ gives you both: pruned scope and compressed memory.
- **"k must equal 256."** Not always, but 256 is standard because it fits in 1 byte. Larger k (e.g., 512) gives better accuracy but requires 2 bytes per sub-vector (doubling memory).

### Exam angle

Expect a question like: "You have 50M vectors, D=1024, m=16, k=256. How much memory does PQ use?" Calculate: N × m × 1 byte = 50,000,000 × 16 = 800 MB. Then compare to raw: 50M × 1024 × 4 = 204.8 GB. Compression ratio = 256×.

Another question might ask: "What is the per-query lookup table size?" Answer: m × k = 16 × 256 = 4,096 entries. This is computed once per query and reused for all N database vectors.

---

## Concept 4: IVF+PQ — The Industry Standard for Billion-Scale

### The problem it solves

IVF alone reduces search scope from O(N) to O(nprobe × N / nlist), but memory is still O(N × D × 4 bytes). PQ alone compresses memory to O(N × m bytes), but search scope is still O(N). Combining them gives you both benefits: pruned scope and compressed memory.

### How it works (step by step)

**Indexing phase:**

1. Run k-means on the full dataset to create nlist coarse centroids (e.g., nlist = 1024).
2. Assign each vector to its nearest coarse centroid.
3. Compute the residual: residual = vector - centroid. The residual captures the "offset" from the centroid.
4. Split each residual into m sub-vectors and run PQ on the residuals (not the original vectors). This creates m codebooks, each with k centroids.
5. Store the PQ codes (m bytes per vector) in each cluster's inverted list.

**Query phase:**

1. Query q arrives.
2. Find the nprobe nearest coarse centroids.
3. For each selected cluster:
   - Shift the query: q_shifted = q - coarse_centroid
   - Build the PQ lookup table (LUT) for q_shifted against the residual codebooks
   - Score all vectors in that cluster using the LUT
4. Aggregate the top-k across all nprobe clusters.

### Why residuals matter

If you applied PQ directly to the original vectors, the quantization error would be large. By first subtracting the coarse centroid, the residuals are smaller and more homogeneous, so PQ can approximate them more accurately. This is the key insight that makes IVF+PQ work.

### Worked example (N=10M, D=1024, nlist=1000, nprobe=10, m=16, k=256)

**Memory:**
- Raw: 10M × 1024 × 4 = 40.96 GB
- IVF+PQ: 10M × 16 bytes = 160 MB (same as PQ alone)

**Search scope:**
- Brute-force: 10M comparisons
- IVF+PQ: nprobe × (N / nlist) = 10 × 10,000 = 100,000 comparisons

**Speedup:** 10M / 100k = 100× fewer comparisons, and each comparison is faster (LUT lookup vs. dot product).

**Recall:** Slightly lower than brute-force because of two approximations: IVF might miss the true cluster, and PQ introduces quantization error. But with nprobe=10 and m=16, recall is typically 90–95% on standard benchmarks.

### Common misunderstandings

- **"IVF+PQ is always better than HNSW."** Not true. HNSW is faster for low-to-medium scale (< 10M vectors) and does not require training. IVF+PQ is better for massive scale (> 50M) and strict memory limits.
- **"nprobe and m are the same thing."** No. nprobe controls how many clusters you search (IVF parameter). m controls how many sub-vectors you split each vector into (PQ parameter).

### Exam angle

Sample paper Q19: "50 million vectors, strict RAM, slight recall drop — which index?" Answer: IVF+PQ. The question tests whether you understand that IVF reduces scope, PQ compresses memory, and together they handle massive scale with tight RAM.

---

## Concept 5: HNSW (Hierarchical Navigable Small World) — Graph-Based ANN

### The problem it solves

HNSW is an alternative to IVF. Instead of clustering, it builds a multi-layer graph where each node is a vector and edges connect nearby neighbors. Search happens by greedily navigating the graph from top layer (few nodes, long jumps) to bottom layer (all nodes, local hops). This gives O(log N) search complexity.

### The intuition (analogy first)

Think of a road network. When you drive from San Francisco to Los Angeles, you do not take local streets the whole way. You take highways (high-level layer) to get close, then exit onto state roads (mid-level layer), then city streets (low-level layer) to reach your exact address. HNSW does the same: start at the top layer with a few well-connected nodes and long-range links, jump quickly to the region near the query, then descend layer by layer until you find the exact nearest neighbors.

### How it works (step by step)

**Build phase (offline):**

1. Decide the maximum number of layers L (typically log2(N)).
2. For each vector, assign it to layer 0 (the bottom, which contains all N vectors) and probabilistically to higher layers with decreasing probability. A small fraction of nodes appear in layer L-1, even fewer in layer L-2, etc.
3. For each node at each layer, connect it to M nearest neighbors (M is a build-time parameter, typically M = 16–32).

**Query phase (online):**

1. Query q arrives. Start at the entry node in the top layer.
2. Greedily hop to the neighbor closest to q. Repeat until no neighbor is closer.
3. Descend to the next layer and repeat.
4. At the bottom layer, run a more thorough search: keep a priority queue of the efSearch closest candidates, explore their neighbors, and update the queue until you have stable nearest neighbors.
5. Return the top-k from the queue.

### The knobs

- **M (build time):** Number of edges per node. Higher M = better recall but more memory and slower build. Typical: 16–32.
- **efSearch (query time):** Size of the candidate priority queue at the bottom layer. Higher efSearch = higher recall but slower. This is the recall dial, analogous to nprobe in IVF.

### Why it is O(log N)

Each layer has geometrically fewer nodes. At the top layer, you might have O(1) nodes. Each descent roughly halves the distance to the query. You make O(log N) hops to reach the bottom layer, and at each hop, you check O(M) neighbors, so the total cost is O(M × log N). In practice, M is small, so this is much faster than O(N).

### Worked example (N=1M, M=32, efSearch=100)

- Top layer: 10 nodes, long jumps
- Layer 1: 100 nodes
- Layer 0: 1,000,000 nodes

Query arrives. Start at top layer, make ~3 hops to get close. Descend, make ~10 hops in layer 1. Descend, explore 100 candidates in layer 0. Total comparisons: ~3 + 10 + 100 × 32 = ~3,200. Compare to brute-force: 1,000,000. Speedup: 300×.

**Memory:** HNSW stores the graph structure (edges), so memory is O(N × M × size_of_ID). For M=32, 4-byte IDs, that is 128 bytes of graph overhead per node, plus the original D × 4 bytes for the vectors. So HNSW uses more memory than IVF+PQ but less than brute-force.

### Common misunderstandings

- **"HNSW always beats IVF."** Not at massive scale. HNSW memory scales with N × M, so at 100M+ vectors, it uses too much RAM. IVF+PQ compresses better.
- **"efSearch is set at build time."** No. efSearch is a query-time parameter, like nprobe.
- **"HNSW is a neural network."** No. It is a graph data structure, not machine learning.

### Exam angle

If a question says "requires 100% recall," the answer is IndexFlatIP (exact brute-force). If it says "can tolerate slight recall loss, fits in RAM, millions of vectors," HNSW is a good choice. If it says "massive scale, strict RAM limits," IVF+PQ wins.

---

## Concept 6: Choosing the Right Index (Decision Tree)

### The framework

Use this flowchart to decide:

1. **Do you need 100% recall?** → IndexFlatIP (exact, brute-force). Fast enough up to ~1M vectors.
2. **Millions of vectors, can tolerate small recall loss, fits in RAM?** → HNSW or IVFFlat. HNSW is simpler (no training). IVFFlat is faster if you tune nlist and nprobe.
3. **Tens of millions, strict memory limits?** → IVF+PQ. This is the sample paper scenario.
4. **Billions of vectors?** → IVF+PQ with distributed sharding (beyond this course).

### Training requirements

- **IVFFlat / IVF+PQ require training:** You need at least 30–40× nlist sample vectors to run k-means. If your dataset is too small, IVF fails.
- **HNSW requires no training:** You can add vectors incrementally.

### Query-time knobs

- **IVF:** nprobe (higher = more recall, slower)
- **HNSW:** efSearch (higher = more recall, slower)

Both are adjustable per query without rebuilding.

### Exam angle

Sample paper Q19: "50M vectors, strict RAM, can tolerate slight recall drop." Answer: B (IVF+PQ). The other options: A (IndexFlatIP) is brute-force, too slow. C (Exact Cross-Encoder) is not even an index. D (HNSW) uses too much RAM at 50M scale.

---

## Concept 7: ColBERT (Late Interaction) — Token-Level Embeddings for Near-Cross-Encoder Accuracy

### The problem it solves

Standard bi-encoders (like Sentence-BERT) compress an entire document into one vector. This loses fine-grained token-level information. Cross-encoders are much more accurate because they compute attention between every query token and every document token, but they are too slow to run at scale (you need N forward passes for N documents). ColBERT sits in the middle: it stores one vector per token (like a cross-encoder's granularity) but computes similarity at query time with a cheap operator (like a bi-encoder).

### The intuition (analogy first)

Imagine you are a teacher grading essays. A bi-encoder approach would be: read the essay once, summarize it in your head as "good" or "bad," and assign a score. A cross-encoder would be: read the essay and the rubric side-by-side, comparing every sentence of the essay to every criterion. ColBERT is like: read the essay, highlight every important sentence, then at grading time, for each rubric criterion, find the best-matching highlighted sentence and score that. You pre-process the essay (highlight sentences = store token vectors), but you defer the detailed comparison until grading time (MaxSim operator).

### How it works (step by step)

**Indexing phase:**

1. Encode each document with BERT, but instead of pooling to one vector, keep one vector per token. A 100-token document becomes 100 vectors.
2. Store all token vectors in the index (not document vectors). The index size is now proportional to the total number of tokens, not documents.

**Query phase:**

1. Encode the query with BERT, keeping one vector per query token.
2. For each query token q_i, find its best match among all tokens in document d: max_j (q_i · d_j). This is called MaxSim.
3. Sum the MaxSim scores across all query tokens: score(q, d) = Σ_i max_j (q_i · d_j).
4. Rank documents by score.

### The MaxSim operator

MaxSim(q, d) = Σ_{i=1 to |q|} max_{j=1 to |d|} (q_i · d_j)

For each query token, find the single best-matching document token, then sum those maxima. This captures fine-grained matching (like a cross-encoder) but requires only dot products (like a bi-encoder), and crucially, the document token vectors are precomputed.

### Worked example (query = "payment error," doc = "The payment gateway returned an HTTP 500 error.")

- Query tokens: ["payment", "error"]
- Doc tokens: ["The", "payment", "gateway", "returned", "an", "HTTP", "500", "error", "."]

For "payment":
- q_payment · d_payment = 0.95 (high match)
- q_payment · d_error = 0.2
- q_payment · d_gateway = 0.4
- ... (check all 9 tokens)
- Best match: 0.95

For "error":
- q_error · d_error = 0.93 (high match)
- q_error · d_500 = 0.5
- ... (check all 9 tokens)
- Best match: 0.93

Total score: 0.95 + 0.93 = 1.88

Compare to a bi-encoder that would compress the entire document into one vector and compute a single dot product (losing the fact that "payment" and "error" both match strongly).

### The trade-off

**Pros:**
- Much higher accuracy than bi-encoders, nearing cross-encoder quality
- Still allows precomputation (token vectors are indexed offline)

**Cons:**
- Index size explodes: instead of N document vectors, you store T token vectors (where T = total token count across all docs, typically 100× to 1000× larger than N)
- Query time is slower than bi-encoders because you need to run MaxSim over many token pairs

### When to use ColBERT

When accuracy is critical (e.g., legal, medical retrieval) and you can afford the storage cost. Not suitable for billion-scale unless you have massive RAM or distributed infra.

### Common misunderstandings

- **"ColBERT is a cross-encoder."** No. Cross-encoders jointly encode query+doc. ColBERT encodes them separately (bi-encoder style) but keeps token-level granularity.
- **"ColBERT is always better."** Not if storage is limited. For large-scale production, bi-encoders + cross-encoder reranking is more common.

### Exam angle

Sample paper Q20: "Enterprise needs near cross-encoder accuracy but cannot afford joint attention at query time. Willing to provision more storage." Answer: D (ColBERT). The key phrase is "willing to provision significantly more storage" — that signals ColBERT's trade-off.

---

## Concept 8: Query Routing — Deciding Which Data Source to Hit BEFORE Retrieval

### The problem it solves

In a production RAG system, you might have multiple vector databases (HR docs, codebase, live logs) or multiple APIs (Splunk, GitHub). If you search all of them for every query, you waste compute and increase latency. Query routing decides, based on the query intent, which single source to hit.

### The intuition (analogy first)

Imagine you walk into a hospital and say, "I need help." The receptionist (the router) asks, "Is this an emergency, a prescription refill, or a billing question?" Based on your answer, they send you to the ER, the pharmacy, or the billing desk. They do not send you to all three. Query routing is the receptionist.

### The four strategies (from cheap to expensive)

1. **Rule-based (logical):** Hardcoded if/else or regex. "If query contains 'PTO' or 'leave', route to HR_DB."
2. **Embedding-based (semantic router):** Build a profile of 20–50 sample queries per database, embed them offline. At query time, embed the live query, find the nearest profile cluster by cosine similarity, route there. No LLM, fast, offline-capable.
3. **Classifier-based:** Train a small ML model (e.g., a small BERT or Random Forest) on query→database label pairs. At query time, classify the query.
4. **LLM-based:** Send the query to an LLM with a prompt: "Given this query, which database should I search: HR, Codebase, or Logs? Respond with JSON." Parse the JSON.

### Strategy 1: Rule-based (deterministic)

**How it works:**

```python
if any(keyword in query.lower() for keyword in ["pto", "leave", "payroll"]):
    return "HR_Database"
elif any(keyword in query.lower() for keyword in ["commit", "PR", "code"]):
    return "Codebase_Database"
else:
    return "Logs_Database"
```

**Pros:** Zero cost, instant, deterministic.

**Cons:** Brittle. If user types "I need time off" (no exact keyword "leave"), the rule fails.

### Strategy 2: Embedding-based (THIS IS SAMPLE PAPER Q23)

**How it works:**

1. **Offline:** For each database, collect 20–50 representative queries. For example, for HR_DB: ["What is the PTO policy?", "How much leave do I have?", "Who is my manager?"]. Embed all these queries and store them labeled by database.
2. **Query time:** Embed the incoming query. Compute cosine similarity to all stored sample embeddings. Find the database whose samples have the highest average similarity. Route there.

**Pros:**
- No LLM call (fast, cheap)
- Works offline (no API dependency)
- Semantic understanding (handles synonyms and paraphrasing)

**Cons:**
- Requires manual curation of sample queries
- May fail on highly ambiguous queries

**Worked example:**

HR_DB samples (embedded): ["What is the PTO policy?", "How do I request leave?", ...]
Codebase_DB samples (embedded): ["How does the auth module work?", "Where is the payment logic?", ...]

Query: "I'm feeling burnt out and need a break next Tuesday."

- Compute similarity to HR_DB samples: [0.72, 0.68, 0.70] → avg = 0.70
- Compute similarity to Codebase_DB samples: [0.12, 0.15, 0.10] → avg = 0.12

Route to HR_DB.

**Exam angle:** Sample paper Q23 describes this exact approach. The question will test whether you understand that embedding-based routing uses offline profiles and cosine similarity, not an LLM.

### Strategy 3: Classifier-based

Train a lightweight model (e.g., a small BERT classifier or Naive Bayes) on historical query logs with labels. At query time, run one forward pass, get probabilities for each database, route to the highest.

**Pros:** Fast, accurate, no LLM cost.

**Cons:** Requires training data and retraining when new databases are added.

### Strategy 4: LLM-based

**How it works:**

Prompt: "You are a query router. The user asked: '{query}'. Which database should I search: HR_Docs, Codebase, or Live_Logs? Respond with JSON: {\"database\": \"...\", \"confidence\": ...}"

Parse the LLM's response, route accordingly.

**Pros:** Most accurate, handles ambiguous cases.

**Cons:** Adds latency (LLM call), costs money, requires API.

### Hybrid waterfall (production best practice)

Try cheap methods first, escalate only if confidence is low:

1. Try rule-based. If match → route.
2. If no match, try embedding-based. If similarity > threshold → route.
3. If still ambiguous, call LLM.

This minimizes cost and latency while handling edge cases.

### Exam angle

Sample paper Q22: "Balance cost and accuracy, avoid expensive LLM for every query, but rule-based is too brittle." Answer: A (Hybrid waterfall: Rule → Embedding → LLM). This tests whether you understand the sequential escalation pattern.

---

## Concept 9: Request Shaping — Transforming Queries Before Retrieval

### The problem it solves

Users often ask vague, ambiguous, or context-dependent questions. "OOMKilled" is not a good query for a vector search. Request shaping transforms these into precise, standalone queries.

### The four techniques

1. **Rewriting:** Use an LLM to expand a shorthand query into a full question. "OOMKilled" → "What causes the OOMKilled container exit code in Kubernetes, and what are the remediation steps?"
2. **Follow-up injection:** If the user says "Does it apply to contractors?" in a multi-turn conversation, inject the previous context to make it standalone: "Does the standard parental leave policy apply to contract employees?"
3. **Multi-query retrieval:** For a complex question, generate multiple related queries and retrieve for all in parallel. "Did the latency spike happen in Mumbai and US-East?" → Q1: "Mumbai latency spike" + Q2: "US-East latency spike".
4. **Step-back prompting:** For a hyper-specific question, generate a broader foundational query first. "Why can't I pull switch association data for rogue units on Floor 3?" → "What is the overall architecture of our network troubleshooting system?"

### Worked example: Step-back prompting (SAMPLE PAPER Q21)

**User query:** "Why can't I pull the switch association data for the rogue units on Floor 3?"

This is so specific that the vector DB has no matching keywords. Retrieval returns zero results.

**Step-back query (LLM-generated):** "What is the overall architecture of our network troubleshooting system, and what data is collected from reporting access points versus rogue units?"

This retrieves the foundational document that explains: "Switch association data is only available for reporting access points, not rogue units."

Now the system can answer the user: "You cannot pull switch association data for rogue units because our system only collects that data for reporting APs, not rogues."

### Exam angle

Sample paper Q21 asks: "User gets 'no results' on hyper-specific queries. Which query transformation fixes this?" Answer: C (Step-Back Prompting). The key is recognizing that the DB has the answer, just not at the same level of specificity as the query.

---

## Concept 10: RAG Security — The Retrieval Gateway and RBAC Filters

### The problem it solves

In an enterprise RAG system, not all users should have access to all documents. A junior engineer asking "What are the latest salary bands?" should get ZERO results, not an LLM refusal. Security must happen at the retrieval layer, not the generation layer.

### The threat model

**Scenario:** A junior engineer submits a query that, if executed naively, would retrieve sensitive HR documents (executive salaries, layoff plans). The LLM then summarizes those documents. Even if the LLM is instructed to refuse to answer, the sensitive data has already been:
1. Retrieved into memory
2. Logged in the retrieval gateway
3. Sent to the LLM's context window

This is a data leak, even if the final answer is "I cannot answer that."

### The solution: Retrieval gateway with RBAC metadata filters

**Architecture:**

1. User submits query.
2. Gateway reads user's identity token (from SSO, AD, OAuth).
3. Gateway reads user's role or clearance level from the identity service.
4. Gateway injects a metadata filter into the vector DB query BEFORE retrieval happens.
5. Vector DB physically cannot return documents that do not match the filter.
6. LLM receives only authorized documents.

**Worked example:**

User: Junior Engineer (clearance_level = 1)

Query: "What are the latest salary bands?"

Gateway constructs filter:

```json
{
  "where": {
    "clearance_level": {"$lte": 1}
  }
}
```

Vector DB searches only documents tagged with clearance_level ≤ 1 (public docs). Sensitive salary docs are tagged with clearance_level = 3. They are never retrieved. The DB returns zero results. The LLM says: "I found no documents matching your query."

**Why this is the only correct approach:**

- **Option B (post-filter with cross-encoder):** Too late. The sensitive docs have already been retrieved into memory and logs.
- **Option C (LLM router classifies query as sensitive):** The router guesses intent but has no document-level control. What if the user's query is innocent but the retrieved docs happen to be sensitive?
- **Option D (instruct LLM to refuse if sensitive data appears):** This is guidance, not enforcement. The data is already in the context window and logs.

Only the gateway injecting a metadata filter at query construction time guarantees that unauthorized docs are never touched.

### The retrieval gateway's four responsibilities

1. **Access control (AuthN/AuthZ):** Inject RBAC filters based on user identity.
2. **Quota management (FinOps):** Enforce rate limits and budget caps to prevent runaway costs.
3. **Semantic caching:** If a similar query was asked recently, serve the cached answer.
4. **Observability:** Log every query, retrieval result, and citation for auditing.

### Exam angle

Sample paper Q24: "Enforce least privilege so junior engineer cannot access exec salary data. Where and how?" Answer: A (Retrieval Gateway, injecting metadata filter into vector DB query BEFORE retrieval). This is the highest-yield security concept in the entire lecture.

---

## Concept 11: Semantic Caching — Serving Duplicate Queries from Cache

### The problem it solves

If 200 engineers ask "Is US-East down?" within 5 minutes, you do not want to run 200 vector searches and 200 LLM calls. Semantic caching detects that these queries mean the same thing and serves the first answer 199 times.

### How it works

1. Gateway receives query q.
2. Gateway embeds q.
3. Gateway searches the cache (a fast in-memory store like Redis) for cached queries.
4. For each cached query q_cache, compute cosine similarity.
5. If max similarity ≥ threshold τ, serve the cached answer.
6. Else, run the full pipeline and cache the answer.

### The threshold τ: Bias HIGH

**The asymmetry:**

- **Cache miss (false negative):** You run the full pipeline. Cost: normal API fees + latency. Not great, but acceptable.
- **False hit (false positive):** You serve the wrong cached answer. Cost: user gets incorrect information, loses trust, makes bad decisions.

Because the cost of a false hit is much higher than the cost of a miss, you should bias τ high (e.g., 0.95 or 0.98), accepting a lower cache hit rate to guarantee accuracy.

### Worked example

Cached query: "Is the US-East region down?"
New query: "Is US-East operational?"

Cosine similarity: 0.92

If τ = 0.90 → cache hit (serve cached answer)
If τ = 0.95 → cache miss (run full pipeline)

With τ = 0.95, you are more conservative, ensuring the queries really mean the same thing.

### Exam angle

Sample paper Q25: "Configure τ for semantic caching. False hit damages trust, cache miss only costs latency. What is optimal?" Answer: B (Bias τ high, accept lower hit rate, guarantee accuracy on cached responses). This tests the asymmetry principle.

---

## Concept 12: Silent RAG Bugs — Three Failures That Give No Errors

### Bug 1: Empty chunks from scanned PDFs

**What happens:** A scanned PDF has no text layer (it is just images). pypdf or similar parsers return empty strings. These get embedded (the embedding of "" is some arbitrary vector), indexed, and retrieved. The LLM sees empty context and hallucinates.

**Fix:** Always probe for text layer. If absent, route to OCR.

### Bug 2: Asymmetric embeddings used without prefixes

**What happens:** Some embedding models (e.g., certain BGE or E5 models) are trained asymmetrically: query embeddings and document embeddings use different prompt prefixes. If you forget to add "query: " or "passage: " prefixes, the embeddings are misaligned and retrieval fails.

**Fix:** Read the model card. If it says "asymmetric," prepend the required prefix.

### Bug 3: Wrong distance metric

**What happens:** Your model was trained with cosine similarity, but your vector DB is configured to use L2 (Euclidean distance). The rankings are wrong, recall drops.

**Fix:** Match the metric to the model's training objective. Most sentence-transformers use cosine. Check the model card.

### Why these are dangerous

None of them throw an error. The system runs, returns results, and the LLM generates answers. But the answers are subtly wrong. These bugs are discovered weeks later when users complain about quality.

---

## Putting it together

Here is how the concepts stack:

1. **Scale drives the need for ANN.** Brute-force fails at 10M+ vectors.
2. **IVF prunes search scope** by clustering. nprobe is the recall dial.
3. **PQ compresses memory** by quantizing sub-vectors. The compression ratio is typically 32× to 256×.
4. **IVF+PQ combines both** for billion-scale systems with tight RAM.
5. **HNSW is an alternative** using graph navigation. Faster than IVF at low-to-medium scale, but uses more RAM at massive scale.
6. **ColBERT stores token vectors** for near-cross-encoder accuracy, at the cost of index size.
7. **Query routing decides which data source to hit.** Embedding-based routing uses offline sample profiles and cosine similarity (sample paper Q23).
8. **Request shaping transforms queries.** Step-back prompting handles hyper-specific queries (sample paper Q21).
9. **The retrieval gateway enforces security** by injecting RBAC metadata filters into the DB query BEFORE retrieval (sample paper Q24).
10. **Semantic caching serves duplicate queries.** Bias the threshold τ high to avoid false hits (sample paper Q25).

When designing a production RAG system, you start with ChromaDB for prototyping (up to ~1M vectors), graduate to FAISS with HNSW for medium scale (1–10M), then move to IVF+PQ for massive scale (50M+). You add query routing to reduce latency, request shaping to improve recall, a retrieval gateway to enforce security, and semantic caching to cut costs.

---

## Check yourself (5 questions)

**Q1:** You have 50 million vectors, D=1024, m=16, k=256. How much memory does PQ use? What is the compression ratio vs. raw storage?

<details>
<summary>Answer</summary>
PQ storage: N × m = 50,000,000 × 16 bytes = 800 MB.
Raw storage: N × D × 4 = 50,000,000 × 1024 × 4 = 204,800 MB = 200 GB.
Compression ratio: 200 GB / 800 MB = 256×.
</details>

**Q2:** You configure IVF with nlist=1000, nprobe=10. Your dataset has 10M vectors. How many vectors do you compare at query time?

<details>
<summary>Answer</summary>
nprobe × (N / nlist) = 10 × (10,000,000 / 1000) = 10 × 10,000 = 100,000 vectors.
</details>

**Q3:** Your RAG system uses embedding-based query routing. What is stored offline, and what happens at query time?

<details>
<summary>Answer</summary>
Offline: Embed 20–50 sample queries per database and store them labeled by DB.
Query time: Embed the incoming query, compute cosine similarity to all samples, route to the DB with highest average similarity.
No LLM call, no API dependency.
</details>

**Q4:** A junior engineer asks "What are the CTO's salary details?" How does the retrieval gateway prevent leakage, and why is post-filtering insufficient?

<details>
<summary>Answer</summary>
The gateway reads the engineer's role (e.g., clearance_level=1), injects a metadata filter (where clearance_level ≤ 1) into the vector DB query BEFORE retrieval. The DB physically cannot return docs with clearance_level=3.
Post-filtering is insufficient because the sensitive docs have already been retrieved into memory and logs.
</details>

**Q5:** You set semantic caching threshold τ=0.85. Your cache hit rate is 40%, but users report occasional wrong answers. What should you do?

<details>
<summary>Answer</summary>
Increase τ to 0.95 or higher. The low threshold is causing false hits (serving cached answers to queries that are similar but not identical). Accept a lower hit rate (maybe 20%) to guarantee accuracy. The cost of a false hit (wrong answer) is much higher than the cost of a miss (extra API call).
</details>

---

## Quick reference (for revision)

**IVF:** Cluster into nlist buckets, search nprobe nearest. Reduces scope from O(N) to O(nprobe × N/nlist). Recall dial = nprobe.

**PQ:** Split vector into m sub-vectors, quantize each to 1 byte. Compression = D×4 / m. Query builds LUT (m×k entries), scores DB vectors with m lookups. Typical compression: 32× to 256×.

**IVF+PQ:** IVF prunes scope, PQ compresses memory. Store PQ codes of residuals. Industry standard for 50M+ vectors.

**HNSW:** Multi-layer graph, greedy descent. O(log N) search. Knobs: M (build), efSearch (query, recall dial). Faster than IVF at <10M scale, but more RAM.

**ColBERT:** Token-level embeddings, MaxSim operator. Near cross-encoder accuracy, bi-encoder precompute. Index size = #tokens, not #docs.

**Query routing strategies:**
- Rule-based: regex, instant, brittle
- Embedding-based: offline sample profiles, cosine match, no LLM (SAMPLE PAPER Q23)
- Classifier: trained ML model
- LLM: prompt for JSON, accurate but slow
- Hybrid waterfall: Rule → Embed → LLM

**Request shaping:**
- Rewrite: expand shorthand
- Follow-up: inject context
- Multi-query: generate parallel queries
- Step-back: generate broader foundational query (SAMPLE PAPER Q21)

**Retrieval gateway (SAMPLE PAPER Q24):**
- Inject RBAC metadata filter BEFORE retrieval
- Enforces security at DB layer, not LLM layer

**Semantic caching:**
- Embed query, cosine match to cache, serve if similarity ≥ τ
- Bias τ HIGH (0.95+) to avoid false hits (SAMPLE PAPER Q25)

**Silent bugs:**
- Scanned PDFs → empty chunks → hallucinations
- Asymmetric models → missing prefixes → low recall
- Wrong distance metric → misaligned rankings

**FAISS index decision tree:**
- <1M or need 100% recall → IndexFlatIP
- 1–10M, tolerate slight recall loss → HNSW or IVFFlat
- 10M+, strict RAM → IVF+PQ

**Sample paper questions:**
- Q19: 50M vectors, strict RAM → IVF+PQ
- Q20: Near cross-encoder accuracy, cannot afford joint attention, willing to provision storage → ColBERT
- Q21: Hyper-specific query, no results, DB has underlying rules → Step-back prompting
- Q22: Balance cost and accuracy, rule-based too brittle → Hybrid waterfall
- Q23: Avoid expensive LLM router, semantic understanding → Embedding-based routing
- Q24: Enforce least privilege, prevent junior engineer from accessing exec salary → Gateway injects metadata filter BEFORE retrieval
- Q25: Configure semantic cache threshold, false hit damages trust → Bias τ high
