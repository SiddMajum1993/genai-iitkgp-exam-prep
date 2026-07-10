# Exam Cheat Sheet — GenAI & Agentic AI (upGrad × IIT Kharagpur)

**Exam date:** 18 July 2026 · **Format:** 24 MCQs × 4 marks · **No negative marking** · **Attempt all 24**

This is the last-day skim document. Every formula, table, and "why-not" the sample paper tested is on this page. Read start-to-finish twice on July 17.

---

## Section 1 — Evaluation Metrics & Model Architecture

### Confusion Matrix Cheat

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

- **Precision** = TP / (TP + FP) — "of alarms raised, how many real?"
- **Recall** = TP / (TP + FN) — "of real positives, how many caught?"
- **F1** = 2 · P · R / (P + R) — harmonic mean
- **Accuracy** = (TP + TN) / Total — MISLEADING on imbalanced data

**When to prioritise which:**
| Scenario | Metric | Why |
|---|---|---|
| Rare disease screening | **Recall** | Missing a positive (FN) is catastrophic |
| Spam / fraud filter | **Precision** | False alarm (FP) annoys or blocks legitimate user |
| Both errors equally bad | **F1** | Balanced harmonic mean |
| Balanced classes only | Accuracy | Otherwise misleading |

**Sample paper trap:** 9,900 healthy / 100 diseased. A model predicting "healthy" always gets 99% accuracy but 0% recall → useless. Answer = Recall.

### CNN vs MLP — the ONE structural difference

| | MLP first layer | CNN first layer |
|---|---|---|
| Connectivity | Every pixel → every neuron (global) | Each neuron → small local patch (LOCAL) |
| Params (224×224×3 image) | Millions | 27 (for a 3×3×3 filter) |

**Answer to "single most important structural difference":** LOCAL CONNECTIVITY. Not activation function, not channel count, not "matrix multiplication vs element-wise" (both use matmul).

### Word2Vec

- Learns from **co-occurrence / distributional hypothesis** (Firth 1957: "a word is known by the company it keeps")
- **NO** thesaurus, **NO** grammar parser, **NO** spelling — just raw co-occurrence
- Two variants:
  - **Skip-gram:** center word → predict context (better for rare words)
  - **CBOW:** context → predict center word (faster)
- `vec(king) − vec(man) + vec(woman) ≈ vec(queen)` because the vector offset between male/female royalty terms is a **learned gender direction** in embedding space
- **Polysemy problem:** one static vector per word; "bank" (finance) = "bank" (river). Fixed later by ELMo / BERT (contextual).

### LSTM output shapes (PyTorch)

```python
lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)
x = torch.randn(8, 15, 10)          # (batch=8, seq=15, input=10)
output, (hn, cn) = lstm(x)
# output.shape = (batch, seq, hidden)         = (8, 15, 32)  — all timesteps
# hn.shape     = (num_layers, batch, hidden)  = (2, 8, 32)   — final state per layer
# cn.shape     = (num_layers, batch, hidden)  = (2, 8, 32)   — final cell state per layer
```

**Traps:**
- `(8, 15, 32)` is `output` shape — not hn/cn.
- `(1, 8, 32)` would only be right if `num_layers=1`.
- `(2, 15, 32)` confuses seq_len with batch; seq_len does NOT appear in hn/cn.

### Architecture family — when to pick which

| Family | Examples | Attention pattern | Best for |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Bidirectional (every token sees full sentence) | Classification, NER, scoring, embedding |
| **Decoder-only** | GPT, LLaMA | Causal / masked (only past tokens) | Text generation, chat, conversational |
| **Encoder-Decoder** | T5, BART | Enc bidir + Dec causal + cross-attn | Translation, summarisation, seq2seq transformation |

**Sample paper trap:** NER (tag people/orgs/locations in legal docs) → **encoder-only** because every token needs full bidirectional context for classification, and BERT is purpose-built for this at lowest inference cost. Not enc-dec (NER is not seq2seq; output aligns 1:1 with input).

---

## Section 2 — LLM Decoding, APIs & Tooling

### Decoding strategies

| Strategy | Randomness | Use when |
|---|---|---|
| **Greedy / temperature=0** | Zero — deterministic | Legal, medical, code, anything needing identical output for identical input |
| **Top-k sampling** | Sample from top k tokens | Balance diversity + quality |
| **Top-p (nucleus)** | Smallest set whose cumulative prob ≥ p | Dynamically adjust candidate pool by confidence |
| **Temperature** | Rescale logits: `logits/T`. T→0 = greedy; T→∞ = uniform | T=0.7 balanced; T=1.5 wild |
| **Beam search** | Keep top-B partial sequences | Translation, summarisation |

**Sample paper trap:** "Legal contract auto-complete, must be deterministic" → **greedy / T≈0**. Top-p, top-k, and high-T all introduce randomness.

### Top-k ∩ Top-p worked example (memorise this pattern)

Probs: A=0.13, B=0.04, C=0.22, D=0.16, E=0.01, F=0.18, G=0.07, H=0.10, I=0.06, J=0.03. Params: k=2, p=0.95.

- **Top-k = 2** → the 2 highest: {C=0.22, F=0.18} → set of 2
- **Top-p = 0.95** → sort desc, add until cumulative ≥ 0.95:
  C 0.22 → +F 0.40 → +D 0.56 → +A 0.69 → +H 0.79 → +G 0.86 → +I 0.92 → +B **0.96 STOP** → set of 8 {C,F,D,A,H,G,I,B}
- **Intersection = {C, F} = 2 tokens**
- **Binding constraint = top-k** (it truncated more aggressively)

### OpenAI Chat Completions API roles

| Role | Purpose |
|---|---|
| `system` | Persona / behavior instructions |
| `user` | End-user input |
| `assistant` | Prior model outputs — used for **few-shot examples** |
| `tool` | Return **function/tool call results** to the model |
| `developer` | OpenAI-specific override (newer) |

**Parameters:**
- `max_tokens` — hard cap on output length
- `temperature`, `top_p` — decoding
- `stop` — stop sequences (halts on string match)
- `frequency_penalty` / `presence_penalty` — reduce repetition
- `tools` / `function schema` — define what tools exist (does NOT inject results)

**Not standard OpenAI:** `repetition_penalty`, `prompt caching` (that's Claude API).

**Sample paper Q8 trap:** To (i) inject tool results (ii) prime with few-shot (iii) cap length → `tool role | assistant role few-shot | max_tokens`.

### Model access — HF Inference API vs local vs Ollama

| Approach | Needs GPU/torch | Privacy | Speed | Model variety |
|---|---|---|---|---|
| **Local Transformers** | Yes | Full | Fast (once loaded) | Widest |
| **HF Hosted Inference API** | No | Data sent to HF | Slower + rate-limited | Wide, but hosted-model list only |
| **Ollama** | Needs local runtime | Full (local) | Fast | Curated subset |

**Sample paper trap:** No GPU, avoid PyTorch, needs Qwen3-32B → **HF Inference API**. Ollama also runs locally but installs its own runtime.

### LogitsProcessor — RegexMask (must know)

- `regex.match(pattern, string, partial=True)` — returns True if string is **consistent** with the pattern even if not yet complete (needed while generating tokens one at a time)
- `re.match` (built-in) — returns True only when the FULL pattern matches
- If you replace `regex.match(..., partial=True)` with `re.match(...)`, tokens that would legally extend the output get masked out prematurely → model stops generating valid prefixes

### Text-generation NLP metrics

| Metric | Formula core | What it captures | When it fits |
|---|---|---|---|
| **BLEU** | n-gram precision + brevity penalty | Precision of matched n-grams | Machine translation |
| **ROUGE-L** | Longest Common Subsequence (LCS) — recall-oriented | Recall + structural ordering | Summarisation (did we capture all key content?) |
| **METEOR** | Alignment with stem / synonym match + F-score | Handles morphology + synonyms | When "indemnify" vs "indemnified" matters |
| **BERTScore** | Cosine sim of contextual token embeddings | Semantic similarity | Synonyms and paraphrase acceptable |
| **Perplexity** | exp(cross-entropy) | Language-model fluency | LM benchmarking |

**Sample paper trap:** "Did the summary capture all clauses?" → **ROUGE-L** (recall-oriented, LCS captures ordering). BERTScore is a good complement but not the primary answer.

### LLM safety — CIA Triad + attack types

| Attack | Description | Primary CIA violation |
|---|---|---|
| **Indirect Prompt Injection** | Malicious instruction hidden in external data (PDF, webpage, email) processed by the LLM | **Confidentiality** (usually exfiltration) |
| **Direct Prompt Injection** | User types "ignore prior instructions..." | Integrity of behavior |
| **Jailbreaking** | Roleplay / rhetorical trick to bypass safety | Integrity |
| **Prompt Leaking** | Extract the hidden system prompt | Confidentiality |
| **Model Poisoning** | Corrupt training data / fine-tune | Integrity |
| **Membership Inference** | Determine if a record was in training data | Confidentiality (privacy) |
| **Adversarial Example** | Crafted input causes misclassification (GCG etc.) | Integrity |
| **DoS / Prompt-bomb** | Force expensive computation | Availability |

**Sample paper trap:** Hidden instruction in a PDF → assistant summarises → assistant silently emails data out. Attack = **Indirect Prompt Injection**; CIA = **Confidentiality**.

---

## Section 3 — Prompting Techniques & NLP Metrics

### How to identify prompting technique from a single prompt

| Technique | Give-away pattern |
|---|---|
| **Zero-shot** | No examples in prompt — just instruction |
| **Few-shot** (plain) | Prompt has 2-5 Q → A pairs; NO intermediate reasoning shown |
| **Few-shot CoT** | Examples include **intermediate reasoning step** ("Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.") before the final answer |
| **Zero-shot CoT** | Instruction "Let's think step by step" — no examples |
| **Auto-CoT** | Automatically generated CoT examples |
| **Self-Consistency** | NOT identifiable from a single prompt — inference-time strategy: sample multiple CoT completions, majority-vote |
| **ReAct** | Prompt shows Thought → Action → Observation cycles |
| **Tree-of-Thoughts** | Model explores multiple reasoning branches, backtracks |

**Sample paper Q13 trap:** Given a prompt with examples that include reasoning ("Adding all the odd numbers gives 39"), technique = **Few-shot CoT**. Not self-consistency (that needs multiple sampled completions to identify).

### Automatic Prompt Engineering — log-prob scoring

Log-probabilities are **negative**. **Value closest to zero = highest probability.**

| Score | exp(score) |
|---|---|
| −0.07 | ≈ 0.93 ✓ (winner) |
| −0.41 | ≈ 0.66 |
| −0.95 | ≈ 0.39 |
| −2.13 | ≈ 0.12 |

### Levenshtein / edit distance

- Cost = 1 per insertion, deletion, substitution
- **HEART → EARTH = 2**: delete leading H (→ EART), insert H at end (→ EARTH)
- Naive per-position substitution would give 4; optimal alignment shifts characters

### RAG — core mechanism

**Retrieval-Augmented Generation:** fetch external info at **inference time**, supply to generator as additional context, generator conditions on that context to produce the answer.

- NOT fine-tuning (that bakes knowledge into weights)
- NOT generate-then-verify (that reverses the order)
- NOT plain search (still has a generator)

### Naive RAG vs Agentic RAG

| | Naive / Linear RAG | Agentic RAG |
|---|---|---|
| Structure | Fixed pipeline: retrieve → generate | LLM-driven controller in a loop |
| Retrievals | One-shot | Multi-step, iterative |
| Routing | Static | Dynamic — picks tool/source per step |
| Correction | None | Can reflect on intermediate results |

**Sample paper Q18 answer:** "Agentic RAG can issue multiple retrieval steps, route across tools or sources, and iterate."

---

## Section 4 — Vector Indexing & RAG Security

### Product Quantization (PQ) — MEMORISE

**Given:** N vectors, D-dim float32, m sub-vectors, k centroids per codebook (typically k=256 → 8 bits = 1 byte per code).

| Quantity | Formula |
|---|---|
| Raw storage | D × 4 × N bytes |
| PQ code storage | m × 1 × N bytes |
| Compression ratio | (D × 4) / m |
| Per-query lookup table | m × k entries |
| Distance per DB vector | m lookups + (m−1) additions |

**Worked example — sample paper Q19–Q22:**
N = 10M, D = 1024, m = 16, k = 256.

| Q | Answer | Calculation |
|---|---|---|
| Raw storage | **40.96 GB** | 1024 × 4 × 10M = 4,096 × 10M bytes |
| PQ storage | **160 MB** | 16 × 10M = 160M bytes |
| Lookup table | **4,096 entries** | 16 × 256 |
| Lookups, additions per vector | **16, 15** | m and m−1 |

### ANN index families

| Index | Structure | Search cost | Memory | Use when |
|---|---|---|---|---|
| **IndexFlatIP / L2** | Brute force | O(N·D) | Full | ≤1M vectors, need 100% recall |
| **IVF (Flat)** | Cluster into `nlist` buckets, search `nprobe` closest | O(nprobe · N/nlist · D) | Full | Millions, RAM OK |
| **IVF + PQ** | Cluster + compress | Same as IVF but tiny memory | **Compressed** | 50M+, strict RAM |
| **HNSW** | Hierarchical skip-graph | O(log N) | O(N · M) | Moderate scale, low latency, memory OK |
| **ColBERT (late interaction)** | Token-level embeds + MaxSim | Slow but accurate | Very large | Near cross-encoder quality, single-stage retrieval |

### Query routing

| Strategy | Setup | Match | LLM in loop? | When |
|---|---|---|---|---|
| **Rule-based** | Hardcoded regex/keywords | String match | No | Stable, keyword-driven |
| **Embedding-based** | 20–50 sample queries per DB, embed offline | Cosine sim of live query to profiles | **No** | Semantic, offline-capable, no LLM cost |
| **Classifier-based** | Train small model on query→label | Softmax | No | High-traffic stable intents |
| **LLM-based** | Prompt LLM with route options, ask for JSON | LLM reasoning | Yes | Ambiguous queries, willing to pay |
| **Hybrid waterfall** | Rule → Embedding → LLM | Escalate on low confidence | Sometimes | Best-practice production |

**Sample paper Q23 answer:** Embedding-based = build profiles of 20–50 sample questions per DB; embed query; cosine match. No LLM. Offline-capable.

### RAG security — where each defense lives

**Enforce Least Privilege / RBAC AT THE RETRIEVAL GATEWAY:**
1. User's identity token (SSO/AD) hits the gateway
2. Gateway reads user's role / clearance
3. Gateway **injects RBAC metadata filter into the Vector DB query BEFORE retrieval**
4. Unauthorized docs are physically never returned

**Why the other layers are wrong:**
| Layer | Why it fails |
|---|---|
| System prompt refuses if sensitive data present | Guidance, not enforcement; context already retrieved |
| Post-filter with cross-encoder | Retrieved everything → data already in logs/memory |
| LLM router classifies query as "sensitive" | Guesses intent; no document-level control |

### Retrieval Gateway responsibilities (beyond auth)

- **AuthN + AuthZ** (RBAC metadata filter)
- **Quota / FinOps** (throttle, route to cheaper model)
- **Semantic caching** (embed query; if cos(q, cached_q) ≥ τ → serve cached answer)
  - **τ should be HIGH:** false-hit (wrong answer) >> cache miss (just $)
- **Observability** (log every query, chunks, latency)

### Silent RAG bugs

- Scanned PDF has no text layer → `pypdf` returns empty strings → **probe first, OCR fallback**
- Asymmetric embedding model used without `query:` / `passage:` prefixes → half the recall, no error
- Wrong distance metric (L2 on a cosine-trained model) → quietly bad ranking

---

## Sample paper answer key (memorise the pattern, not the letters)

| Q | Topic | Right answer keyword |
|---|---|---|
| Q1 | Metric for disease screening | **Recall** — missed positive is worst error |
| Q2 | Precision from confusion matrix | 80/(80+20) = 0.80 |
| Q3 | CNN vs MLP structural difference | **Local connectivity** |
| Q4 | Word2Vec king-queen | Learned from **co-occurrence** (distributional) |
| Q5 | LSTM shapes | hn=(2,8,32), cn=(2,8,32) |
| Q6 | NER architecture | **Encoder-only** (BERT) |
| Q7 | Deterministic legal generation | **Greedy / T≈0** |
| Q8 | OpenAI API for 3 needs | `tool` + assistant few-shot + `max_tokens` |
| Q9 | Weekend prototype, no GPU, Qwen3-32B | **HF Hosted Inference API** |
| Q10 | RegexMask `re.match` bug | Full-match kills partial prefixes |
| Q11 | Metric for capturing all clauses | **ROUGE-L** |
| Q12 | Hidden instruction in PDF | **Indirect Prompt Injection / Confidentiality** |
| Q13 | Prompt with reasoning example | **Few-shot CoT** |
| Q14 | Top-k ∩ Top-p intersection (probs above) | **2 tokens; top-k binding** |
| Q15 | Levenshtein HEART→EARTH | **2** |
| Q16 | APE best log-prob | **−0.07** (closest to zero) |
| Q17 | RAG core mechanism | Retrieve at inference → augment → generate |
| Q18 | Agentic RAG distinguisher | Multi-step, tool routing, iteration |
| Q19 | Raw storage 10M × 1024 float32 | **40.96 GB** |
| Q20 | PQ storage m=16 | **160 MB** |
| Q21 | Lookup table m×k | **4,096 entries** |
| Q22 | Lookups + additions per vector | **16, 15** |
| Q23 | Embedding-based routing setup | 20–50 profiles per DB, cosine match |
| Q24 | Least Privilege for salary docs | **Gateway injects RBAC metadata filter before retrieval** |

---

## 60-second exam-day tactics

1. **No negative marking → ANSWER ALL 24.** Never leave blank.
2. If unsure, eliminate 2 obviously wrong options → pick the more **specific** of the remaining two (correct options in this style tend to name the exact mechanism, not use vague language).
3. Beware options that are **true but off-topic** (e.g., BERTScore is real, but the question wanted recall).
4. Numerical questions: compute quickly on scratch, don't trust intuition.
5. If two options are grammatical opposites, one of them is usually right. Focus there.
6. Read the **stem twice**. Most traps are hidden in the scenario, not the options.
7. Time budget: 24 questions in 90 min = 3:45 per question. Do easy ones first, flag hard ones, return.
