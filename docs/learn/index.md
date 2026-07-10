# Learn — Deep Guides

These are the **primary reading materials**. Each guide takes a single lecture and explains it in plain English — the problem it solves, an intuition (usually with an analogy), a worked example, the formula with meanings, common misunderstandings, and how it shows up in the exam.

Every guide ends with 5 practice MCQs and a quick-reference summary.

!!! tip "How to read one"
    Budget 20–35 minutes per guide. Read straight through, then attempt the 5 questions at the end WITH THE ANSWERS COVERED. Whatever you got wrong or guessed on — reread just that concept block before moving to the next lecture.

## Recommended reading order

The lectures build on each other. If you have time, read them in order. If you're time-crunched, start with Lecture 11 (vector indexing) — it alone accounts for 6 of the 24 sample paper questions.

| # | Guide | What you'll learn | Approx. read time |
|---|---|---|---|
| 1 | [ML foundations](01-ml-foundations.md) | Precision, recall, F1, confusion matrix, ML pipeline stages | 25 min |
| 2 | [Neural networks & word embeddings](02-neural-networks.md) | MLP → CNN → RNN → LSTM → Word2Vec → architecture families | 30 min |
| 3 | [Transformers & attention](03-transformers.md) | Self-attention, Q/K/V, positional encoding, multi-head | 30 min |
| 4 | [LLM decoding & APIs](04-llm-decoding-apis.md) | Greedy/top-k/top-p/temperature, OpenAI Chat Completions, HF Inference vs Ollama | 25 min |
| 5 | [Structured outputs & evaluation](05-structured-outputs-evaluation.md) | JSON schema, LogitsProcessor, BLEU / ROUGE / METEOR / BERTScore | 30 min |
| 6 | [AI safety & CIA triad](06-ai-safety.md) | Confidentiality/Integrity/Availability, prompt injection, jailbreaks | 25 min |
| 7 | [Advanced prompting](07-advanced-prompting.md) | Zero-shot / Few-shot / CoT / Self-Consistency / ReAct / Tree-of-Thoughts | 30 min |
| 8 | [Prompt security & APE](08-prompt-security-ape.md) | Levenshtein distance, log-probability scoring, robustness | 25 min |
| 9 | [RAG fundamentals](09-rag-fundamentals.md) | Retrieve → augment → generate, naive vs advanced vs agentic | 25 min |
| 10 | [Dense retrieval & rerankers](10-dense-retrieval.md) | Bi-encoder vs cross-encoder, ColBERT, two-stage retrieval | 25 min |
| 11 | [Vector indexing & PQ math](11-vector-indexing.md) | Product Quantization, IVF, HNSW, query routing, RBAC gateway | **35 min — highest yield** |
| 12 | [Agentic RAG](12-agentic-rag.md) | Observe-Think-Act loop, tool use, iterative retrieval | 25 min |

## What "learn" means here

Each guide follows the same pattern per concept:

1. **What problem this solves** — why anyone invented this
2. **Intuition** — usually an analogy from everyday life
3. **Worked example** — a small numeric or textual example you can follow
4. **Formula with meaning** — every symbol explained
5. **Common misunderstandings** — the traps students fall into
6. **Exam angle** — how this exact concept appears in questions

If a concept doesn't click on first pass, reread just the intuition and the worked example — they're the fastest path to a mental model.
