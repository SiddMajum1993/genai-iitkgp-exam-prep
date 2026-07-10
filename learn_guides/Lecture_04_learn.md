# Lecture 4 — LLMs, Decoding, and Prompting APIs
**Estimated read time:** 25–30 min · **Difficulty:** Core

## What this lecture is about

You know what a Transformer is. Now you need to understand how those Transformers became Large Language Models—trained on hundreds of billions of tokens, aligned with human feedback, and served via APIs. This lecture covers the LLM training pipeline (pretraining, fine-tuning, RLHF), the major model families (BERT, GPT, T5, LLaMA, Mistral), and the critical practical topic: **how models actually generate text**. You will learn the difference between greedy decoding, beam search, temperature sampling, top-k, top-p, and when to use each. Then we dive into the OpenAI Chat Completions API—roles (system, user, assistant, tool), parameters (temperature, top_p, max_tokens), function calling, and how to constrain generation with custom logits processors. By the end, you will be able to call an LLM API intelligently and debug why your outputs are repetitive or incoherent.

This is high-yield exam material. Decoding strategies and API parameters show up in every MCQ set.

## Prerequisites

- **Transformer basics:** You should have read Lecture 3. You need to know what an encoder, decoder, self-attention, and autoregressive generation are.
- **Softmax:** Used everywhere in decoding. Converts logits to probabilities.
- **Logits:** The raw scores before softmax. For a vocabulary of 50,000 tokens, the model outputs 50,000 logits at each step.
- **Probability distributions:** Sampling from a distribution, cumulative probabilities.
- **APIs and REST calls:** Basic familiarity with HTTP requests, JSON payloads. We will show Python code using the OpenAI client.

---

## Concept 1: The LLM Training Pipeline (High-Level)

### The problem it solves

A randomly initialized Transformer is useless. It outputs gibberish. How do you turn it into ChatGPT? Three stages: pretraining, fine-tuning, and alignment (RLHF).

### Stage 1: Pretraining (Unsupervised)

**Objective:** Learn general language patterns from massive text corpora.

**Data:** Wikipedia, books, web crawl (e.g., Common Crawl). For GPT-3, ~300 billion tokens. For LLaMA 2, ~2 trillion tokens.

**Task:** Next-token prediction (for decoder-only models like GPT) or masked language modeling (for encoder-only models like BERT).

**Example:** Input: "The cat sat on the ___". Model predicts "mat" by computing probabilities over the entire vocabulary.

**Output:** A base model that understands syntax, grammar, facts, but has no instruction-following ability. If you prompt it with "Translate this to French," it might just continue the text instead of translating.

### Stage 2: Fine-Tuning (Supervised)

**Objective:** Teach the model to follow instructions and perform specific tasks.

**Data:** Human-written prompt-response pairs. Example:
```
Prompt: "Summarize the following article: [text]"
Response: "The article discusses..."
```

**Task:** Standard supervised learning. Maximize the log-probability of the response given the prompt.

**Output:** An instruction-following model. Now "Translate this to French" produces a translation.

This stage is also called Supervised Fine-Tuning (SFT).

### Stage 3: Alignment via RLHF (Reinforcement Learning from Human Feedback)

**Objective:** Align the model's outputs with human preferences (helpful, harmless, honest).

**Process:**
1. **Collect comparisons:** Humans rank multiple model outputs for the same prompt (e.g., "Which response is more helpful?").
2. **Train a reward model:** A separate model that predicts which output humans prefer. Input: prompt + response. Output: scalar score.
3. **Optimize the policy:** Use reinforcement learning (PPO—Proximal Policy Optimization) to maximize the reward model's score while keeping the model close to the SFT version (KL penalty prevents the model from drifting too far and generating nonsense to game the reward).

**Output:** A model that not only follows instructions but also aligns with human values. This is InstructGPT, ChatGPT, Claude, etc.

### Why this matters

When you see "GPT-4" or "LLaMA 2," you are looking at the result of all three stages. The exam will ask: "What is the purpose of RLHF?" Answer: **to align the model with human preferences using reinforcement learning on pairwise comparisons**. A trap is "to pretrain the model on more data" (that is stage 1, not RLHF).

### Common misunderstandings

- **"Fine-tuning and RLHF are the same."** No. Fine-tuning is supervised (you have ground-truth responses). RLHF is RL-based (you have preferences, not ground truth).
- **"Pretraining is task-specific."** No. Pretraining is general. Fine-tuning is where task-specific behavior emerges.
- **"RLHF guarantees the model never lies."** No. Alignment reduces harmful outputs but does not eliminate hallucinations or errors.

### Exam angle

"What is the primary objective of the pretraining stage for a decoder-only LLM like GPT?" Answer: **Next-token prediction on large-scale unsupervised text data**. Trap: "Instruction following" (that is fine-tuning), "Human preference alignment" (that is RLHF).

---

## Concept 2: Major Model Families (Recap and Context)

### BERT (2018, Google)

**Architecture:** Encoder-only.  
**Training:** Masked language modeling (MLM)—randomly mask 15% of tokens, predict them. Also next-sentence prediction (NSP), though NSP was dropped in later models.  
**Use case:** Classification, NER, QA. Bidirectional context.  
**Benchmark:** GLUE (General Language Understanding Evaluation). BERT-Large scored 82.1, crushing prior SOTA (71.0).  
**Why it mattered:** Established pretrain-then-finetune as the NLP paradigm.

### GPT-1 and GPT-2 (2018–2019, OpenAI)

**Architecture:** Decoder-only.  
**Training:** Next-token prediction on BookCorpus (GPT-1) and WebText (GPT-2, 40 GB).  
**GPT-2 scale:** 1.5B parameters.  
**Use case:** Text generation. Zero-shot task transfer via prompts.  
**Benchmark:** LAMBADA (long-range cloze). GPT-2 1.5B scored 63.2% zero-shot, surpassing prior SOTA (59.2%).  
**Why it mattered:** Showed that a sufficiently large decoder-only model can perform tasks from prompts alone, no fine-tuning needed.

### T5 and BART (2019–2020, Google and Facebook)

**Architecture:** Encoder-decoder.  
**Training:** T5 uses span corruption (mask random spans, predict them). BART uses varied denoising (delete tokens, shuffle, mask).  
**Use case:** Translation, summarization, any seq2seq task.  
**Benchmark:** SuperGLUE. T5-11B scored 88.9, close to human baseline (89.8).  
**Why it mattered:** Unified all NLP tasks as text-to-text. Strong seq2seq performance.

### GPT-3 (2020, OpenAI)

**Scale:** 175B parameters, ~300B training tokens.  
**Key insight:** In-context learning. Give the model a few examples in the prompt (few-shot), and it solves the task without any gradient updates.  
**Example:** Translate English to French. Provide 3 example pairs in the prompt, then ask for a 4th translation. GPT-3 does it.  
**Benchmark:** TriviaQA (closed-book QA). Zero-shot: 64.3%. 64-shot: 71.2% (matched prior fine-tuned SOTA).  
**Why it mattered:** Few-shot learning emerged as a capability at scale. The era of prompt engineering began.

### InstructGPT and ChatGPT (2022, OpenAI)

**Training:** GPT-3 + SFT + RLHF.  
**Result:** Labelers preferred the 1.3B InstructGPT over the 175B base GPT-3. Alignment matters more than scale.  
**Why it mattered:** ChatGPT (InstructGPT released publicly) became the first widely-used conversational AI. Defined what "using an LLM" feels like.

### LLaMA, Mistral, Falcon (2023–2024, Meta, Mistral AI, TII)

**LLaMA:** Open-weights models (7B–70B). Commercial license (LLaMA 2). Template for open models.  
**Mistral:** 7B dense and Mixtral 8×7B (Mixture of Experts). Apache-2.0 license. Strong quality-per-parameter.  
**Falcon:** Falcon-40B, Falcon-180B. Training data (RefinedWeb) released. Emphasis on transparency.  
**Why it mattered:** Open models caught up to closed models (Mistral 7B scored 60.1 on MMLU vs GPT-3.5's 70.0). A rich ecosystem of fine-tunes (Alpaca, Vicuna) made LLM customization accessible.

### GPT-4, Gemini, Claude (2023–2024)

**Multimodal:** GPT-4 accepts images. Gemini is trained on text, images, audio, video. Claude 3 and 4 offer long-context vision-language reasoning.  
**Scale:** GPT-4 is rumored to be ~1.76T parameters (not confirmed).  
**Why it matters:** Frontier models now handle multiple modalities natively, not as bolt-ons.

### o1, o3, DeepSeek-R1 (2024–2025)

**Reasoning models:** Spend inference-time compute on chain-of-thought search, optimized via RL on verifiable outcomes (math, code).  
**Why it matters:** First models to reliably solve competition-level math and programming problems. Inference-time scaling (more thinking = better answers) complements training-time scaling.

### Exam angle

"Which model architecture is best for text generation?" Answer: **Decoder-only (GPT, LLaMA)**. Trap: "Encoder-only (BERT)" (BERT cannot generate autoregressively).

"What was the key innovation of GPT-3?" Answer: **In-context learning (few-shot learning from prompts, no gradient updates)**. Trap: "Multimodal capability" (that is GPT-4), "Open-source weights" (GPT-3 is closed).

---

## Concept 3: Basic Prompting

### The problem it solves

How do you get the model to do what you want? You write a prompt. The prompt is your only interface.

### Elements of a Prompt

1. **Instruction:** "Translate the following text to French."
2. **Context:** Background information. Example: "You are a professional translator."
3. **Input data:** The actual text to process. Example: "The cat sat on the mat."
4. **Output indicator:** A hint for the model to start generating. Example: "Translation: "

### Example Prompt

```
Classify the text into neutral, negative, or positive.

Text: I think the food was okay.

Sentiment:
```

The model completes: "neutral."

### Why this works

Decoder-only models are trained to predict the next token. When you write "Sentiment:", the model sees it as "the next token should be a sentiment label." It generates "neutral" or "positive" or "negative" depending on the input.

### Common misunderstandings

- **"Prompts have a fixed format."** No. Prompts are arbitrary text. Different phrasings produce different outputs. This is why prompt engineering exists.
- **"Longer prompts are always better."** Not necessarily. Long prompts can confuse the model or waste context window. Be concise but clear.

### Exam angle

"What is the purpose of an output indicator in a prompt?" Answer: **To signal where the model should start generating the answer**. Trap: "To provide examples" (that is context/few-shot examples).

---

## Concept 4: Decoding Strategies — The Core of Generation

### The problem it solves

At each generation step, the model outputs a probability distribution over the entire vocabulary (e.g., 50,000 tokens). How do you pick the next token? This is the decoding problem.

Different strategies trade off quality, diversity, and determinism.

---

## Concept 5: Greedy Decoding

### The problem it solves

Greedy decoding is the simplest strategy: always pick the token with the highest probability (argmax).

### The intuition

At each step, look at the probability distribution and pick the top token. Example:
- Step 1: "The" (prob 0.4), "A" (prob 0.3), "This" (prob 0.2) → pick "The"
- Step 2: "cat" (prob 0.5), "dog" (prob 0.3), "house" (prob 0.1) → pick "cat"
- Continue until <EOS> or max length.

Deterministic. Same input → same output, always.

### The formula

next_token = argmax(P(token | context))

### Worked example

Prompt: "The capital of France is"

Logits (raw scores): {"Paris": 10.0, "London": 5.0, "Berlin": 3.0, ...}

Softmax → probabilities: {"Paris": 0.95, "London": 0.04, "Berlin": 0.01, ...}

Greedy: Pick "Paris" (highest prob).

Next step: "The capital of France is Paris"

Logits: {".": 8.0, ",": 6.0, "and": 2.0, ...}

Greedy: Pick "." (highest prob).

Output: "The capital of France is Paris."

### The problem with greedy decoding

**Repetition:** The model can get stuck in loops. Example: "I think I think I think I think..."

**Globally suboptimal:** Greedy picks the best token at each step, but that might not lead to the best overall sequence. Consider:
- Step 1: "The" (prob 0.6) vs "A" (prob 0.4). Greedy picks "The."
- But "A beautiful day" (prob of full sequence: 0.4 × 0.8 = 0.32) might be better than "The beautiful day" (prob 0.6 × 0.5 = 0.30).

Greedy is myopic. It does not look ahead.

### Common misunderstandings

- **"Greedy decoding is the best strategy."** No. It is the fastest and simplest, but often produces repetitive or boring text.
- **"Greedy decoding uses randomness."** No. It is deterministic. No sampling involved.

### Exam angle

"What is the main disadvantage of greedy decoding?" Answer: **It is myopic and can produce repetitive or globally suboptimal outputs**. Trap: "It is too slow" (greedy is the fastest method).

---

## Concept 6: Beam Search

### The problem it solves

Beam search is less myopic than greedy. It explores multiple candidate sequences in parallel and picks the best overall sequence.

### The intuition

Instead of keeping only the top-1 token at each step, keep the top-B candidates (beams). Expand each beam by generating the next token, score all resulting sequences, and prune to the top-B again.

Think of it as exploring a tree of possibilities, but pruning aggressively to keep only the most promising branches.

### The process (beam size B=2)

Step 1:
- Top-2 tokens: "The" (prob 0.6), "A" (prob 0.4)
- Keep both beams: ["The"], ["A"]

Step 2:
- From "The": top-2 continuations are "cat" (prob 0.5) and "dog" (prob 0.3)
  - "The cat" (cumulative prob: 0.6 × 0.5 = 0.30)
  - "The dog" (cumulative prob: 0.6 × 0.3 = 0.18)
- From "A": top-2 continuations are "beautiful" (prob 0.8) and "small" (prob 0.1)
  - "A beautiful" (cumulative prob: 0.4 × 0.8 = 0.32)
  - "A small" (cumulative prob: 0.4 × 0.1 = 0.04)

Now we have 4 candidates. Prune to top-2:
- "A beautiful" (0.32)
- "The cat" (0.30)

Continue until all beams hit <EOS> or max length. Return the beam with the highest cumulative probability.

### The formula

At each step, for each beam:
- Compute probabilities for all tokens.
- Expand: beam_prob × token_prob.
- Keep top-B candidates globally.

### The problem with beam search

**Deterministic:** Like greedy, beam search is deterministic. Same input → same output.

**Generic outputs:** Beam search tends to produce safe, boring text. It favors high-probability sequences, which are often generic ("I think that is a good idea.").

**Not suitable for open-ended generation:** For creative writing or chat, you want diversity. Beam search kills diversity.

### Worked example (lecture slide)

The lecture asks: "What is globally optimal?" and shows a tree. The optimal path might not be the greedily chosen path at each step. Beam search is better than greedy because it considers multiple paths, but it is still not guaranteed to find the global optimum (unless beam size = vocabulary size, which is intractable).

### Common misunderstandings

- **"Beam search always finds the optimal sequence."** No. It prunes most of the tree. Only exhaustive search guarantees optimality, which is exponential in length.
- **"Beam search is always better than greedy."** For tasks with a clear right answer (e.g., translation), yes. For open-ended generation, beam search is often worse (too generic).

### Exam angle

"Why is beam search preferred over greedy decoding for machine translation?" Answer: **Beam search explores multiple hypotheses and finds higher-quality translations**. Trap: "Beam search is faster" (false, it is slower than greedy).

"What is the main disadvantage of beam search for creative text generation?" Answer: **It produces generic, low-diversity outputs**. Trap: "It is too slow" (true but not the main conceptual issue).

---

## Concept 7: Temperature Sampling

### The problem it solves

Greedy and beam search are deterministic. Sometimes you want randomness—to get diverse outputs, to simulate human-like variation, or to explore creative completions. Sampling introduces randomness by picking tokens according to their probabilities (not always the top-1).

But raw sampling can produce too much randomness (incoherent text). Temperature controls the randomness.

### The intuition

Temperature is a dial that controls how "sharp" or "flat" the probability distribution is.

- **T → 0 (very cold):** Distribution becomes a spike. Almost deterministic (argmax). Output: "Paris is the capital of France."
- **T = 1 (default):** Use the raw distribution from the model. Balanced randomness.
- **T → ∞ (very hot):** Distribution becomes uniform. Every token is equally likely. Output: "Paris is cheese-dream cobblestone forever!"

Analogy: Imagine a die. Normally it has faces 1–6 with equal probability (1/6 each). Temperature is like adding weights:
- Cold: faces 1 and 2 are heavily weighted, others almost impossible.
- Default: all faces equally likely.
- Hot: faces are still equally likely, but now you also added extra faces with nonsense symbols.

### The formula

Logits are the raw scores before softmax. The model outputs logits z_i for each token i.

Normally, P(token_i) = softmax(z_i) = exp(z_i) / Σ_j exp(z_j)

With temperature T:

P(token_i) = exp(z_i / T) / Σ_j exp(z_j / T)

**Effect of T:**
- T < 1: Divide logits by a small number → larger logits get even larger relative to smaller ones → sharper distribution (more confident).
- T > 1: Divide logits by a large number → all logits get closer to each other → flatter distribution (less confident).

### Worked example

Logits: {"Paris": 6.0, "London": 4.0, "Berlin": 2.0}

**T = 1.0 (default):**
```
P("Paris") = exp(6.0) / (exp(6.0) + exp(4.0) + exp(2.0))
           = 403.4 / (403.4 + 54.6 + 7.4)
           = 403.4 / 465.4 ≈ 0.87
P("London") ≈ 0.12
P("Berlin") ≈ 0.02
```

**T = 0.5 (cold, more confident):**
```
Logits / T: {"Paris": 12.0, "London": 8.0, "Berlin": 4.0}
P("Paris") = exp(12.0) / (exp(12.0) + exp(8.0) + exp(4.0))
           ≈ 162754.8 / 165914.3 ≈ 0.98
P("London") ≈ 0.018
P("Berlin") ≈ 0.0003
```
Almost deterministic. "Paris" is picked ~98% of the time.

**T = 2.0 (hot, less confident):**
```
Logits / T: {"Paris": 3.0, "London": 2.0, "Berlin": 1.0}
P("Paris") = exp(3.0) / (exp(3.0) + exp(2.0) + exp(1.0))
           = 20.1 / (20.1 + 7.4 + 2.7)
           ≈ 20.1 / 30.2 ≈ 0.67
P("London") ≈ 0.24
P("Berlin") ≈ 0.09
```
Much flatter. "London" and "Berlin" are now plausible.

### When to use each temperature

- **T ≈ 0.3:** Factual tasks (QA, summarization). You want the most likely answer.
- **T ≈ 0.7–1.0:** Balanced (chat, general generation).
- **T ≈ 1.2–1.5:** Creative writing, brainstorming. You want variety.
- **T > 1.5:** Rarely useful. Output becomes incoherent.

### Common misunderstandings

- **"Temperature changes the model."** No. Temperature is applied at inference time, after the model outputs logits. It does not change the model's weights or training.
- **"Temperature = 0 is always best."** No. For factual tasks, yes. For creative tasks, you want higher temperature.
- **"Temperature is the only parameter I need to tune."** No. Temperature combines with top-k, top-p, and other parameters. Tune them together.

### Exam angle

"What is the effect of setting temperature T > 1 during text generation?" Answer: **Flatter probability distribution, more random and diverse outputs**. Trap: "Sharper distribution" (that is T < 1), "Faster generation" (temperature does not affect speed).

---

## Concept 8: Top-k Sampling

### The problem it solves

Even with temperature, sampling from the full vocabulary can produce low-probability nonsense tokens. Top-k sampling restricts the sampling pool: only consider the top-k highest-probability tokens, discard the rest, renormalize, and sample.

### The intuition

Imagine you have 50,000 tokens. The top-10 tokens account for 90% of the probability mass. The bottom 49,990 tokens account for 10%, but they are mostly noise ("cobblestone," "xylophone," random unicode).

Top-k says: ignore the bottom 49,990. Only sample from the top-10 (or top-50, or top-100, depending on k).

### The process (k=4)

Vocabulary probabilities (top 8 shown):
```
dog:    0.28
cat:    0.22
bird:   0.15
fish:   0.10
horse:  0.08
rabbit: 0.06
turtle: 0.04
zebra:  0.02
(+ 49,992 more tokens with tiny probabilities)
```

**Step 1:** Keep only top-k=4 tokens: {dog, cat, bird, fish}.

**Step 2:** Discard the rest: {horse, rabbit, turtle, zebra, ...}.

**Step 3:** Renormalize probabilities so top-4 sum to 1:
```
dog:  0.28 / (0.28 + 0.22 + 0.15 + 0.10) = 0.28 / 0.75 ≈ 0.37
cat:  0.22 / 0.75 ≈ 0.29
bird: 0.15 / 0.75 ≈ 0.20
fish: 0.10 / 0.75 ≈ 0.13
```

**Step 4:** Sample from the renormalized distribution. You might pick "cat" (29% chance), "dog" (37% chance), etc.

### Fixed-count cutoff

Top-k always keeps exactly k tokens, regardless of their probabilities. This is both a strength and a weakness:
- **Strength:** Conservative. Prevents sampling from the long tail of garbage tokens.
- **Weakness:** Inflexible. If the distribution is very confident (one token has 99% probability), top-k still forces you to consider k-1 other tokens. If the distribution is flat, top-k might cut off valid alternatives too early.

### Common misunderstandings

- **"Top-k = 1 is greedy decoding."** Close, but not quite. If you apply temperature before top-k, even top-k=1 can have randomness. In practice, top-k=1 with T=0 is greedy.
- **"Higher k is always better."** No. If k is too large (e.g., k=1000), you reintroduce noise. If k is too small (e.g., k=2), you lose diversity. Typical: k=40–50.

### Exam angle

"What is the purpose of top-k sampling?" Answer: **To restrict sampling to the k most probable tokens, filtering out low-probability noise**. Trap: "To make generation deterministic" (false, top-k still samples randomly within the top-k).

---

## Concept 9: Top-p (Nucleus) Sampling

### The problem it solves

Top-k has a fixed cutoff (k tokens). But sometimes the model is very confident (one token dominates), and you do not need k candidates. Other times the model is uncertain, and k candidates are not enough. Top-p adapts dynamically.

Top-p (also called nucleus sampling) keeps the smallest set of tokens whose cumulative probability exceeds p (e.g., p=0.9). The cutoff is probability-based, not count-based.

### The intuition

Imagine you are packing a suitcase. Top-k says "pack exactly 10 items." Top-p says "pack items until your suitcase is 90% full." If 5 items fill it, stop at 5. If you need 15 items, use 15.

### The process (p=0.85)

Vocabulary probabilities (top 8 shown):
```
dog:    0.28  cumulative: 0.28
cat:    0.22  cumulative: 0.50
bird:   0.15  cumulative: 0.65
fish:   0.10  cumulative: 0.75
horse:  0.08  cumulative: 0.83
rabbit: 0.06  cumulative: 0.89 ← exceeds p=0.85
turtle: 0.04  cumulative: 0.93
zebra:  0.02  cumulative: 0.95
```

**Step 1:** Sort tokens by probability descending.

**Step 2:** Accumulate probabilities until cumulative prob ≥ p=0.85.

**Step 3:** Stop at "rabbit" (cumulative = 0.89). Keep {dog, cat, bird, fish, horse, rabbit}.

**Step 4:** Discard the rest: {turtle, zebra, ...}.

**Step 5:** Renormalize and sample.

### Dynamic cutoff

The number of tokens selected varies:
- High confidence case: "Paris" has prob 0.95. Top-p (p=0.9) will pick only "Paris" (cumulative 0.95 exceeds 0.9 immediately). Effective nucleus size: 1.
- Low confidence case: 10 tokens each have prob 0.1. Top-p (p=0.9) will pick 9 of them (cumulative 0.9). Effective nucleus size: 9.

### Top-p vs Top-k

| Aspect               | Top-k                        | Top-p                            |
|----------------------|------------------------------|----------------------------------|
| Cutoff type          | Fixed count (k tokens)       | Dynamic (by probability mass)    |
| Low confidence case  | Always uses k tokens         | Restricts to fewer tokens        |
| High confidence case | May include garbage          | Stays focused                    |
| Typical value        | k = 40–50                    | p = 0.9–0.95                     |

### Common misunderstandings

- **"Top-p and top-k are mutually exclusive."** No. You can use both together. Apply top-k first, then top-p on the reduced set (or vice versa). The intersection is what you sample from. More on this below.
- **"Top-p guarantees exactly p cumulative probability."** No. Top-p includes tokens until cumulative prob ≥ p. You might overshoot (e.g., cumulative = 0.92 when p=0.9).

### Exam angle

"What is the main advantage of top-p sampling over top-k sampling?" Answer: **Top-p adapts the cutoff dynamically based on the confidence of the model, avoiding low-probability tokens when the model is certain**. Trap: "Top-p is faster" (not the reason), "Top-p is deterministic" (false).

---

## Concept 10: Combining Top-k and Top-p (Intersection)

### The problem it solves

In practice, many APIs (including OpenAI and HuggingFace) allow you to set both top-k and top-p. How do they interact?

**Answer:** Both filters are applied, and you sample from the intersection.

### Worked example (HIGH YIELD FOR EXAM)

Given token probabilities:
```
A: 0.13
B: 0.04
C: 0.22
D: 0.16
E: 0.01
F: 0.18
G: 0.07
H: 0.10
I: 0.06
J: 0.03
```

Parameters: k=2, p=0.95

**Step 1: Apply top-k (k=2)**

Sort by probability: C (0.22), F (0.18), D (0.16), A (0.13), H (0.10), G (0.07), I (0.06), B (0.04), J (0.03), E (0.01)

Keep top-2: {C, F}

**Step 2: Apply top-p (p=0.95)**

Cumulative probabilities:
```
C: 0.22  cumulative: 0.22
F: 0.18  cumulative: 0.40
D: 0.16  cumulative: 0.56
A: 0.13  cumulative: 0.69
H: 0.10  cumulative: 0.79
G: 0.07  cumulative: 0.86
I: 0.06  cumulative: 0.92
B: 0.04  cumulative: 0.96 ← exceeds p=0.95
```

Keep: {C, F, D, A, H, G, I, B}

**Step 3: Intersection**

Top-k set: {C, F}
Top-p set: {C, F, D, A, H, G, I, B}
Intersection: {C, F}

**Step 4: Which constraint is binding?**

Top-k is binding. It restricted the set to {C, F}, even though top-p would have allowed more tokens.

**Final sampling set:** {C, F}

Renormalize: C: 0.22 / (0.22 + 0.18) = 0.55, F: 0.45. Sample from this distribution.

### When is top-p binding, when is top-k binding?

- **Top-k binding:** When the top-k set is smaller than the top-p set. Example above.
- **Top-p binding:** When the model is very confident. Example: C has prob 0.99. Top-p (p=0.95) includes only C. Top-k (k=50) would include 50 tokens. Intersection is {C}. Top-p is binding.

### Common misunderstandings

- **"Top-k and top-p are always used together."** No. Many APIs default to top-p only. You can use one or the other or both.
- **"Using both is redundant."** Not always. Top-k prevents extreme long-tail sampling. Top-p prevents including garbage when the model is confident. Together, they provide both a floor (k) and a ceiling (p).

### Exam angle (HIGH YIELD)

"Given token probabilities A=0.13, B=0.04, C=0.22, D=0.16, E=0.01, F=0.18, G=0.07, H=0.10, I=0.06, J=0.03 with k=2 and p=0.95, what is the final sampling set?" Answer: **{C, F} (intersection of top-k and top-p)**. Walk through the calculation as shown above. This exact type of problem appears in exams.

---

## Concept 11: Repetition Penalty and Frequency Penalty

### The problem it solves

Models can get stuck in repetitive loops: "I think I think I think..." Repetition penalties discourage this by reducing the probability of tokens that have already appeared.

### Repetition Penalty (HuggingFace style)

**Formula:** logit[t] /= penalty (if token t has appeared before)

**Effect:** Divide the logit (raw score) by the penalty value (e.g., 1.3). After softmax, the token's probability decreases.

**Example:**
```
Without penalty: "The cat sat. The cat sat. The cat sat."
With penalty (1.3): "The cat sat on the mat and yawned."
```

**Typical values:** 1.0 = no penalty. 1.2–1.5 typical. >1.5 risks avoiding necessary repetitions (e.g., "the" appears multiple times legitimately).

### Frequency Penalty (OpenAI style)

**Formula:** logit[t] -= frequency_penalty × count(t)

**Effect:** Subtract a penalty proportional to how many times token t has appeared.

**Example:**
```
Without penalty: "Paris Paris Paris is Paris the capital Paris Paris"
With penalty (0.5): "Paris is the capital of France and a major hub."
```

**Typical values:** 0.0 = no penalty. 0.4–1.0 typical. >2.0 aggressively avoids repetition.

**Presence Penalty (OpenAI):** Binary version. Penalty is applied if the token appeared at all (count ≥ 1), regardless of how many times. Simpler but less nuanced.

### Common misunderstandings

- **"Repetition penalty prevents all repetition."** No. It just makes repeated tokens less likely. Common words like "the" will still repeat because they are highly probable.
- **"Repetition penalty and frequency penalty are the same."** Similar idea, different implementations. Repetition penalty divides logits, frequency penalty subtracts. Both reduce repeated token probabilities.

### Exam angle

"What is the purpose of frequency penalty in LLM generation?" Answer: **To penalize tokens proportional to how many times they have appeared, reducing repetitive text**. Trap: "To increase generation speed" (false).

---

## Concept 12: Max Tokens and Stop Sequences

### The problem it solves

You need a way to control when generation stops.

### Max Tokens

**Definition:** The maximum number of tokens the model will generate in one response.

**Units:** Tokens, not words. 1 word ≈ 1.3 tokens on average. "ChatGPT" = 3 tokens.

**Example:**
```
max_tokens=20:
"The transformer architecture was introduced in 2017 by Google researchers in the..."

max_tokens=50:
"...paper Attention Is All You Need. It replaced RNNs with self-attention mechanisms, enabling much faster training."
```

**Why it matters:** Limits cost (you pay per token), prevents runaway generation, and ensures responses fit in your UI.

### Stop Sequences

**Definition:** Custom strings that signal the model to stop immediately when encountered.

**Example 1 (code generation):**
```
stop=["\n```", "def "]
```
Stop after the code block ends or before the next function definition.

**Example 2 (Q&A):**
```
stop=["\nUser:"]
```
Stop before the next user turn in a chat.

**Example 3 (JSON):**
```
stop=["}\n"]
```
Stop after the JSON object closes.

**Why it matters:** Structured output often has clear end markers. Stop sequences prevent the model from continuing past the logical endpoint.

### Common misunderstandings

- **"Max tokens is a hard limit."** Correct. If you set max_tokens=100, generation stops at 100 tokens even if the sentence is incomplete.
- **"Stop sequences are optional."** Yes, but highly recommended for structured tasks (code, JSON, multi-turn chat).

### Exam angle

"What is the difference between max_tokens and stop sequences?" Answer: **Max tokens is a hard limit on token count; stop sequences halt generation when a specific string is encountered**. Trap: "Stop sequences are faster" (not the reason).

---

## Concept 13: OpenAI Chat Completions API

### The problem it solves

You want to call an LLM from your code. OpenAI provides a REST API. The key endpoint is `chat.completions.create`.

### Core Message Roles

**system:** Sets the behavior, tone, and constraints for the assistant.
```
{"role": "system", "content": "You are a helpful assistant."}
```
The model tries to follow this instruction throughout the conversation.

**user:** The input from the human.
```
{"role": "user", "content": "What is AI?"}
```

**assistant:** Previous responses from the model. Used to maintain conversation history or provide few-shot examples.
```
{"role": "assistant", "content": "AI is the simulation of human intelligence by machines."}
```

**tool:** (New in function calling) Contains the result of a tool/function call. More on this below.

**developer:** (New) Replaces `system` in some contexts. Provides developer-specific instructions with stricter enforcement.

### Message Array Structure

Messages are passed as a list:
```python
messages = [
    {"role": "system", "content": "You are a helpful chatbot."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is the simulation of human intelligence by machines."},
    {"role": "user", "content": "Give an example."}
]
```

The model sees the full conversation history and generates the next assistant turn.

### What Happens Under the Hood

Before reaching the model, the structured messages are concatenated into a single token sequence:
```
<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
What is AI?<|im_end|>
<|im_start|>assistant
AI is the simulation of human intelligence by machines.<|im_end|>
<|im_start|>user
Give an example.<|im_end|>
<|im_start|>assistant
```

The model predicts the next token after the last `<|im_start|>assistant`. The special tokens `<|im_start|>` and `<|im_end|>` are part of the tokenizer vocabulary (for models like GPT-3.5 and GPT-4).

### Why Roles Matter

Roles provide structure. The system role is more "authoritative" than user role. The model is trained to respect system instructions more strictly. Few-shot examples are provided via assistant messages.

### Example: Few-Shot Prompting

You want the model to classify sentiment. Provide examples:
```python
messages = [
    {"role": "system", "content": "You are a sentiment classifier."},
    {"role": "user", "content": "I love this product!"},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "This is terrible."},
    {"role": "assistant", "content": "negative"},
    {"role": "user", "content": "It's okay, I guess."}
]
```

The model sees the pattern and generates "neutral."

### Common misunderstandings

- **"System role is required."** No. You can skip it, but then you lose behavioral control.
- **"Assistant messages are only for conversation history."** No. They are also used for few-shot examples (even if no actual conversation happened).
- **"Roles change the model."** No. Roles are just input formatting. The same model processes all roles. The training data included role markers, so the model learned to respect them.

### Exam angle

"What is the purpose of the system role in the OpenAI Chat Completions API?" Answer: **To set the overall behavior, tone, and constraints for the assistant**. Trap: "To store conversation history" (that is the assistant role).

---

## Concept 14: API Parameters (Temperature, Top-p, Max Tokens, etc.)

### Key Parameters

**temperature:** Controls randomness. 0.0 = deterministic, 1.0 = default, >1.0 = creative. Typical: 0.3 for factual, 0.7 for balanced, 1.2 for creative.

**top_p:** Nucleus sampling. 0.0–1.0. Typical: 0.9–0.95. Use instead of top-k (OpenAI API does not expose top-k directly, but HuggingFace does).

**max_tokens:** Hard limit on output length. 1–128k+ depending on model. Typical: 256 for factual, 1024 for creative.

**frequency_penalty:** 0.0–2.0. Penalizes tokens based on frequency. Reduces repetition. Typical: 0.4 for creative, 0.1 for factual.

**presence_penalty:** 0.0–2.0. Binary version of frequency penalty (penalty if token appeared at least once). Typical: 0.0–0.5.

**stop:** List of strings. Stop generation when any string is encountered. Example: `["\n\n", "User:"]`.

### Example API Call

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."}
    ],
    temperature=0.7,
    max_tokens=200,
    top_p=0.9,
    frequency_penalty=0.2,
    stop=["\n\n"]
)

print(response.choices[0].message.content)
```

**Output:** "RAG (Retrieval-Augmented Generation) combines a retriever that fetches relevant documents with a language model that generates answers using those documents."

### Common misunderstandings

- **"Temperature and top_p do the same thing."** No. Temperature reshapes the distribution. Top-p truncates it. They can be used together, but typically you tune one or the other.
- **"Higher max_tokens is always better."** No. Longer outputs cost more and can be verbose. Set max_tokens to what you need.

### Exam angle

"Which parameter is most effective in reducing hallucinations?" Answer: **Temperature (low temperature, e.g., 0.3, reduces randomness and sticks to high-probability, factual tokens)**. Trap: "Top-k sampling" (helps, but temperature is more direct), "Max tokens" (irrelevant to hallucination).

---

## Concept 15: Function / Tool Calling

### The problem it solves

You want the model to call external functions (e.g., get_weather, query_database, send_email) and integrate the results into its response. Function calling (also called tool calling) enables this.

### How it works

1. **Define tools:** Provide a JSON schema describing each function (name, description, parameters).
2. **User query:** "What is the weather in Paris?"
3. **Model decides:** The model outputs a structured tool call: `{"name": "get_weather", "arguments": {"location": "Paris"}}`.
4. **You execute:** Your code calls the actual `get_weather("Paris")` function, gets the result: `{"temperature": 18, "condition": "cloudy"}`.
5. **Feed result back:** Add a message with role `tool` containing the function result.
6. **Model responds:** "The weather in Paris is 18°C and cloudy."

### Tool Schema Example

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]
```

### API Call with Tools

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What is the weather in Paris?"}
    ],
    tools=tools,
    tool_choice="auto"  # model decides whether to call a tool
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # Execute function (your code)
    result = get_weather(arguments["location"])
    
    # Feed result back
    messages.append(response.choices[0].message)  # assistant message with tool call
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })
    
    # Second API call to get final response
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(final_response.choices[0].message.content)
```

### Why This Matters

Function calling enables agentic workflows: the model can query databases, call APIs, search the web, and integrate real-time data. It is how ChatGPT plugins, GPT-4 code interpreter, and retrieval-augmented generation (RAG) work under the hood.

### Common misunderstandings

- **"The model executes the function."** No. The model only generates the function call as structured JSON. You execute it in your code.
- **"Tool calling requires fine-tuning."** No. GPT-4 and GPT-3.5-turbo are trained to recognize tool schemas and output valid JSON. You provide the schema, the model handles the rest.

### Exam angle

"In the OpenAI function calling workflow, what role does the function result use?" Answer: **tool**. Trap: "assistant" (that is the model's response), "user" (that is human input).

"Who executes the function in function calling?" Answer: **The developer (your code)**. Trap: "The model" (false, the model only generates the call).

---

## Concept 16: LogitsProcessor and Constrained Generation

### The problem it solves

Sometimes you want to constrain generation to follow a pattern (e.g., generate only valid JSON, only words matching a regex, only SQL keywords). LogitsProcessor is a HuggingFace mechanism to manipulate logits before sampling, enforcing constraints.

### How it works

A LogitsProcessor is a function that takes the current logits (raw scores for all tokens) and modifies them before sampling. You can:
- Set invalid tokens to -∞ (so softmax assigns them 0 probability).
- Boost valid tokens (increase their logits).

### Example: RegexMask (from Lecture 4)

You want to generate text that matches a regex pattern (e.g., an email address). At each step, check which tokens would keep the partial string consistent with the regex. Mask out tokens that violate the regex.

**Key insight from the lecture:** Python's built-in `re.match(pattern, string)` only checks if the pattern matches the full string. But during token-by-token generation, you need **partial matching**—check if the current partial string could eventually match the pattern. The `regex` library (not built-in `re`) supports `regex.match(pattern, string, partial=True)`, which returns `True` if the string is a valid prefix.

### Code Sketch (Conceptual)

```python
import regex  # not built-in re
import torch

class RegexLogitsProcessor:
    def __init__(self, pattern, tokenizer):
        self.pattern = regex.compile(pattern)
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores):
        # input_ids: tokens generated so far
        # scores: logits for next token (vocab_size)
        
        current_text = self.tokenizer.decode(input_ids[0])
        
        for token_id in range(len(scores[0])):
            candidate_text = current_text + self.tokenizer.decode([token_id])
            
            # Check if candidate_text could match pattern (partial=True)
            if not self.pattern.match(candidate_text, partial=True):
                scores[0, token_id] = float('-inf')  # mask out invalid token
        
        return scores
```

**Usage:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

logits_processor = RegexLogitsProcessor(r'\w+@\w+\.\w+', tokenizer)

output = model.generate(
    input_ids,
    logits_processor=[logits_processor],
    max_length=50
)
```

The model will only generate tokens that keep the string on track to form a valid email address.

### Why Python's `re.match` Fails

`re.match(pattern, string)` checks if the *entire* string matches. Example:
```python
import re
pattern = r'\w+@\w+\.\w+'
re.match(pattern, "abc")  # False (not a complete email)
```

But "abc" is a valid prefix (it could become "abc@example.com"). We need partial matching.

`regex` library (third-party) supports `partial=True`:
```python
import regex
pattern = regex.compile(r'\w+@\w+\.\w+')
pattern.match("abc", partial=True)  # <regex.Match object> (valid prefix)
pattern.match("abc@", partial=True)  # <regex.Match object> (valid prefix)
pattern.match("abc@def.com", partial=True)  # <regex.Match object> (complete match)
pattern.match("abc#", partial=True)  # None (invalid)
```

### Common misunderstandings

- **"LogitsProcessor slows down generation."** True, but the slowdown is often acceptable for structured output tasks where you need guarantees (e.g., valid JSON, SQL).
- **"Only HuggingFace supports this."** No. OpenAI's `response_format={"type": "json_object"}` is a form of constrained generation (forces valid JSON). You can also use external libraries like `guidance` or `outlines` to constrain any model.

### Exam angle

"Why is `regex.match(pattern, string, partial=True)` necessary for token-by-token constrained generation?" Answer: **Because Python's built-in `re.match` only checks full matches, not partial prefixes, so it would reject valid intermediate tokens**. Trap: "To speed up matching" (not the reason), "To support Unicode" (irrelevant).

---

## Concept 17: HuggingFace Hosted Inference API vs Local Transformers vs Ollama

### The problem it solves

You want to run an LLM. Three options: HuggingFace Hosted Inference API (cloud), local Transformers (your GPU), or Ollama (local, optimized runtime).

### HuggingFace Hosted Inference API

**Pros:** No GPU needed. Pay-per-request or free tier. Access to thousands of models.

**Cons:** Rate limits. Latency (network round-trip). No control over infrastructure.

**Use case:** Prototyping, low-volume inference, trying different models quickly.

### Local Transformers (transformers + torch)

**Pros:** Full control. No API rate limits. Privacy (data never leaves your machine). Can fine-tune.

**Cons:** Requires GPU (for large models). Memory-intensive (e.g., LLaMA 70B needs 140 GB VRAM). Complex setup.

**Use case:** Research, fine-tuning, high-volume inference, sensitive data.

### Ollama

**Pros:** Optimized local runtime. Simple CLI. Good for CPU inference (quantized models). Fast download and setup.

**Cons:** Limited to models in Ollama's library. Less flexible than raw Transformers.

**Use case:** Local development, running LLaMA/Mistral/Gemma on your laptop.

### Trade-offs Summary

| Aspect          | HuggingFace API | Local Transformers | Ollama            |
|-----------------|-----------------|---------------------|-------------------|
| **GPU needed**  | No              | Yes (for large)     | No (uses CPU)     |
| **Privacy**     | Data sent out   | Fully local         | Fully local       |
| **Cost**        | Pay-per-token   | One-time GPU cost   | Free (your HW)    |
| **Rate limits** | Yes             | No                  | No                |
| **Setup**       | Easy            | Complex             | Easy              |

### Exam angle

"What is the main advantage of running a local Transformers model compared to using an API?" Answer: **Privacy (data never leaves your machine) and no rate limits**. Trap: "Lower cost" (depends on GPU cost vs API cost), "Higher accuracy" (same model, same accuracy).

---

## Putting it together

LLMs are trained in three stages: pretraining (next-token prediction on massive data), fine-tuning (instruction-following on human demos), and RLHF (alignment with human preferences via RL). The major families are BERT (encoder-only, understanding), GPT (decoder-only, generation), and T5 (encoder-decoder, seq2seq). When generating text, you choose a decoding strategy: greedy (deterministic, fast, repetitive), beam search (explores top-B paths, generic), temperature sampling (adds randomness, controlled by T), top-k (restricts to top-k tokens), top-p (dynamic cutoff by cumulative prob), or combinations. Practical APIs like OpenAI Chat Completions let you control generation via roles (system, user, assistant, tool), parameters (temperature, top_p, max_tokens, frequency_penalty), and tools (function calling). Constrained generation (LogitsProcessor, regex matching) lets you enforce structure. The exam will test your understanding of decoding strategies (especially the top-k + top-p intersection problem), API roles, and when to use which parameter.

---

## Check yourself (5 questions)

**Q1.** You want factual, deterministic outputs from an LLM. Which decoding strategy and temperature should you use?

A) Beam search with T=1.5  
B) Greedy decoding with T=0.3  
C) Top-p sampling with T=1.0  
D) Top-k sampling with T=2.0  

**Answer:** B — Greedy or low temperature (T→0) are both deterministic. T=0.3 is low (factual). Beam search is also deterministic, but greedy is simpler. High T (1.5, 2.0) adds randomness, which you do not want.

---

**Q2.** Given token probabilities: {A: 0.5, B: 0.3, C: 0.1, D: 0.05, E: 0.05} with top-k=3 and top-p=0.8, what is the final sampling set?

A) {A, B, C}  
B) {A, B}  
C) {A, B, C, D, E}  
D) {A}  

**Answer:** B — Top-k=3 gives {A, B, C}. Top-p=0.8: cumulative prob of A+B = 0.5+0.3=0.8, so top-p gives {A, B}. Intersection is {A, B}. Top-p is binding.

---

**Q3.** In the OpenAI Chat Completions API, what is the purpose of the `tool` role?

A) To define the tool schema  
B) To return the result of a function call to the model  
C) To provide examples of tool usage  
D) To store conversation history  

**Answer:** B — After the model requests a tool call and you execute the function, you return the result via a message with role `tool`. The schema is defined in the `tools` parameter, not a message.

---

**Q4.** Why is `regex.match(pattern, string, partial=True)` necessary for token-by-token constrained generation?

A) To speed up regex matching  
B) To support Unicode characters  
C) To check if a partial string is a valid prefix for the pattern, not just a full match  
D) To compile the regex once and reuse it  

**Answer:** C — During generation, you need to know if "abc" could become "abc@example.com" (valid prefix). Python's `re.match` only checks full matches. `regex` with `partial=True` checks prefixes.

---

**Q5.** What is the main purpose of RLHF in LLM training?

A) To pretrain the model on a large corpus  
B) To fine-tune the model on instruction-following tasks  
C) To align the model's outputs with human preferences using reinforcement learning on pairwise comparisons  
D) To reduce the model's parameter count  

**Answer:** C — RLHF uses human preference comparisons to train a reward model, then optimizes the policy via PPO. Pretraining is unsupervised (next-token prediction), fine-tuning is supervised (demos). RLHF is RL-based alignment.

---

## Quick reference (for last-day revision)

**LLM training pipeline:** Pretraining (next-token on massive data) → SFT (instruction-following on demos) → RLHF (align with preferences via reward model + PPO).

**Model families:**
- BERT: Encoder-only, MLM, classification/QA.
- GPT: Decoder-only, causal LM, generation.
- T5: Encoder-decoder, span corruption, seq2seq.
- LLaMA/Mistral: Open-weights, strong performance, fine-tunable.

**Decoding strategies:**
- **Greedy:** argmax. Deterministic, fast, repetitive.
- **Beam search:** Keep top-B paths. Deterministic, generic, better than greedy for translation.
- **Temperature:** T < 1 = sharp (factual), T = 1 = default, T > 1 = flat (creative). P(token) = softmax(logits / T).
- **Top-k:** Keep top-k tokens, renormalize, sample. Fixed count. Typical k=40–50.
- **Top-p (nucleus):** Keep smallest set with cumulative prob ≥ p. Dynamic cutoff. Typical p=0.9–0.95.
- **Top-k + top-p:** Apply both, sample from intersection. Exam: compute which constraint binds.

**Repetition penalty:** logit[t] /= penalty if t seen. Reduces loops.

**Frequency penalty:** logit[t] -= penalty × count(t). Penalizes by frequency.

**Max tokens:** Hard limit on output length. Stop generation when reached.

**Stop sequences:** List of strings. Stop when any string encountered.

**OpenAI API roles:**
- **system:** Behavior/constraints.
- **user:** Human input.
- **assistant:** Model output or few-shot examples.
- **tool:** Function call result.

**API parameters:**
- **temperature:** 0.0–2.0. Lower = deterministic.
- **top_p:** 0.0–1.0. Nucleus sampling.
- **max_tokens:** Output length cap.
- **frequency_penalty:** 0.0–2.0. Reduce repetition.
- **stop:** List of strings.

**Function calling:** Define tool schema (JSON) → model outputs tool call → you execute → feed result back via `tool` role → model responds.

**LogitsProcessor:** Modify logits before sampling. Use `regex` library with `partial=True` for token-by-token regex matching (Python's `re` only does full matches).

**HF API vs Local vs Ollama:**
- **HF API:** Cloud, no GPU, rate limits.
- **Local Transformers:** GPU required, full control, privacy.
- **Ollama:** Local, CPU-friendly, simple CLI.
