# Lecture 4 — LLMs, Decoding, and Prompting APIs
**Instructor:** Prof. Sourangshu Bhattacharya

## TL;DR (5 bullets)
- LLM architecture families: Encoder-only (BERT, classification/NER), Decoder-only (GPT, generation), Encoder-Decoder (T5/BART, translation/summarization). BERT scored 82.1 on GLUE (2018), GPT-3 175B achieved 71.2% zero-shot TriviaQA with in-context learning.
- Decoding strategies trade determinism for diversity: Greedy (argmax, deterministic), Beam Search (keep top-k candidates, deterministic), Temperature (scale logits, T≈0 deterministic, T>1 creative), Top-k (keep k highest prob tokens), Top-p/Nucleus (cumulative prob threshold p).
- Temperature scaling formula: P(token) = softmax(logits / T). T=0.3 → near-deterministic, T=1.0 → default distribution, T=1.8 → high randomness. Use T≈0.7 for factual tasks, T≈1.2 for creative writing.
- Top-k and Top-p intersection math: Given token probs, apply both filters and sample from intersection. Example: k=2 selects 2 highest, p=0.95 selects cumulative ≥0.95. Intersection = tokens passing BOTH constraints. If k is smaller, k is binding.
- OpenAI Chat Completions API roles: system (behavior/constraints), developer (specialized instructions), user (human input), assistant (model responses, used for conversation history and few-shot examples), tool (function call results). max_tokens is hard output cap.

## Exam-relevant concepts

### BERT (Encoder-Only)
- **Definition:** Transformer encoder-only model trained on Masked Language Modeling (MLM, predict 15% masked tokens) and Next Sentence Prediction. Bidirectional context.
- **Numbers:** BERT-Base: 110M params, 12 layers. BERT-Large: 340M params, 24 layers. Training data: BookCorpus + Wikipedia (~3.3B tokens).
- **When it matters (exam trap):** Encoder-only → classification, NER, QA (tasks needing full bidirectional context). Cannot generate text autoregressively. Fine-tuned with small task-specific head.
- **Common confusion:** BERT sees entire input at once (bidirectional). GPT sees only past tokens (causal/autoregressive). BERT is NOT generative.

### GLUE Benchmark
- **Definition:** General Language Understanding Evaluation — 9 diverse NLU tasks (sentiment, paraphrase, inference, linguistic acceptability). Single average score.
- **Numbers:** BERT-Large scored 82.1 (2018), RoBERTa 84.6. Human baseline ~89.8 on SuperGLUE.
- **When it matters (exam trap):** GLUE = encoder-only model benchmark (classification tasks). NOT for generation (use LAMBADA, TriviaQA for that).
- **Common confusion:** GLUE tasks include CoLA (linguistic acceptability, Matthew's correlation), SST-2 (sentiment, accuracy), MRPC (paraphrase), MNLI (inference).

### GPT-1/GPT-2 (Decoder-Only)
- **Definition:** Causal decoder-only Transformer trained to predict next token. GPT-1: 117M params, BookCorpus. GPT-2: 1.5B params, 40GB WebText.
- **Numbers:** GPT-2 1.5B: 63.2% zero-shot LAMBADA accuracy (prior SOTA 59.2%). First model to show emergent zero-shot task performance from prompts.
- **When it matters (exam trap):** Decoder-only → text generation, translation, summarization via prompts (no fine-tuning). GPT-2 introduced staged release due to misuse concerns.
- **Common confusion:** GPT-1 used fine-tuning. GPT-2 showed zero-shot transfer (no task-specific training). GPT-3 scaled to in-context learning.

### LAMBADA Benchmark
- **Definition:** Long-range cloze task: predict last word of narrative passage where local context is insufficient (requires full discourse). 10K passages from BookCrossing fiction.
- **When it matters (exam trap):** Tests discourse comprehension, not co-occurrence. N-gram models score near 0%. GPT-2 1.5B: 63.2% zero-shot (first model to beat prior SOTA).
- **Common confusion:** LAMBADA is NOT a standard language modeling benchmark (perplexity). It's a cloze task requiring 4+ sentence context.

### T5 and BART (Encoder-Decoder)
- **Definition:** Seq2seq models with bidirectional encoder + autoregressive decoder. T5 casts all tasks as text→text. BART uses denoising objective (varied corruptions).
- **Numbers:** T5 scales 60M to 11B params. Trained on C4 (750GB cleaned web text). T5-11B: 88.9 SuperGLUE (human 89.8).
- **When it matters (exam trap):** Encoder-decoder → translation, summarization, generation tasks requiring input→output mapping. T5 unifies all NLP tasks as text→text.
- **Common confusion:** BART encoder is BERT-like (bidirectional), decoder is GPT-like (causal). Used for summarization (mBART, Pegasus).

### GPT-3 and In-Context Learning
- **Definition:** 175B params, ~300B training tokens, 2048 context window. Zero-shot/few-shot task performance from prompt examples (no gradient updates).
- **Numbers:** TriviaQA (closed-book): 64.3% zero-shot, 68.0% 1-shot, 71.2% 64-shot. Prior fine-tuned SOTA: 68.0%.
- **When it matters (exam trap):** In-context learning emerges with scale. Smaller GPT-3 variants show weaker performance (not a smooth trend — qualitative shift).
- **Common confusion:** Zero-shot = task description only. Few-shot = 10-100 examples in prompt. Approaches fine-tuned smaller models without weight updates.

### InstructGPT and RLHF
- **Definition:** Align GPT-3 with human preferences via 3 stages: (1) Supervised Fine-Tuning on human demos, (2) Reward Model from pairwise comparisons, (3) RL with PPO against reward model.
- **Numbers:** InstructGPT 1.3B preferred over GPT-3 175B (71% vs 37% win-rate). Released as ChatGPT (late 2022).
- **When it matters (exam trap):** Raw LM predicts likely continuations, not helpful answers. RLHF teaches helpful/harmless/honest responses. KL penalty keeps policy near SFT.
- **Common confusion:** Reward model is trained on annotator rankings (pairwise comparisons), not absolute scores. PPO optimizes policy against reward model.

### LLaMA / Mistral / Open-Weights Models
- **Definition:** Meta's LLaMA (Feb 2023): 7B-70B dense models, commercial license (LLaMA 2). Mistral 7B (Sep 2023): compact dense, Apache-2.0. Mixtral 8×7B: open MoE.
- **Numbers:** MMLU 5-shot: Mistral 7B 60.1, LLaMA-2 70B 68.9, Mixtral 8×7B 70.6, GPT-3.5 70.0.
- **When it matters (exam trap):** Open-weights models enable on-device, fine-tunable, auditable LLMs. Template for Alpaca, Vicuna, thousands of community fine-tunes.
- **Common confusion:** MMLU = Massive Multitask Language Understanding (57 subjects, STEM + humanities). Standard knowledge test for model reports.

### Multimodal and Reasoning Models
- **Definition:** GPT-4/GPT-4V (text+image), Gemini (text+image+audio+video), Claude 3/4 (vision+long context). o1/o3/DeepSeek-R1 spend inference-time compute on chain-of-thought search (RL on verifiable outcomes).
- **When it matters (exam trap):** Reasoning models solve competition-level math/coding. Disadvantage: time/token consuming (higher latency, cost).
- **Common confusion:** Reasoning models shift compute from training to inference. Chain-of-thought is first-class output (model "thinks" before answering).

### Greedy Decoding
- **Definition:** Select token with highest probability at each step. Deterministic.
- **Formula:** token = argmax(P(token | context)).
- **When it matters (exam trap):** Temperature ≈ 0 approximates greedy (near-deterministic). Use for legal docs, high-consistency tasks. No diversity.
- **Common confusion:** Greedy is locally optimal at each step, NOT globally optimal (may miss better overall sequence).

### Beam Search
- **Definition:** Keep top-k candidates at each step. Explore k branches in parallel. Deterministic if beam_size and tie-breaking rules are fixed.
- **When it matters (exam trap):** Better than greedy (explores multiple paths), but still deterministic. Common beam_size = 5-10. May still miss global optimum.
- **Common confusion:** Beam search with beam_size=1 is greedy. Larger beam_size → better quality but slower.

### Temperature Scaling
- **Definition:** Scale logits before softmax. P(token) = softmax(logits / T).
- **Numbers:** T=0.3 (very cold, nearly deterministic, factual), T=1.0 (default distribution), T=1.8 (very hot, high randomness, creative).
- **When it matters (exam trap):** T≈0.7 for factual Q&A, T≈1.2 for creative writing, T>1.5 for brainstorming. T≈0 reduces hallucinations (deterministic output).
- **Common confusion:** T=0 is undefined (division by zero). Use T=0.01 or greedy decoding instead. T→0 makes softmax sharper (winner-takes-all).

### Top-k Sampling
- **Definition:** Keep only k highest-probability tokens, redistribute probability mass among them, then sample.
- **Numbers:** Typical k=40-50. Low k (e.g., 10) = conservative. High k (e.g., 80) = diverse.
- **When it matters (exam trap):** Fixed vocabulary cutoff (always k tokens, regardless of confidence). May include low-prob tokens if top-k is large.
- **Common confusion:** Top-k is static (always k tokens). Top-p is dynamic (varies by probability mass).

### Top-p (Nucleus) Sampling
- **Definition:** Sample from smallest set of tokens whose cumulative probability ≥ p. Dynamic vocabulary cutoff.
- **Numbers:** Typical p=0.9-0.95. Low confidence → restricts to fewer tokens. High confidence → stays focused.
- **When it matters (exam trap):** Top-p=0.95 means "keep tokens until cumulative prob reaches 95%, discard rest." More adaptive than top-k.
- **Common confusion:** Top-p stops when cumulative prob exceeds p. May include 2 tokens or 20 tokens depending on distribution.

### Intersection of Top-k and Top-p (Exam Math)
- **Example:** Token probs A=0.13, B=0.04, C=0.22, D=0.16, E=0.01, F=0.18, G=0.07, H=0.10, I=0.06, J=0.03.
  - Top-k=2: Keep C (0.22), F (0.18).
  - Top-p=0.95: Cumulative = C(0.22) + F(0.18) + D(0.16) + A(0.13) + H(0.10) + G(0.07) + I(0.06) + B(0.04) = 0.96 → keep C,F,D,A,H,G,I,B.
  - Intersection = {C, F} (2 tokens). k=2 is binding constraint (smaller set).
- **When it matters (exam trap):** Apply both filters, sample from intersection. If k < |top-p set|, k dominates. If p is restrictive, p dominates.
- **Common confusion:** "Never use top-k and top-p together at extreme values" (sample paper warning). They compound and may produce degenerate output. Use one or the other, then tune temperature.

### Repetition Penalty
- **Definition:** Divide logit of any token that already appeared in context by penalty value.
- **Formula:** logit[t] /= penalty (if token t seen before).
- **Numbers:** Default 1.0 (no effect). Typical 1.2-1.5. Too high → avoids even necessary repeats.
- **When it matters (exam trap):** Prevents "The cat sat. The cat sat. The cat sat." loops. Used in code generation to avoid variable name repetition.
- **Common confusion:** Frequency penalty (OpenAI style) penalizes proportional to count: logit[t] -= freq_penalty × count(t). Presence penalty is binary (0 or 1).

### Max Tokens
- **Definition:** Hard cap on number of tokens generated in one response. Tokens ≠ words (1 word ≈ 1.3 tokens on average).
- **When it matters (exam trap):** "ChatGPT" = 3 tokens. Max_tokens=256 for factual Q&A, 1024 for creative writing. Output truncates if limit reached.
- **Common confusion:** Max tokens is OUTPUT cap, not input limit. Context window (e.g., 2048, 128k) is input+output combined.

### Stop Sequences
- **Definition:** Custom strings that signal model to stop generating immediately. Useful for structured output (code blocks, JSON, Q&A turns).
- **Examples:** stop=["```"] for code generation, stop=["\nUser:"] for chatbot, stop=["}\n"] for JSON.
- **When it matters (exam trap):** Combine with max_tokens for full output control. Stop sequences prevent over-generation.
- **Common confusion:** Stop sequences are literal string matches. Model stops AS SOON AS sequence appears.

### OpenAI Chat Completions API Roles
- **Definition:** Structure for multi-turn conversations.
  - system: Sets behavior, tone, constraints ("You are a helpful assistant").
  - developer: Specialized instructions (more control than system).
  - user: Human input (questions, instructions).
  - assistant: Model responses (used for conversation history and few-shot examples).
  - tool: Function/tool call results (return values from external APIs).
- **When it matters (exam trap):** assistant role with prior turns → few-shot prompting (examples in messages array). tool role → structured function outputs. max_tokens → hard output cap (NOT context window limit).
- **Common confusion:** "What is tool role?" → Returns function call results to model. "What is developer role?" → Developer-specific instructions (new, not in all APIs).

### Function/Tool Calling
- **Definition:** Model generates structured JSON to call external functions. Flow: User query → Model → Tool selection → Function call → Response.
- **When it matters (exam trap):** JSON schema-based inputs. Model picks tool and generates arguments. Example: "What is LCM of 84 and 120?" → model calls lcm(84, 120) → returns result → model formats answer.
- **Common confusion:** Tool calling is TWO LLM calls: (1) Model decides which tool and generates args. (2) Function executes, result fed back to model for final answer. Total tokens: query + tool_call + result + final_answer.

### HF Hosted Inference API vs Local Transformers vs Ollama
- **Definition:**
  - HF Hosted Inference API: Cloud-hosted, pay-per-token, no local GPU needed. Managed service.
  - Local Transformers: Download weights, run locally with torch/CUDA. Full control, privacy, requires GPU.
  - Ollama: Local CLI for running LLMs (LLaMA, Mistral). Easy setup, no Python needed. Runs on CPU/GPU.
- **When it matters (exam trap):** Cloud API = scalable, no GPU cost upfront. Local = privacy, no per-token cost, requires hardware. Ollama = on-device, offline inference.
- **Common confusion:** HF Transformers library is for loading/running models locally (requires torch). HF Inference API is cloud service (REST API, no torch needed).

### LogitsProcessor Mechanics
- **Definition:** Custom logic to modify logits before sampling. Example: RegexMask with partial=True ensures output matches regex.
- **When it matters (exam trap):** re.match (Python std lib) checks full string match → breaks partial matching. regex.match (regex lib) supports partial=True → allows progressive matching as tokens generate.
- **Common confusion:** "Why does re.match break with partial=True?" → re.match requires full match by default. Use regex library for partial matching during generation.

## Diagrams / architectures / API sketches described

### LLM Architecture Evolution Timeline
```
Era              | Architecture       | Training Data           | Alignment/Inference
-----------------|--------------------|-----------------------|---------------------
Pre-2018         | Shallow embeddings | Small corpora         | Feature-based transfer
BERT (2018)      | Encoder-only       | 3.3B tokens (MLM+NSP) | Fine-tune with task head
GPT-1/2 (18-19)  | Decoder-only       | BookCorpus, WebText   | Zero-shot via prompts
T5/BART (19-20)  | Encoder-decoder    | C4 span-corruption    | Unified text-to-text
GPT-3 (2020)     | 175B dense         | ~300B tokens          | In-context learning
InstructGPT (22) | Base + reward model| Human demos + prefs   | SFT + RLHF
Open-weights (23)| LLaMA, Mistral     | Open corpora          | On-device, fine-tunable
Multimodal (23-25)| Vision-language   | Text+image+audio      | Inference-time CoT search
```

### Decoding Parameter Cheat Sheet
```
Parameter          | Range     | Default | Creative | Factual | Description
-------------------|-----------|---------|----------|---------|---------------------------
temperature        | 0.0–2.0   | 1.0     | 1.2      | 0.3     | Scales randomness
top_k              | 1–1000+   | 50      | 80       | 10      | Keeps k highest-prob tokens
top_p              | 0.0–1.0   | 1.0     | 0.95     | 0.7     | Cumulative prob cutoff
repetition_penalty | 1.0–2.0   | 1.0     | 1.2      | 1.1     | Reduces repeated tokens
frequency_penalty  | 0.0–2.0   | 0.0     | 0.4      | 0.1     | Proportional penalty
max_tokens         | 1–128k+   | Model   | 1024     | 256     | Hard output cap
```

### OpenAI Chat Completions API Structure
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain RAG in one sentence."}
    ],
    temperature=0.7,
    max_tokens=200
)
print(response.choices[0].message.content)
```

### Tool Calling Flow (LCM Example)
```
1. User: "What is LCM of 84 and 120?"
   ↓
2. LLM Call #1 (total_tokens=425):
   Model decides: call tool "compute_lcm" with args {"a": 84, "b": 120}
   ↓
3. Function Execution:
   def compute_lcm(a, b): return (a*b) // gcd(a,b)
   Result: 840
   ↓
4. LLM Call #2 (total_tokens=123):
   Input: User query + tool result (840)
   Model formats: "The LCM of 84 and 120 is 840."
   ↓
5. Final Response to User

Total tokens: 425 (call #1) + 123 (call #2) = 548
```

### Tool Calling JSON Schema Example
```python
tools = [{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string"}
      },
      "required": ["location"]
    }
  }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)
# Model generates: {"name": "get_weather", "arguments": {"location": "Paris"}}
```

### Few-Shot Prompting with Assistant Role
```python
messages = [
    {"role": "system", "content": "You are a sentiment classifier."},
    {"role": "user", "content": "I love this product!"},
    {"role": "assistant", "content": "Positive"},
    {"role": "user", "content": "This is terrible."},
    {"role": "assistant", "content": "Negative"},
    {"role": "user", "content": "It's okay, nothing special."}  # Query
]
# Model infers: "Neutral" (learns pattern from examples)
```

### Temperature Visualization (Conceptual)
```
T = 0.3 (Very Cold):
Probs: [0.90, 0.05, 0.03, 0.02] → Almost always picks first token

T = 1.0 (Default):
Probs: [0.50, 0.25, 0.15, 0.10] → Natural distribution

T = 1.8 (Very Hot):
Probs: [0.30, 0.25, 0.23, 0.22] → High randomness, all tokens plausible
```

### Top-k vs Top-p Example (k=4, p=0.85)
```
Token Probs (sorted descending):
dog:    28%  ← cum: 28%  ✓ k=4, ✓ p=0.85
cat:    22%  ← cum: 50%  ✓ k=4, ✓ p=0.85
bird:   15%  ← cum: 65%  ✓ k=4, ✓ p=0.85
fish:   10%  ← cum: 75%  ✓ k=4, ✓ p=0.85
horse:   8%  ← cum: 83%  ✗ k (only 4), ✓ p (still <85%)
rabbit:  6%  ← cum: 89%  ✗ k, ✗ p (exceeds 85%)
...

Top-k=4: Keep {dog, cat, bird, fish}
Top-p=0.85: Keep {dog, cat, bird, fish, horse, rabbit} (stops at 89% ≥ 85%)
Intersection: {dog, cat, bird, fish} (k is binding)
```

## Likely MCQ angles

1. **Architecture family selection:** "Which model for classification tasks?" → BERT (encoder-only). "Which for text generation?" → GPT (decoder-only). "Which for translation?" → T5/BART (encoder-decoder).

2. **Decoding determinism:** "Which decoding strategies are deterministic?" → Greedy, Beam Search. "Which introduce randomness?" → Temperature, Top-k, Top-p, Sampling. "Temperature ≈ 0 for?" → Legal docs, high-consistency tasks.

3. **Top-k and Top-p intersection math:** Given token probs and k/p values, calculate intersection. Binding constraint = smaller set. "k=2, p=0.95 with probs A-J → intersection?" → Apply both filters, pick smaller.

4. **OpenAI API roles:** "Which role returns function results?" → tool. "Which role for few-shot examples?" → assistant (with prior turns in messages array). "What is max_tokens?" → Hard output cap, NOT context window.

5. **Temperature scaling behavior:** "T=0.3 output?" → Near-deterministic, factual. "T=1.8 output?" → Creative, high randomness, may be incoherent. "T≈0 reduces hallucinations?" → Yes (picks highest-prob token consistently).

6. **Top-k vs Top-p tradeoffs:** "Top-k advantage?" → Simple, fixed count. "Top-k disadvantage?" → May include garbage if k is large. "Top-p advantage?" → Adaptive (dynamic vocab cutoff). "Top-p disadvantage?" → Variable token count, harder to tune.

7. **RLHF stages:** "What are the 3 stages of InstructGPT training?" → (1) Supervised fine-tuning on demos, (2) Reward model from pairwise comparisons, (3) RL with PPO against reward model.

8. **Reasoning model tradeoff:** "Major disadvantage of reasoning models (o1/o3)?" → Time/token consuming (higher latency, cost). "Advantage?" → Solve competition-level math/coding via chain-of-thought search.

9. **HF Hosted Inference API vs Local Transformers:** "When to use cloud API?" → Scalable, no GPU cost upfront, managed service. "When to use local?" → Privacy, no per-token cost, requires GPU/torch.

10. **LogitsProcessor regex matching:** "Why does re.match break with partial=True?" → re.match checks full string match (no partial support). "Solution?" → Use regex library with partial=True for progressive matching.

11. **Stop sequences:** "What are stop sequences for?" → Custom strings to halt generation (e.g., stop=["```"] for code, stop=["\nUser:"] for chatbot turns). "Combine with?" → max_tokens for full output control.

12. **In-context learning emergence:** "GPT-3 175B TriviaQA accuracy: 0-shot vs 64-shot?" → 64.3% vs 71.2%. "Does in-context learning improve smoothly with examples?" → No, shows qualitative shifts (not smooth trend).

## One-line flashcards

- Q: BERT architecture family → A: Encoder-only (bidirectional, classification/NER)
- Q: GPT architecture family → A: Decoder-only (causal, text generation)
- Q: T5/BART architecture family → A: Encoder-decoder (seq2seq, translation/summarization)
- Q: BERT-Large GLUE score (2018) → A: 82.1 (human baseline ~89.8 on SuperGLUE)
- Q: GPT-2 1.5B LAMBADA accuracy (zero-shot) → A: 63.2% (prior SOTA 59.2%)
- Q: GPT-3 175B parameter count → A: 175B params, ~300B training tokens, 2048 context window
- Q: GPT-3 TriviaQA 64-shot accuracy → A: 71.2% (prior fine-tuned SOTA 68.0%)
- Q: InstructGPT 1.3B vs GPT-3 175B win-rate → A: 71% vs 37% (1.3B preferred)
- Q: Greedy decoding formula → A: token = argmax(P(token | context))
- Q: Temperature formula → A: P(token) = softmax(logits / T)
- Q: Temperature T=0.3 use case → A: Factual Q&A, legal docs (near-deterministic)
- Q: Temperature T=1.2 use case → A: Creative writing, brainstorming
- Q: Temperature T≈0 effect on hallucinations → A: Reduces (deterministic output, highest-prob token)
- Q: Top-k sampling definition → A: Keep k highest-prob tokens, redistribute, sample
- Q: Top-p (nucleus) sampling definition → A: Keep smallest set with cumulative prob ≥ p
- Q: Typical top-k value → A: 40-50 (low k=10 conservative, high k=80 diverse)
- Q: Typical top-p value → A: 0.9-0.95 (adaptive vocab cutoff)
- Q: Repetition penalty formula → A: logit[t] /= penalty (if t seen before)
- Q: Frequency penalty formula → A: logit[t] -= freq_penalty × count(t)
- Q: Max tokens definition → A: Hard cap on generated token count (output only)
- Q: Stop sequences example (code generation) → A: stop=["```", "def "]
- Q: OpenAI system role → A: Sets behavior, tone, constraints
- Q: OpenAI developer role → A: Specialized instructions (more control than system)
- Q: OpenAI user role → A: Human input (questions, instructions)
- Q: OpenAI assistant role → A: Model responses (conversation history, few-shot examples)
- Q: OpenAI tool role → A: Function/tool call results (return values)
- Q: Tool calling flow → A: User query → Model → Tool selection → Function call → Response
- Q: Tool calling token cost (LCM example) → A: Call #1 (425) + Call #2 (123) = 548 total
- Q: Few-shot prompting with assistant role → A: Prior user/assistant turns in messages array
- Q: HF Hosted Inference API vs Local Transformers → A: Cloud (scalable, no GPU) vs Local (privacy, requires GPU/torch)
- Q: Ollama → A: Local CLI for running LLMs (LLaMA, Mistral) on CPU/GPU, offline
- Q: LogitsProcessor regex issue → A: re.match (no partial support) vs regex.match (partial=True)
- Q: MMLU benchmark → A: Massive Multitask Language Understanding (57 subjects, STEM+humanities)
- Q: Mistral 7B MMLU score → A: 60.1 (5-shot)
- Q: LLaMA-2 70B MMLU score → A: 68.9 (5-shot)
- Q: Mixtral 8×7B MMLU score → A: 70.6 (5-shot, open MoE)
- Q: Reasoning model disadvantage → A: Time/token consuming (higher latency, cost)
- Q: LAMBADA benchmark → A: Long-range cloze (predict last word, requires full discourse)
- Q: BERT pretraining objective → A: MLM (15% masked) + Next Sentence Prediction
- Q: T5 training data → A: C4 (750GB cleaned web text)
- Q: RLHF stages → A: (1) SFT on demos, (2) Reward model, (3) PPO against reward
- Q: Beam search with beam_size=1 → A: Equivalent to greedy decoding
- Q: Top-k and Top-p intersection (k=2, p=0.95 example) → A: Apply both, sample from intersection (binding = smaller set)
- Q: Temperature T=1.0 → A: Default distribution (natural balance)
- Q: Max tokens vs context window → A: Max tokens = output cap. Context window = input+output combined.
