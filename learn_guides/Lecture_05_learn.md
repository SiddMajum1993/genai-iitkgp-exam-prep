# Lecture 5 — Prompting APIs, Structured Outputs, LLM Evaluations
**Estimated read time:** 25–30 min · **Difficulty:** Core

## What this lecture is about

This lecture bridges three critical gaps in your LLM deployment knowledge. First, it clarifies how you actually run LLMs — not just the abstract "call an API" but the concrete choices: local inference with HuggingFace Transformers (using your GPU or Mac MPS), remote hosted inference (where HuggingFace servers do the work), and Ollama (a local runtime that bundles models and a REST server). Each has different trade-offs in privacy, speed, setup effort, and hardware requirements.

Second, it tackles the brittleness of unstructured LLM outputs. When you ask an LLM for JSON, it often returns markdown code fences or prose with embedded objects — breaking your `json.loads()` calls. The lecture shows you how to force valid JSON using OpenAI's response_format, Pydantic schemas paired with the `instructor` library, and constrained decoding via LogitsProcessor — a token-level masking technique that prevents the model from generating illegal tokens at generation time. Third, it introduces the metrics that measure LLM output quality: BLEU (precision of n-gram matches with brevity penalty, used in machine translation), ROUGE variants (recall-oriented, used in summarization, including ROUGE-L which uses Longest Common Subsequence), METEOR (adds stems and synonyms to F-score), and BERTScore (contextual embedding cosine similarity). These metrics will appear in exam MCQs asking "which metric for which task."

## Prerequisites

- Comfortable with Python dictionaries and JSON
- Conceptual understanding of tokenization: text splits into tokens, each token has an integer ID
- Basic ML awareness: what a validation set is, why we need metrics
- Familiar with the term "autoregressive generation" (each token is predicted using all previous tokens as context)
- You have used ChatGPT or similar tools, but you do not need to know how they work internally

---

## Concept 1: Three Ways to Run Open-Source LLMs

### The problem it solves

Cloud APIs like OpenAI charge per token, and your data leaves your machine. You need to understand the trade-offs: when to load a model locally, when to use a hosted inference provider, and when to use a bundled runtime like Ollama. This is a common exam scenario: "Which approach gives you the fastest inference on a Mac M3 with no internet?"

### The intuition (analogy first)

Think of three ways to cook a meal:

1. **Transformers Locally (Approach 1)**: You buy raw ingredients (model weights) and cook in your own kitchen (your GPU/MPS). Full control, full privacy, but you need the right pots and pans (torch, transformers, CUDA or MPS).
2. **HuggingFace Hosted Inference (Approach 2)**: You order delivery (send a request to HuggingFace servers). Fastest to set up, but slower per-request (network latency), your order details (prompts) leave your house, and the free tier throttles you.
3. **Ollama (Approach 3)**: You buy a meal kit (Ollama bundles the model and a pre-configured server). You cook at home, but the kit handles most setup. No torch required, fully private, but model selection is limited to what Ollama curates.

### Worked example (real numbers)

#### Approach 1: Transformers Locally

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

inputs = tokenizer("What is quantum computing?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Key steps:**
- `AutoTokenizer.from_pretrained(model_id)` downloads the vocabulary.
- `AutoModelForCausalLM.from_pretrained(model_id)` downloads ~4 billion parameters.
- `model.to(device)` moves all weights to GPU/MPS. On Mac M1/M2/M3, `mps` is much faster than `cpu`.
- `tokenizer(..., return_tensors="pt")` converts text to PyTorch tensors (token IDs).
- `model.generate(**inputs, max_new_tokens=100)` iteratively predicts the next token until done.

**Performance:** On Mac M3, MPS gives 3–5x speedup over CPU. On NVIDIA GPU, `device = "cuda"` is faster still.

#### Approach 2: HuggingFace Hosted Inference API

```python
import requests

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = requests.post(
    API_URL, headers=headers,
    json={
        "messages": [{"role": "user", "content": "What is quantum computing?"}],
        "model": "Qwen/Qwen3-32B:nscale"
    }
)
print(response.json())
```

**Key steps:**
- No `torch` install needed.
- Your prompt is sent to HuggingFace servers. They run inference and return the response.
- Free tier is rate-limited (e.g., 100 requests/hour).

**Trade-off:** Zero local setup, but data leaves your machine (confidentiality risk) and network latency adds ~500ms per request.

#### Approach 3: Ollama (Local, Zero-Config)

```bash
brew install ollama
ollama run llama3
```

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "What is quantum computing?",
        "stream": False
    }
)
print(response.json()["response"])
```

**Key steps:**
- `ollama run llama3` downloads the model and starts a local server at `localhost:11434`.
- Your Python script sends a POST request to the local server. No data leaves your machine.
- No `torch` or `transformers` needed — Ollama bundles its own runtime.

**Trade-off:** Easiest setup for 7B+ models, fully private, but model variety is limited (Llama 3, Mistral, Gemma — no GPT-4 or Claude).

### The formula (with meaning)

| Criterion          | Transformers (Local) | HF Hosted API     | Ollama (Local)     |
|--------------------|----------------------|-------------------|--------------------|
| Needs torch?       | ✅ Yes               | ❌ No             | ❌ No              |
| Needs internet?    | ❌ No                | ✅ Yes            | ❌ No              |
| Speed              | ⚡ Fast (MPS/GPU)   | 🐢 Slow (network) | ⚡ Fast (local)    |
| Privacy            | 🔒 Full              | ⚠️ Data sent to HF | 🔒 Full            |
| Setup effort       | 🔧 Medium            | ✅ Very easy      | ✅ Easy (brew)     |
| Model variety      | 🌐 Thousands         | 🌐 Thousands      | 📦 Popular ones    |
| Free?              | ✅ Yes (hardware)    | ✅ Yes (throttled) | ✅ Yes             |
| Best for           | Research, fine-tune  | Quick prototypes  | 7B+ models, daily use |

### Common misunderstandings

1. **"HuggingFace Hosted API is faster because it runs on better GPUs."** — No. Network latency (often 500–1000ms) dominates. Local inference on MPS is faster for most use cases once the model is loaded.
2. **"Ollama is just HuggingFace Transformers with a wrapper."** — No. Ollama uses its own inference engine (based on llama.cpp), not torch. This is why it does not need the `transformers` library.
3. **"Approach 1 always requires CUDA."** — No. Mac M1/M2/M3 have MPS (Metal Performance Shaders), which PyTorch supports via `device="mps"`. You do not need NVIDIA hardware.

### Exam angle

Expect a question like:
> "A developer wants to run Llama 3 on a Mac M3 laptop with no internet connection, using the least setup effort. Which approach should they choose?"
>
> (A) Transformers Locally  
> (B) HuggingFace Hosted Inference API  
> (C) Ollama  
> (D) OpenAI API

**Answer:** (C) Ollama — it is local (no internet), easy to set up (`brew install ollama`), and does not require torch. (B) needs internet, (D) needs internet, (A) requires more setup (`pip install torch transformers`).

---

## Concept 2: Constrained Decoding via LogitsProcessor

### The problem it solves

LLMs generate text token-by-token, choosing from a vocabulary of ~50,000 tokens at each step. Without constraints, the model can output anything — including invalid JSON, malformed URLs, or prose when you need only digits. Constrained decoding restricts which tokens the model can pick at each step, ensuring the output conforms to a specific structure (e.g., only digits, valid JSON, a regex pattern).

### The intuition (analogy first)

Imagine a child playing Mad Libs. The template says: "I have [number] apples." Without constraints, the child might write "I have banana apples" or "I have !!! apples." With constraints, you tell them: "You can only write a number here — 1, 2, 3, etc." At each blank, you physically remove all non-numeric cards from their hand. That is constrained decoding: at each token position, you mask (set logit to −∞) all tokens that would violate your rule.

### Worked example (real numbers)

#### Rule: Allow only digits (0–9) and spaces.

**Unconstrained output:**
```python
# Prompt: "2, 4, 8,"
# Unconstrained output: "16, 32, 64, 128, ..." (includes commas!)
```

**Constrained output:**
```python
from transformers import LogitsProcessor
import torch

class LogitsMask(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores) * -10  # penalize all tokens
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0  # zero penalty for allowed tokens
        scores = scores + mask
        return scores

# Tokenize digits and space
allowed_token_ids = [
    tokenizer.encode(t, add_special_tokens=False)[0]
    for t in list("0123456789 ")
]
allowed_token_ids += [tokenizer.eos_token_id]  # always allow EOS

model_inputs = tokenizer("2, 4, 8,", return_tensors="pt").to(device)
output = model.generate(
    **model_inputs,
    max_new_tokens=20,
    logits_processor=[LogitsMask(allowed_token_ids)]
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Output: '16 32 64 128 256 512' (no commas!)
```

**How it works:**
1. At each generation step, the model outputs logits (raw log-probability scores) for every vocabulary token.
2. `LogitsMask.__call__(input_ids, scores)` is called. It creates a mask of −10 for all tokens.
3. For each allowed token (digits, space, EOS), it sets the mask to 0 (no penalty).
4. `scores + mask` ensures disallowed tokens have very low logits.
5. Softmax converts logits to probabilities. A logit of −10 → probability ≈ 0. The model never picks disallowed tokens.

#### Rule: Enforce valid URLs with regex.

**Regex:** `r" ?https?:\/\/(www\.)?\w{1,256}\.\w{1,6}\/?$"`

```python
import regex

class RegexMask(LogitsProcessor):
    def __init__(self, regex_pattern, tokenizer):
        self.regex = regex_pattern
        self.tokenizer = tokenizer
        self.vocab_map = {
            tid: tokenizer.decode(tid)
            for tid in tokenizer.get_vocab().values()
        }
        self.start_index = 0

    def __call__(self, input_ids, scores):
        partial = self.tokenizer.decode(input_ids[0, self.start_index:])
        allowed = [self.tokenizer.eos_token_id]
        for token_id, token_str in self.vocab_map.items():
            # partial=True: match even if string is not yet complete
            if self.regex.match(partial + token_str, partial=True):
                allowed.append(token_id)
        mask = -torch.inf * torch.ones_like(scores)
        for tid in allowed:
            mask[:, tid] = 0
        scores = scores + mask
        return scores

regex_http = regex.compile(r" ?https?:\/\/(www\.)?\w{1,256}\.\w{1,6}\/?$")
rm = RegexMask(regex_http, tokenizer)
rm.start_index = len(tokenizer.encode("One of the most common web addresses is:")[0])

model_inputs = tokenizer("One of the most common web addresses is:", return_tensors="pt").to(device)
output = model.generate(**model_inputs, max_new_tokens=50, logits_processor=[rm])
print(tokenizer.decode(output[0], skip_special_tokens=True))
# Output: 'http://www.google.com' (valid URL, no trailing prose!)
```

**Key insight:**
- `partial=True` is critical. At step 5, the output is 'http://' — not a complete URL, but consistent with the regex. Without `partial=True`, the regex would reject it, and the model could not continue.
- Python's built-in `re` module does NOT support `partial=True`. You must use the `regex` library (`pip install regex`).

### The formula (with meaning)

**Logit masking:**
```
P(token_i) = exp(L_i) / Σ exp(L_j)

where:
  L_i = logit for token i
  Masking = setting L_i to −∞ (so exp(−∞) = 0)
```

**Greedy sampling:** Pick the token with the highest logit.

**Multinomial sampling:** Sample proportionally. Temperature `t → 0` approaches greedy. Constrained decoding works with both.

### Common misunderstandings

1. **"LogitsProcessor guarantees grammatically correct output."** — No. It only ensures each token is *locally* allowed (does not violate the constraint at this step). It has no lookahead, so it can "paint itself into a corner" where no valid tokens remain. Structured decoding (e.g., OpenAI's JSON mode) uses a state machine to avoid this.
2. **"Setting logit to −10 completely disables a token."** — Not quite. −10 drives probability very low (≈4.5e-5) but not zero. For zero probability, use `−torch.inf`.
3. **"Regex constraints work character-by-character."** — No. Constraints operate on tokens, not characters. The token for `http` is ` http` (with leading space), not four separate characters.

### Exam angle

Expect a question like:
> "Which of the following is NOT enforced by the LogitsProcessor?"
>
> (A) Only allowed tokens are generated.  
> (B) Some of the allowed tokens are generated preferentially.  
> (C) The output follows the input grammar completely.  
> (D) Partial matches from nested grammars.

**Answer:** (C) — LogitsProcessor has no lookahead, so it cannot guarantee the *complete* output satisfies the grammar (it might get stuck). (A) is true (disallowed tokens are masked), (B) is true (the model still samples from allowed tokens based on their original logits), (D) is true (regex `partial=True` supports nested patterns).

---

## Concept 3: Structured Outputs with Pydantic and Instructor

### The problem it solves

When you ask an LLM to "return JSON," it often returns markdown code fences (` ```json\n{...}\n``` `) or prose with embedded JSON. `json.loads(response)` raises `JSONDecodeError`. You need a way to *force* the LLM to return valid, schema-compliant JSON every time — no retries, no post-hoc parsing.

### The intuition (analogy first)

Think of ordering at a fast-food restaurant:

1. **Naive approach (prompt only):** You say "I want a burger." The cashier might write down "Sure! One burger coming up!" (prose), or "burger" (missing the quantity), or forget to include fries. You get inconsistent output.
2. **Pydantic schema + tool calling:** You hand the cashier a form (Pydantic schema) with fields: `item: str`, `quantity: int`, `add_fries: bool`. The cashier must fill in every field. If they write "many" for `quantity`, Pydantic rejects it (validation error). The form guarantees structure.
3. **Instructor:** A wrapper that automatically generates the form from your Python class definition and hands it to the cashier. You just say "I want a burger" and get back a fully typed `Order` object.

### Worked example (real numbers)

#### Problem: LLMs return inconsistent JSON.

**Naive prompt:**
```python
from openai import OpenAI

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": "Return the name and author of pydantic in a JSON object."
    }]
)
print(resp.choices[0].message.content)
```

**Output (failure case 1):**
```
```json
{
  "name": "pydantic",
  "author": "Samuel Colvin"
}
```
```

`json.loads(response)` raises `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` because of the markdown fences.

**Output (failure case 2):**
```
Ok, here's the author of pydantic: Samuel Colvin, and the name is: pydantic. Here's the JSON:
{ "name": "pydantic", "author": "Samuel Colvin" }
```

Again, `json.loads(response)` fails because of the leading prose.

#### Solution 1: OpenAI tool calling with Pydantic schema.

```python
from pydantic import BaseModel
from typing import List

class PythonPackage(BaseModel):
    name: str
    author: str

class Packages(BaseModel):
    packages: List[PythonPackage]

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Pydantic and FastAPI?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "Requirements",
            "parameters": Packages.model_json_schema(),
        }
    }],
    tool_choice={"type": "function", "function": {"name": "Requirements"}},
)

result = Packages.model_validate_json(
    resp.choices[0].message.tool_calls[0].function.arguments
)
print(result)
# Packages(packages=[PythonPackage(name='pydantic', author='Samuel Colvin'), ...])
```

**How it works:**
1. `Packages.model_json_schema()` generates the full JSON Schema (including nested `$defs` for `PythonPackage`). You do not write the schema by hand.
2. `tool_choice=...` forces the model to call the `Requirements` function. The model cannot return prose — it must emit JSON conforming to the schema.
3. `model_validate_json(...)` parses AND validates the response in one step. If the model returns `"age": "ten"` instead of `"age": 10`, Pydantic raises `ValidationError`.

#### Solution 2: Instructor (cleanest API).

```python
import instructor

client = instructor.patch(OpenAI())

packages = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Pydantic and FastAPI?"}],
    response_model=Packages,
)

# Result is already a typed Packages object!
assert isinstance(packages, Packages)
assert isinstance(packages.packages[0], PythonPackage)
```

**Key difference:**
- `instructor.patch(OpenAI())` wraps the client. You no longer pass `tools` or `tool_choice` manually.
- `response_model=Packages` handles schema generation, tool calling, and validation automatically.
- The return value IS the Pydantic object, not a raw API response. Fully type-safe.

### The formula (with meaning)

**Pydantic validation:**
```python
class Person(BaseModel):
    name: str
    age: int

# Type coercion: "10" (str) → 10 (int)
p = Person(name="Sam", age="10")
# Person(name='Sam', age=10)

# Validation error: "13.4" cannot be coerced to int
Person(name="Sam", age="13.4")
# ValidationError: 1 validation error for Person
#   age
#     Input should be a valid integer, unable to parse string as an integer
```

**JSON Schema generation:**
```python
Person.model_json_schema()
# {
#   "properties": {
#     "name": {"title": "Name", "type": "string"},
#     "age":  {"title": "Age",  "type": "integer"}
#   },
#   "required": ["name", "age"],
#   "title": "Person",
#   "type": "object"
# }
```

**OpenAI tool calling:**
- `tools=[{"type": "function", "function": {"name": "...", "parameters": <schema>}}]` tells the model which schemas are available.
- `tool_choice={"type": "function", "function": {"name": "..."}}` forces the model to call the specified function. The model must emit JSON conforming to `parameters`.

### Common misunderstandings

1. **"Tool calling executes the function on OpenAI's servers."** — No. The LLM only decides *which function to call* and *what arguments to pass*. Your Python code (the client) executes the function.
2. **"Pydantic is only for validation."** — No. Pydantic also generates JSON Schema via `model_json_schema()`. This is the key to structured outputs — you define a Python class, and Pydantic auto-generates the schema the LLM must follow.
3. **"Instructor is a separate API provider."** — No. Instructor wraps the OpenAI client (or Mistral, Anyscale, etc.). It is syntactic sugar over Pydantic + tool calling.

### Exam angle

Expect a question like:
> "When we execute functions using OpenAI's response API with the 'tools' option, where does the function actually get executed?"
>
> (A) On the user's machine.  
> (B) On the OpenAI servers.  
> (C) On a combination of the user's machine and some other (third party) servers.  
> (D) On a combination of the user's machine and OpenAI servers.

**Answer:** (A) — The LLM only picks which function to call and generates the arguments. Your client-side code executes the function.

Another common question:
> "Which of the following statements is true?"
>
> (A) Pydantic is only used for output validation.  
> (B) Instructor does schema validation.  
> (C) Pydantic can generate and validate schema.  
> (D) Instructor automatically generates the tool calls from pydantic schema.

**Answer:** (D) — Instructor wraps Pydantic and OpenAI to auto-generate tool calls. (C) is also true, but (D) is more complete.

---

## Concept 4: BLEU — Precision-Based N-gram Scoring

### The problem it solves

You need to measure how similar a candidate text (LLM output) is to a reference text (gold standard). BLEU measures n-gram precision: what fraction of the candidate's n-grams appear in the reference? It penalizes outputs that are too short (brevity penalty). BLEU is the standard metric for machine translation.

### The intuition (analogy first)

Imagine grading a student's translation of a French sentence:

- **Reference:** "The quick brown fox jumps over the lazy dog."
- **Candidate:** "The quick brown dog jumps over the lazy fox."

You check how many phrases match: "The quick brown" (match), "quick brown dog" (no match — reference has "fox"), etc. BLEU counts these matches at different phrase lengths (unigrams, bigrams, trigrams). If the student writes only "The quick brown," that is 3/9 words correct (precision = 0.33), but the output is way too short. The brevity penalty reduces the score.

### Worked example (real numbers)

```python
from nltk.translate.bleu_score import sentence_bleu

reference = [[
    'The', 'quick', 'brown', 'fox',
    'jumps', 'over', 'the', 'lazy', 'dog'
]]
candidate = [
    'The', 'quick', 'brown', 'dog',
    'jumps', 'over', 'the', 'lazy', 'fox'
]

score = sentence_bleu(reference, candidate)
print(score)  # 0.4597
```

**Why 0.46?**
- **Unigram precision (1-gram):** All 9 words appear in the reference → 9/9 = 1.0.
- **Bigram precision (2-gram):** "The quick", "quick brown", "brown dog" (no match — reference has "brown fox"), "dog jumps" (no match), "jumps over", "over the", "the lazy", "lazy fox" (no match — reference has "lazy dog"). Matches: 5/8 = 0.625.
- **Trigram, 4-gram:** Even lower precision due to word order changes.
- **BLEU = geometric mean of n-gram precisions × brevity penalty.** The geometric mean of 1.0, 0.625, and lower 3-gram/4-gram scores is ~0.46.

### The formula (with meaning)

```
BLEU = BP × exp(Σ w_n × log(p_n))

where:
  BP = brevity penalty = exp(1 - |r|/|c|) if |c| < |r|, else 1
  p_n = n-gram precision = (matched n-grams) / (total candidate n-grams)
  w_n = weight (typically 1/N, so all n-grams contribute equally)
  |r| = reference length, |c| = candidate length
```

**Brevity penalty (BP):**
- If candidate is shorter than reference, BP < 1 (penalty).
- If candidate is same length or longer, BP = 1 (no penalty).
- Why? Without BP, the candidate "The" would have perfect unigram precision (1/1 = 1.0) but is clearly a bad output.

### Common misunderstandings

1. **"BLEU measures semantic similarity."** — No. BLEU is purely syntactic — it counts exact n-gram matches. "The fast brown fox" vs "The quick brown fox" has low BLEU, even though 'fast' and 'quick' are synonyms.
2. **"Higher BLEU always means better translation."** — No. BLEU is precision-focused. It does not measure recall (did the candidate cover all reference content?) or fluency (is the output grammatical?).
3. **"BLEU of 0.46 means 46% of words match."** — No. BLEU is a geometric mean of n-gram precisions, not a simple word-match percentage. The formula is more complex.

### Exam angle

Expect a question like:
> "Which metric is most suitable for evaluating machine translation quality?"
>
> (A) ROUGE-L  
> (B) BLEU  
> (C) BERTScore  
> (D) METEOR

**Answer:** (B) BLEU — it is the standard metric for translation. (D) METEOR is also used for translation but less common. (A) ROUGE-L is for summarization. (C) BERTScore is for semantic similarity, not phrase-level translation quality.

---

## Concept 5: ROUGE — Recall-Oriented N-gram Overlap

### The problem it solves

BLEU measures precision (what fraction of candidate n-grams are in the reference). ROUGE measures recall (what fraction of reference n-grams are in the candidate). For summarization, recall is more important: did the summary capture all the key content from the reference? ROUGE-1 counts unigram matches, ROUGE-2 counts bigram matches, ROUGE-L uses Longest Common Subsequence (captures word order).

### The intuition (analogy first)

Imagine summarizing a 500-word article into 50 words. A good summary must cover the article's main points (high recall), even if it adds a few extra words (lower precision is tolerable). ROUGE asks: "Of the important words in the reference summary, how many appear in the candidate summary?" If the reference mentions "revenue," "iPhone 15," and "EU regulation," and your summary includes all three, ROUGE-1 recall is high.

### Worked example (real numbers)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'], use_stemmer=True
)
scores = scorer.score(
    'The quick brown dog jumps over the lazy fox.',
    'The quick brown fox jumps over the lazy dog.'
)
print(scores)
# rouge1: Score(precision=1.0, recall=1.0, fmeasure=1.0)
# rougeL: Score(precision=0.778, recall=0.778, fmeasure=0.778)
```

**Why ROUGE-1 = 1.0 but ROUGE-L = 0.778?**
- **ROUGE-1:** All unigrams in the candidate ("The", "quick", "brown", "fox", ...) appear in the reference, and vice versa. Unigram recall = 9/9 = 1.0, unigram precision = 9/9 = 1.0.
- **ROUGE-L:** Longest Common Subsequence (LCS) = "The quick brown jumps over the lazy" (7 tokens). LCS does not include "dog" or "fox" because their positions are swapped. ROUGE-L precision = LCS / candidate length = 7/9 ≈ 0.778. ROUGE-L recall = 7/9 ≈ 0.778.

### The formula (with meaning)

```
ROUGE-N = Σ(matched N-grams) / Σ(reference N-grams)

ROUGE-L = LCS(candidate, reference) / length(reference)

where:
  LCS = Longest Common Subsequence (order-sensitive)
  use_stemmer=True → reduce words to stems before matching
    (e.g., 'running' → 'run')
```

**ROUGE-1 vs ROUGE-2:**
- ROUGE-1: unigram recall (does the candidate mention all the important words?)
- ROUGE-2: bigram recall (does the candidate preserve phrase-level structure?)

**ROUGE-L:**
- Captures sentence-level structure better than ROUGE-N because LCS considers word order.
- Penalizes reordering more harshly than ROUGE-1.

### Common misunderstandings

1. **"ROUGE measures precision."** — No. ROUGE is recall-oriented. The denominator is *reference* n-grams, not candidate n-grams. (ROUGE-1 precision is reported, but recall is the primary metric.)
2. **"ROUGE-L is just ROUGE-1 with longest matches."** — No. ROUGE-L uses Longest Common Subsequence (a dynamic programming algorithm), which is order-sensitive. "A B C D" and "D C B A" have LCS = 1 (only one word in order), even though all four words match.
3. **"High ROUGE means the summary is good."** — Not always. A summary that copies the reference verbatim has ROUGE = 1.0 but may be too long or not fluent. ROUGE does not measure factual accuracy or coherence.

### Exam angle

Expect a question like:
> "Which metric is most suitable for evaluating summarization quality?"
>
> (A) BLEU  
> (B) ROUGE  
> (C) BERTScore  
> (D) METEOR

**Answer:** (B) ROUGE — it is recall-oriented, so it measures whether the summary captures all the reference content. (A) BLEU is precision-oriented (used for translation). (C) BERTScore is semantic (used for chatbots/QA). (D) METEOR is for translation.

---

## Concept 6: METEOR — F-score, Stems & Synonyms

### The problem it solves

BLEU is precision-only, ROUGE is recall-oriented. METEOR balances both via F-score (harmonic mean) and adds linguistic awareness: it matches word stems ("running" ↔ "run") and synonyms ("fast" ↔ "quick") via WordNet. It also penalizes fragmented matches (if matched words are not consecutive, the score drops).

### The intuition (analogy first)

Imagine grading a translation exam:

- **BLEU:** You only count how many phrases the student got right. If they write "The fast brown fox" instead of "The quick brown fox," you mark "fast" as wrong because it is not an exact match.
- **METEOR:** You recognize that "fast" is a synonym of "quick," so you count it as a match. You also check if the student missed any key words (recall) and whether the matched words are in the right order (chunk penalty).

### Worked example (real numbers)

```python
from nltk.translate import meteor_score
from nltk import word_tokenize
import nltk
nltk.download('wordnet')

reference = "The quick brown fox jumps over the lazy dog"
candidate = "The fast brown fox jumps over the lazy dog"

score = meteor_score.meteor_score(
    [word_tokenize(reference)],
    word_tokenize(candidate)
)
print(score)  # 0.99
```

**Why 0.99?**
- **Exact matches:** "The", "brown", "fox", "jumps", "over", "the", "lazy", "dog" → 8/9 tokens.
- **Synonym match:** "fast" ↔ "quick" via WordNet → 1/9 tokens.
- **Precision:** 9/9 = 1.0 (all candidate tokens match).
- **Recall:** 9/9 = 1.0 (all reference tokens covered).
- **Chunk penalty:** All matches are consecutive (one chunk) → no penalty.
- **METEOR = 2 × (P × R) / (P + R) - chunk_penalty ≈ 0.99.**

Compare to BLEU on the same pair: BLEU = 0.46 (because "fast" does not match "quick" at the n-gram level). METEOR scores 0.99 because of synonym matching.

### The formula (with meaning)

```
METEOR = 2 × (P × R) / (P + R) - chunk_penalty

where:
  P = precision = (matched tokens) / (candidate tokens)
  R = recall = (matched tokens) / (reference tokens)
  chunk_penalty = penalizes fragmented matches
    (more chunks → higher penalty)
```

**Matching hierarchy:**
1. Exact token match
2. Stem match ("running" → "run")
3. Synonym match via WordNet ("fast" → "quick")

**Chunk penalty:**
- If matched words are consecutive, they form one chunk → low penalty.
- If matched words are scattered, they form many chunks → high penalty.
- Example: "The dog quick fox" vs "The quick brown fox" → matched words "The", "quick", "fox" are non-consecutive → 3 chunks → higher penalty.

### Common misunderstandings

1. **"METEOR is just BLEU with synonyms."** — No. METEOR uses F-score (balances precision and recall), stem matching, synonym matching, AND chunk penalty. BLEU uses only precision + brevity penalty.
2. **"Higher METEOR always beats higher BLEU."** — Not necessarily. METEOR correlates better with human judgments on average, but for specific tasks (e.g., translation into morphologically rich languages), BLEU may be more appropriate.
3. **"METEOR works for all languages."** — No. Synonym matching requires WordNet, which is best-developed for English. For other languages, METEOR falls back to stem matching only.

### Exam angle

Expect a question like:
> "A candidate translation is 'The fast brown fox jumps' and the reference is 'The quick brown fox leaps.' Which metric will score this highest?"
>
> (A) BLEU  
> (B) ROUGE-1  
> (C) METEOR  
> (D) BERTScore

**Answer:** (D) BERTScore or (C) METEOR — both handle semantic similarity. METEOR matches "fast" ↔ "quick" via synonyms. BERTScore uses contextual embeddings, which also capture "fast" ≈ "quick". (A) BLEU and (B) ROUGE-1 only match exact tokens, so they penalize "fast" vs "quick" and "jumps" vs "leaps."

---

## Concept 7: BERTScore — Semantic Similarity via Contextual Embeddings

### The problem it solves

All the previous metrics (BLEU, ROUGE, METEOR) are lexical: they match surface-form tokens (or stems/synonyms). BERTScore uses BERT contextual embeddings to measure semantic similarity: "The fast brown fox" and "The quick brown fox" have high BERTScore because 'fast' and 'quick' are semantically similar in context, even if they are different tokens.

### The intuition (analogy first)

Imagine grading an essay:

- **Lexical metrics (BLEU/ROUGE):** You count how many exact words match the reference essay. If the student writes "automobile" instead of "car," you mark it wrong.
- **BERTScore:** You encode each word in context (BERT embeddings). "automobile" and "car" have high cosine similarity in embedding space, so you count them as a match. You also check if the student covered all the reference points (recall) and did not add irrelevant words (precision).

### Worked example (real numbers)

```python
import torch
from bert_score import score

cands = ['The quick brown dog jumps over the lazy fox.']
refs = ['The quick brown fox jumps over the lazy dog.']

P, R, F1 = score(cands, refs, lang='en', verbose=True)

print(f'Precision: {P.mean():.4f}')  # ~0.9640
print(f'Recall:    {R.mean():.4f}')  # ~0.9640
print(f'F1:        {F1.mean():.4f}')  # ~0.9640
```

**Why 0.964?**
- BERT encodes each token in context: "dog" in "brown dog jumps" gets a different embedding than "dog" in "lazy dog."
- For each candidate token, BERTScore finds the most similar reference token (cosine similarity). "dog" (in candidate) has high similarity with "dog" (in reference), even though their positions differ. "fox" (in candidate) has high similarity with "fox" (in reference).
- Because "dog" and "fox" are both animals in similar contexts, their embeddings are close, so swapping them only slightly reduces BERTScore.
- Compare to BLEU (0.46), ROUGE-L (0.778), and METEOR (0.99) on the same pair. BERTScore is the second-highest because it is semantic.

### The formula (with meaning)

```
BERTScore Precision = (1 / |C|) × Σ_{c in C} max_{r in R} cos_sim(E(c), E(r))
BERTScore Recall    = (1 / |R|) × Σ_{r in R} max_{c in C} cos_sim(E(c), E(r))
BERTScore F1        = 2 × (P × R) / (P + R)

where:
  E(c) = BERT embedding of candidate token c
  E(r) = BERT embedding of reference token r
  cos_sim(a, b) = (a · b) / (|a| |b|)  (cosine similarity)
```

**Precision:** For each candidate token, find the most similar reference token. Average over all candidate tokens. (Did the candidate avoid irrelevant words?)

**Recall:** For each reference token, find the most similar candidate token. Average over all reference tokens. (Did the candidate cover all reference content?)

### Common misunderstandings

1. **"BERTScore is always better than BLEU/ROUGE."** — No. BERTScore measures semantic similarity, but if your task requires exact phrase matches (e.g., code generation, formatting), lexical metrics may be more appropriate. Also, BERTScore is much slower (requires BERT inference).
2. **"BERTScore requires fine-tuning BERT on your task."** — No. BERTScore uses pre-trained BERT (e.g., `bert-base-uncased`). You can specify a different checkpoint via `model_type=`, but the default works for most tasks.
3. **"BERTScore measures factual accuracy."** — No. BERTScore measures token-level embedding similarity. If the candidate says "Paris is the capital of Germany" and the reference says "Paris is the capital of France," BERTScore will be high (most tokens match), even though the fact is wrong. Use fact-checking tools (e.g., NLI models) for factuality.

### Exam angle

Expect a question like:
> "Which score is most suitable for evaluating a chatbot that needs to understand user intent (not exact wording)?"
>
> (A) ROUGE-L  
> (B) BERTScore  
> (C) BLEU  
> (D) METEOR

**Answer:** (B) BERTScore — it is semantic (handles paraphrases, synonyms). (A) ROUGE-L, (C) BLEU, and (D) METEOR are lexical (penalize wording changes more harshly).

---

## Concept 8: Perplexity — Language Model Fluency

### The problem it solves

Perplexity measures how "surprised" a language model is by a test sequence. Lower perplexity means the model assigns high probability to the actual next tokens (fluent, expected text). Higher perplexity means the model assigns low probability (awkward, unexpected text). Perplexity is used to compare language models, not to evaluate task-specific outputs (use BLEU/ROUGE for that).

### The intuition (analogy first)

Imagine reading a sentence word-by-word:

- **Expected next word:** "The cat sat on the [mat]." You are not surprised by "mat" → low perplexity.
- **Unexpected next word:** "The cat sat on the [volcano]." You are very surprised → high perplexity.

A good language model should assign high probability to "mat" and low probability to "volcano." Perplexity quantifies this: it is the exponential of the average negative log-probability per token.

### The formula (with meaning)

```
Perplexity = exp(cross-entropy) = exp(- (1/N) × Σ log P(token_i | context))

where:
  N = number of tokens
  P(token_i | context) = model's probability for the i-th token given previous tokens
```

**Lower perplexity = better model:**
- If the model assigns P = 0.8 to the correct token, log(0.8) ≈ -0.22, perplexity ≈ 1.2.
- If the model assigns P = 0.01 to the correct token, log(0.01) ≈ -4.6, perplexity ≈ 100.

### Common misunderstandings

1. **"Perplexity measures output quality for a specific task."** — No. Perplexity measures language model fluency (how well the model predicts the test set), not task-specific correctness. Use BLEU/ROUGE/METEOR for task evaluation.
2. **"Lower perplexity always means better generation."** — Not always. A model with very low perplexity on the training set may be overfitting (it memorizes training data but does not generalize). Perplexity should be measured on a held-out test set.

### Exam angle

Perplexity is less commonly tested than BLEU/ROUGE, but you might see:
> "Which metric measures language model fluency (not task-specific accuracy)?"
>
> (A) BLEU  
> (B) Perplexity  
> (C) ROUGE-L  
> (D) BERTScore

**Answer:** (B) Perplexity.

---

## Putting it together

Here is how the concepts fit:

1. **Three ways to run LLMs** (Transformers, HF Hosted, Ollama) set the foundation: you need to run models before you can evaluate them. Exam questions test trade-offs (privacy, speed, setup).
2. **Constrained decoding** (LogitsProcessor, RegexMask) solves the problem of invalid outputs. You mask disallowed tokens at each generation step. This is a precursor to structured outputs.
3. **Structured outputs** (Pydantic, Instructor, tool calling) enforce valid JSON by auto-generating schemas and forcing the model to conform. This is the production-ready solution for reliable LLM outputs.
4. **Evaluation metrics** (BLEU, ROUGE, METEOR, BERTScore) measure output quality:
   - **BLEU:** Precision-focused, for translation. Penalizes word reordering harshly.
   - **ROUGE:** Recall-focused, for summarization. ROUGE-L uses LCS (order-sensitive).
   - **METEOR:** F-score + stems + synonyms + chunk penalty. Balances precision and recall.
   - **BERTScore:** Semantic similarity via embeddings. Best for paraphrase-tolerant tasks (chatbots, QA).
5. **Perplexity** measures language model fluency (lower = better), not task-specific accuracy.

**Exam strategy:**
- For "which metric for which task" questions, remember: BLEU → translation, ROUGE → summarization, BERTScore → chatbots/QA.
- For "which approach to run LLMs" questions, check: privacy (Ollama/Transformers), speed (local > hosted if on MPS/GPU), setup (Ollama easiest).
- For "where does function execution happen" questions, answer: client-side (tool calling only generates arguments).

---

## Check yourself (5 questions)

### Question 1
A developer wants to run Llama 3 on a Mac M3 laptop with no internet connection, using the least setup effort. Which approach should they choose?

(A) HuggingFace Transformers Locally (torch + transformers)  
(B) HuggingFace Hosted Inference API  
(C) Ollama  
(D) OpenAI API

**Answer: (C)**  
Ollama is local (no internet), easy to set up (`brew install ollama`), and does not require torch. (B) and (D) need internet. (A) requires more setup.

---

### Question 2
Which of the following is NOT enforced by the LogitsProcessor?

(A) Only allowed tokens are generated.  
(B) Some of the allowed tokens are generated preferentially.  
(C) The output follows the input grammar completely.  
(D) Partial matches from nested grammars.

**Answer: (C)**  
LogitsProcessor has no lookahead, so it cannot guarantee the *complete* output satisfies the grammar. It can "paint itself into a corner." (A), (B), and (D) are true.

---

### Question 3
When we execute functions using OpenAI's response API with the "tools" option, where does the function actually get executed?

(A) On the user's machine.  
(B) On the OpenAI servers.  
(C) On a combination of the user's machine and some other (third party) servers.  
(D) On a combination of the user's machine and OpenAI servers.

**Answer: (A)**  
The LLM only decides which function to call and generates the arguments. Your client-side code executes the function.

---

### Question 4
Which metric is most suitable for evaluating summarization quality (did the summary capture all the key content)?

(A) BLEU  
(B) ROUGE  
(C) BERTScore  
(D) METEOR

**Answer: (B)**  
ROUGE is recall-oriented — it measures whether the summary covers all the reference content. (A) BLEU is precision-focused (translation). (C) BERTScore is semantic (chatbots/QA). (D) METEOR is for translation.

---

### Question 5
A candidate translation is "The fast brown fox jumps" and the reference is "The quick brown fox leaps." Which metric will score this pair highest?

(A) BLEU  
(B) ROUGE-1  
(C) METEOR  
(D) BERTScore

**Answer: (C) or (D) (both acceptable)**  
METEOR matches "fast" ↔ "quick" via synonyms. BERTScore uses contextual embeddings, which also capture "fast" ≈ "quick". (A) BLEU and (B) ROUGE-1 only match exact tokens, so they heavily penalize synonym differences.

---

## Quick reference (for last-day revision)

### Three Ways to Run LLMs
| Approach | Needs torch? | Needs internet? | Speed | Privacy | Best for |
|----------|--------------|----------------|-------|---------|----------|
| Transformers Locally | ✅ | ❌ | ⚡ Fast (MPS/GPU) | 🔒 Full | Research, fine-tune |
| HF Hosted API | ❌ | ✅ | 🐢 Slow (network) | ⚠️ Data sent | Quick prototypes |
| Ollama | ❌ | ❌ | ⚡ Fast (local) | 🔒 Full | 7B+ models, daily use |

### Constrained Decoding
- **LogitsProcessor:** Masks disallowed tokens at each step (set logit to −∞).
- **RegexMask:** Uses `regex.match(..., partial=True)` to test if appending a token keeps output consistent with a regex.
- **Limitation:** No lookahead — can get stuck. Use structured decoding (OpenAI JSON mode) for grammar guarantees.

### Structured Outputs
- **Pydantic:** `BaseModel` validates types, coerces values, generates JSON Schema via `model_json_schema()`.
- **Tool calling:** `tools=[...], tool_choice={...}` forces model to emit JSON conforming to schema.
- **Instructor:** Wraps OpenAI client. `response_model=MyModel` handles schema generation, tool calling, and validation automatically.

### Evaluation Metrics
| Metric | Measures | Best For | Formula |
|--------|----------|----------|---------|
| BLEU | N-gram precision + BP | Machine translation | `BP × exp(Σ w_n log(p_n))` |
| ROUGE | N-gram recall or LCS | Summarization | `Σ(matched N-grams) / Σ(ref N-grams)` |
| METEOR | F-score + stems + synonyms | Translation & generation | `2PR/(P+R) - chunk_penalty` |
| BERTScore | Cosine sim of embeddings | Chatbots, QA (semantic) | `max cos_sim per token` |
| Perplexity | LM fluency | Comparing LMs | `exp(- Σ log P / N)` |

**Key distinctions:**
- BLEU: precision-only, penalizes reordering.
- ROUGE: recall-oriented, ROUGE-L uses LCS (order-sensitive).
- METEOR: balances P & R, handles synonyms.
- BERTScore: semantic, slowest, best for paraphrases.

### Common Exam Traps
1. **Tool calling executes on OpenAI servers.** — False. Client executes.
2. **LogitsProcessor guarantees grammar correctness.** — False. No lookahead.
3. **BLEU measures semantic similarity.** — False. It is lexical (n-grams).
4. **Higher ROUGE-1 always means better summary.** — False. It does not measure coherence or factuality.
5. **Pydantic is only for validation.** — False. It also generates JSON Schema.
