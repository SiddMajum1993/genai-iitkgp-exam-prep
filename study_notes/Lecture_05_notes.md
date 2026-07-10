# Lecture 5 - Prompting APIs, Structured Outputs, LLM Evaluations
**Instructor:** Prof. Sourangshu Bhattacharya

## TL;DR (5 bullets)
- Three approaches for running LLMs: local transformers with GPU acceleration (MPS/CUDA), HuggingFace hosted API (cloud-based, rate-limited), and Ollama (local, zero-config, no torch dependency).
- Function calling in OpenAI API executes on the user's machine, not OpenAI servers. OpenAI only returns structured JSON arguments.
- Structured outputs use constrained decoding (LogitsProcessor for token masking, regex partial matching) or JSON schema validation (Pydantic + instructor library) to guarantee valid format.
- Pydantic provides schema validation with type coercion, auto-generated JSON schemas for tool calling, and validation errors. Dataclasses accept wrong types silently. Instructor library wraps OpenAI client for automatic structured output.
- Evaluation metrics: ROUGE (recall-oriented, summarization), BLEU (precision-oriented with brevity penalty, translation), METEOR (F-score with stems/synonyms), BERTScore (semantic similarity via embeddings).

## Exam-relevant concepts

### Three approaches for running LLMs

- **Definition:** Local transformers, HuggingFace API, Ollama.
- **Formula / numbers:** Transformers need torch. HuggingFace API is rate-limited. Ollama uses localhost:11434.
- **When it matters (exam trap):** Transformers approach requires torch and can use MPS (Apple Silicon GPU) or CUDA. HuggingFace API does not need torch but requires internet and is rate-limited. Ollama is local, no torch needed, best for 7B+ models.
- **Common confusion:** Thinking HuggingFace API needs torch. It does not. Only local transformers need torch.

### Function execution location in OpenAI API

- **Definition:** When using tools parameter in OpenAI API, functions execute on the user's machine, not OpenAI servers.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** OpenAI returns structured JSON arguments via tool_calls. The actual function execution happens client-side.
- **Common confusion:** Thinking OpenAI executes the function. It only generates the function call arguments.

### Device selection for local inference

- **Definition:** PyTorch device selection for model placement.
- **Formula / numbers:** "mps" for Apple Silicon (M1/M2/M3/M4), "cuda" for NVIDIA GPU, "cpu" for fallback.
- **When it matters (exam trap):** torch.backends.mps.is_available() checks for Apple Silicon GPU. model.to(device) moves model weights to GPU/CPU.
- **Common confusion:** Not moving inputs to the same device as model causes errors.

### Tokenization mechanics

- **Definition:** Text to integer IDs and back. Token IDs are what models process.
- **Formula / numbers:** tokenizer.encode() returns list of integer IDs. tokenizer.decode(id) returns string token.
- **When it matters (exam trap):** Leading spaces in tokens matter. Token for http is ' http' not 'http'. Constraints operate on tokens, not characters.
- **Common confusion:** Thinking constraints work on character level. They work on token level.

### Logits and masking

- **Definition:** Raw unnormalized log-probability scores output by model's final layer. One per vocabulary token.
- **Formula / numbers:** P(token i) = exp(Li) / Sigma exp(Lj). Setting Li = -infinity makes exp(-infinity) = 0, so probability is zero.
- **When it matters (exam trap):** Masking sets logits to -infinity for disallowed tokens. Greedy sampling picks highest logit. Multinomial sampling samples proportionally.
- **Common confusion:** Using -10 penalty instead of -infinity. -10 can be overcome by strong logits. -infinity guarantees zero probability.

### LogitsProcessor for constrained generation

- **Definition:** HuggingFace hook called at every generation step to modify logits.
- **Formula / numbers:** mask = torch.ones_like(scores) * -10. For allowed tokens, mask[:, token_id] = 0.
- **When it matters (exam trap):** LogitsMask allows only specified token IDs. RegexMask uses partial matching to ensure output follows regex.
- **Common confusion:** LogitsProcessor does not guarantee grammatical correctness. It can paint model into corner. Use structured decoding for JSON.

### Partial regex matching

- **Definition:** regex library's partial=True returns match if string is consistent with pattern so far, even if not complete.
- **Formula / numbers:** regex.match(text, partial=True)
- **When it matters (exam trap):** Standard re.match() fails on incomplete strings. regex.match() with partial=True succeeds if string could become valid.
- **Common confusion:** Using Python's built-in re module. It does not support partial matching. Install regex package.

### LogitsProcessor vs Structured Decoding

- **Definition:** LogitsProcessor is token-by-token masking. Structured decoding uses grammar/parse state.
- **Formula / numbers:** LogitsProcessor has no lookahead. Structured decoding uses live DFA for valid paths.
- **When it matters (exam trap):** LogitsProcessor for simple flat constraints (digits-only). Structured decoding for grammar correctness (JSON, nested schemas).
- **Common confusion:** LogitsProcessor does not enforce complete grammar compliance. Only structured decoding guarantees valid JSON.

### OpenAI Structured Decoding mechanisms

- **Definition:** Constrains token generation to match JSON schema at each step.
- **Formula / numbers:** 6 mechanisms: Schema-Constrained Token Selection, Incremental Parsing, Finite-State Guidance, Type Enforcement, Required Fields Tracking, Controlled Termination.
- **When it matters (exam trap):** Invalid continuations are pruned before sampling. Decoder tracks required fields. Generation stops only when JSON is complete.
- **Common confusion:** Thinking prompt-based "return JSON" is reliable. It is fragile. Use structured decoding or tool calling.

### Pydantic validation

- **Definition:** Schema validation for Python with type coercion and auto-generated JSON schemas.
- **Formula / numbers:** model_json_schema() generates schema. model_validate_json() parses and validates.
- **When it matters (exam trap):** Pydantic coerces compatible types (string "10" becomes integer 10). Raises ValidationError if coercion fails (string "13.4" cannot be int).
- **Common confusion:** Thinking dataclasses validate types. They do not. Pydantic does.

### Pydantic vs dataclass

- **Definition:** Dataclasses accept wrong types silently. Pydantic validates on construction.
- **Formula / numbers:** @dataclass accepts age="10" as string. BaseModel coerces to int.
- **When it matters (exam trap):** LLM outputs may return "42" (string) instead of 42 (integer). Pydantic fixes automatically or raises error.
- **Common confusion:** Using dataclasses for LLM output. Use Pydantic for type safety.

### Instructor library

- **Definition:** Wraps OpenAI client to handle schema generation, tool calling, validation automatically.
- **Formula / numbers:** client = instructor.patch(OpenAI()). Pass response_model=Packages.
- **When it matters (exam trap):** Return value is typed Pydantic object. No parsing step, no json.loads().
- **Common confusion:** Instructor only works with OpenAI. False. It works with any provider supporting OpenAI API format.

### ROUGE metrics

- **Definition:** Recall-Oriented Understudy for Gisting Evaluation. Measures overlap between generated and reference texts.
- **Formula / numbers:** ROUGE-N = Sigma(matched N-grams) / Sigma(reference N-grams). ROUGE-1 uses unigrams. ROUGE-2 uses bigrams. ROUGE-L uses Longest Common Subsequence.
- **When it matters (exam trap):** ROUGE-L captures sentence-level structure and word order. use_stemmer=True reduces words to stems before matching.
- **Common confusion:** ROUGE-L is about LCS (structural recall), not just word overlap. Lower ROUGE-L with swapped words even if ROUGE-1 is high.

### BLEU score

- **Definition:** Bilingual Evaluation Understudy. Measures N-gram precision of candidate against reference.
- **Formula / numbers:** BLEU = BP x exp(Sigma wn x log(pn)). BP = exp(1 - |r|/|c|) when |c| < |r|. Brevity Penalty penalizes short outputs.
- **When it matters (exam trap):** Score range 0 to 1. 1.0 is perfect match. Above 0.3 is reasonable for translation. Supports multiple references.
- **Common confusion:** BLEU is precision-based. ROUGE is recall-based. BLEU sensitive to n-gram order.

### METEOR score

- **Definition:** Metric for Evaluation of Translation with Explicit ORdering. Balances precision and recall, weights recall higher.
- **Formula / numbers:** METEOR = 2 x (P x R) / (P + R). Extra weight to recall. Matches stems and synonyms via WordNet.
- **When it matters (exam trap):** Chunk Penalty penalizes fragmented matches. More chunks = higher penalty for disorder.
- **Common confusion:** METEOR scores higher than BLEU when synonyms are used. METEOR captures paraphrase equivalence.

### BERTScore

- **Definition:** Uses BERT contextual embeddings to measure semantic similarity.
- **Formula / numbers:** Pairwise cosine similarity between candidate and reference tokens. Returns Precision, Recall, F1.
- **When it matters (exam trap):** Correlates better with human judgments than BLEU/ROUGE. lang='en' selects pre-trained BERT model. Slower than ROUGE/BLEU.
- **Common confusion:** BERTScore is semantic. BLEU/ROUGE are lexical. BERTScore captures meaning, not just word overlap.

### Metric comparison

- **Definition:** Each metric measures different dimensions of output quality.
- **Formula / numbers:** ROUGE: recall of N-grams/LCS, fast, no semantic awareness. BLEU: precision of N-grams + BP, fast, no semantic awareness. METEOR: harmonic P&R + stems/synonyms, medium speed, partial semantic. BERTScore: cosine sim of BERT embeddings, slow, full semantic.
- **When it matters (exam trap):** Summarization uses ROUGE (recall priority). Translation uses BLEU + METEOR. Chatbots/QA use BERTScore (semantic equivalence).
- **Common confusion:** Using single metric. Use multiple metrics for comprehensive evaluation.

### Levenshtein edit distance (mentioned in prompt, not in lecture)

- **Definition:** Minimum number of single-character edits (insertions, deletions, substitutions) to transform one string to another.
- **Formula / numbers:** HEART to EARTH = 2 edits (delete H at start, insert H at end).
- **When it matters (exam trap):** Not covered in this lecture. Likely in another lecture or sample paper.
- **Common confusion:** Not applicable to this lecture.

## Diagrams / algorithms described

### Autoregressive token generation

Prompt "2, 4, 8," generates Step 1: 16, Step 2: 32, Step 3: 64, Step 4: 128. At each step, model evaluates every token in vocabulary (approx 50,000 tokens). Highest probability wins. Greedy sampling picks max-probability token. Multinomial sampling adds creative diversity.

### Masking mechanism for constrained generation

Rule: Allow odd numbers only at start or after even number. If last token was odd (e.g. "3"), set all odd-digit logits to -infinity. After masking odds, only evens valid. Generates valid sequences like 1234567, 24142167. Invalid sequences like 1235, 11, 148233 are impossible.

### OpenAI Structured Decoding flow

Candidate & Reference to BERT encoder to contextual vectors to pairwise cosine similarity matrix to greedy match to aggregate P/R/F1.

### Pydantic nested models

DateRange model with start and end datetime.date fields. SearchQuery model with rewritten_query string, published_daterange DateRange, domains_allow_list List[str]. model_dump() serializes entire nested structure to dict for API calls.

## Likely MCQ angles

1. Which approach requires torch? (Local transformers, not HuggingFace API or Ollama)
2. Where do OpenAI tool-call functions execute? (User's machine, not OpenAI servers)
3. What does torch.backends.mps.is_available() check? (Apple Silicon GPU availability)
4. What is the unique advantage of HuggingFace transformers over other APIs? (Access to open-source models with all inner functioning)
5. Which task is not suitable for prompting LLMs? (Forecasting stock prices - requires time-series modeling, not text generation)
6. What does LogitsProcessor not enforce? (Complete grammar compliance - it only masks tokens)
7. Why use regex library instead of re? (Supports partial matching with partial=True)
8. What does Pydantic auto-generate? (JSON schema via model_json_schema())
9. Which statement is true? (Instructor automatically generates tool calls from Pydantic schema, OR Pydantic can generate and validate schema)
10. Which metric is recall-oriented? (ROUGE, not BLEU which is precision-oriented)
11. Which metric captures semantic similarity? (BERTScore, not ROUGE/BLEU/METEOR)
12. Which metric is best for summarization? (ROUGE-L - recall of reference content)
13. Which metric is best for translation? (BLEU + METEOR together)
14. Which metric handles synonyms? (METEOR via WordNet, and BERTScore via embeddings)
15. What is BLEU's brevity penalty? (BP = exp(1 - |r|/|c|) when candidate shorter than reference)

## One-line flashcards (10-20)

- Q: Where do OpenAI tool-call functions execute? -> A: User's machine, not OpenAI servers.
- Q: Which approach needs torch? -> A: Local transformers. HuggingFace API and Ollama do not.
- Q: What device code for Apple Silicon GPU? -> A: "mps" (Metal Performance Shaders).
- Q: What does tokenizer.encode() return? -> A: List of integer token IDs.
- Q: What is logit masking for constrained generation? -> A: Setting logit to -infinity makes token probability zero.
- Q: What is partial regex matching? -> A: regex.match(text, partial=True) returns match if string could become valid.
- Q: LogitsProcessor vs structured decoding? -> A: LogitsProcessor for simple constraints. Structured decoding for JSON grammar.
- Q: Pydantic vs dataclass? -> A: Pydantic validates and coerces types. Dataclass accepts wrong types silently.
- Q: What does model_json_schema() do? -> A: Auto-generates JSON schema from Pydantic model.
- Q: What is instructor library? -> A: Wraps OpenAI client for automatic structured output with response_model parameter.
- Q: ROUGE metric focus? -> A: Recall of N-grams or LCS. Best for summarization.
- Q: BLEU metric focus? -> A: Precision of N-grams + brevity penalty. Best for translation.
- Q: METEOR advantage over BLEU? -> A: Matches stems and synonyms via WordNet. Higher F-score weight to recall.
- Q: BERTScore advantage? -> A: Semantic similarity via BERT embeddings. Best human correlation.
- Q: ROUGE-L vs ROUGE-1? -> A: ROUGE-L uses LCS (structure). ROUGE-1 uses unigrams (word overlap).
- Q: BLEU brevity penalty formula? -> A: BP = exp(1 - |r|/|c|) when candidate shorter than reference.
- Q: Which metric is fastest? -> A: ROUGE and BLEU (no model inference). BERTScore is slowest (BERT inference).
- Q: Which metric for chatbots? -> A: BERTScore - semantic equivalence matters more than exact words.
- Q: What is use_stemmer=True for ROUGE? -> A: Reduces words to stems before matching (running to run).
- Q: What does vocab_map in RegexMask do? -> A: Pre-computes string for every token ID once to avoid repeated decode calls.
