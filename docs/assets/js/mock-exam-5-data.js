window.MOCK_EXAM_5 = {
  title: "Mock Exam 5 — Challenge",
  durationMinutes: 90,
  marksPerQuestion: 4,
  sections: [
    {
      name: "Section 1 — Evaluation Metrics & Model Architecture",
      questions: [
        {
          id: "q1",
          stem: "A resume screening system processes 2,000 applications for 100 open positions. The model flags 140 candidates as qualified. Of these, 85 are actually qualified and 55 are false positives. Of the actual qualified candidates, 15 are missed by the model. What is the model's F1 score, and which error type is more costly in this context?",
          options: {
            A: "F1 = 0.607; False Positives are costlier because they waste recruiter time reviewing unqualified candidates.",
            B: "F1 = 0.692; False Negatives are costlier because missing a qualified candidate means losing potential talent and extending the hiring cycle.",
            C: "F1 = 0.750; Neither error is more costly — F1 balances both equally.",
            D: "F1 = 0.850; The model's accuracy of 92.75% shows it is performing well."
          },
          correct: "B",
          explanation: "TP=85, FP=55, FN=15. Precision = 85/(85+55) = 0.607. Recall = 85/(85+15) = 0.850. F1 = 2×(0.607×0.850)/(0.607+0.850) ≈ 0.692. FN (missing good candidates) extends hiring cycles and loses talent — costlier than FP (wasting time on bad resumes, which is annoying but fixable). (A) computes precision, not F1. (C) F1 = 0.692, not 0.750. (D) confuses accuracy with F1.",
          reference: "Learn: 01 ML foundations — F1 score, Precision/Recall trade-offs"
        },
        {
          id: "q2",
          stem: "A COVID-19 screening model at an airport has the following confusion matrix: TP=90, FP=150, FN=10, TN=4750. What is the model's precision, and what does prioritizing Recall over Precision mean for this deployment?",
          options: {
            A: "Precision = 0.375; prioritizing Recall means catching all COVID cases (minimize FN=10) even if many healthy travelers are flagged for secondary testing (high FP=150).",
            B: "Precision = 0.90; prioritizing Recall would reduce accuracy.",
            C: "Precision = 0.60; F1 score is the only valid metric for imbalanced medical data.",
            D: "Precision = 0.900; the model is already optimal because recall is high."
          },
          correct: "A",
          explanation: "Precision = TP/(TP+FP) = 90/(90+150) = 90/240 = 0.375. Only 37.5% of flagged passengers actually have COVID. Recall = TP/(TP+FN) = 90/(90+10) = 0.90 (catches 90% of cases). Prioritizing Recall means accepting the high FP rate (150 false alarms) to minimize FN (missing a real case could cause an outbreak). FP means extra testing (inconvenient); FN means undetected spread (catastrophic). (B) computes recall, not precision. (C) F1 is not 'the only valid' metric. (D) precision is 0.375, not 0.90.",
          reference: "Learn: 01 ML foundations — When to prioritize Recall"
        },
        {
          id: "q3",
          stem: "A research team argues that Vision Transformers (ViT) outperform CNNs on ImageNet classification when both are trained on identical datasets and compute budgets. A colleague counters that CNNs have a structural inductive bias that ViTs lack. Which statement correctly identifies this bias and its trade-off?",
          options: {
            A: "CNNs use ReLU which prevents vanishing gradients; ViTs use LayerNorm which is slower.",
            B: "CNNs encode local connectivity and translation equivariance via convolutions, making them more sample-efficient on small datasets. ViTs treat the image as a sequence of patches with no built-in spatial prior, requiring massive data to learn these properties from scratch.",
            C: "CNNs downsample with pooling; ViTs downsample with patch embeddings — this is the key structural difference.",
            D: "CNNs are encoder-only; ViTs are decoder-only — architecture family determines inductive bias."
          },
          correct: "B",
          explanation: "CNNs have two hard-coded priors: LOCAL CONNECTIVITY (each filter sees a small patch) and TRANSLATION EQUIVARIANCE (same filter weights slide across the image). This makes CNNs sample-efficient — they work well with 50k images. ViTs treat the image as a bag of patches with self-attention (no local bias). They need millions of images to learn spatial structure. On small datasets, CNNs win. On massive datasets (JFT-300M), ViTs can match or beat CNNs. (A) ReLU/LayerNorm are orthogonal. (C) pooling vs patch-embedding is a detail, not the core structural bias. (D) both CNNs and ViTs are encoder-only for classification.",
          reference: "Learn: 02 Neural networks — CNNs vs Vision Transformers"
        },
        {
          id: "q4",
          stem: "A sentiment analysis model using Word2Vec embeddings fails on the sentence: 'The bank refused to loan money because the account was overdrawn.' The word 'bank' is closer to 'river' than 'finance' in the embedding space, causing misclassification. A colleague suggests switching to BERT. Why does BERT solve this problem?",
          options: {
            A: "BERT uses subword tokenization (WordPiece) which splits 'bank' into 'ban' + '##k', disambiguating meanings.",
            B: "BERT is bidirectional while Word2Vec is unidirectional, so BERT sees more context.",
            C: "BERT produces context-dependent embeddings via self-attention over the full sentence. The vector for 'bank' in 'refused to loan' is different from 'bank' in 'river bank', resolving polysemy.",
            D: "BERT is pretrained on more data than Word2Vec, so it has better coverage of rare words."
          },
          correct: "C",
          explanation: "Word2Vec produces ONE static vector per word regardless of context. 'bank' always gets the same vector (averaged over all senses). BERT computes a DIFFERENT vector for 'bank' in each sentence by running self-attention over the tokens. In 'refused to loan', the surrounding tokens ('loan', 'account', 'overdrawn') pull the 'bank' vector toward the financial sense. This is contextualised embedding. (A) subword tokenization helps with OOV words, not polysemy. (B) Word2Vec skip-gram is actually bidirectional in context window; the key difference is static vs contextual. (D) data scale is not the core reason.",
          reference: "Learn: 02 Neural networks — Static vs Contextual embeddings"
        },
        {
          id: "q5",
          stem: "What are the shapes of `output`, `hn`, and `cn` after the following LSTM forward pass?",
          code: "lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)\\nx = torch.randn(16, 30, 128)  # (batch=16, seq=30, input=128)\\noutput, (hn, cn) = lstm(x)",
          options: {
            A: "output: (16, 30, 256), hn: (2, 16, 256), cn: (2, 16, 256)",
            B: "output: (16, 30, 512), hn: (4, 16, 256), cn: (4, 16, 256)",
            C: "output: (16, 30, 256), hn: (4, 16, 256), cn: (4, 16, 256)",
            D: "output: (16, 30, 512), hn: (2, 16, 512), cn: (2, 16, 512)"
          },
          correct: "B",
          explanation: "bidirectional=True doubles the first dim of hn/cn. num_layers=2. So hn/cn have shape (2×2, batch, hidden) = (4, 16, 256). output has shape (batch, seq, hidden×2) = (16, 30, 512) because bidirectional concatenates forward and backward hidden states at every timestep. (A) ignores bidirectional doubling of hn/cn and output. (C) output should be 512, not 256. (D) hn/cn should be (4, 16, 256), not (2, 16, 512).",
          reference: "Learn: 02 Neural networks — LSTM bidirectional shapes"
        },
        {
          id: "q6",
          stem: "A startup is building a customer support chatbot that answers questions by retrieving relevant FAQ documents and generating a conversational response. The team debates whether to use an encoder-only, decoder-only, or encoder-decoder architecture. Which choice is correct, and why?",
          options: {
            A: "Encoder-only (BERT), because the task is understanding user intent and matching it to FAQs, which is a classification/retrieval task.",
            B: "Decoder-only (GPT), because RAG requires generating a natural-language answer by conditioning on retrieved context, which is a generation task. The retrieval step uses a separate bi-encoder; the chatbot itself is a generator.",
            C: "Encoder-decoder (T5), because RAG is a sequence-to-sequence task where the input is the query and the output is the answer.",
            D: "Decoder-only (GPT), because only decoder-only models support in-context learning from retrieved documents."
          },
          correct: "B",
          explanation: "RAG = Retrieve (bi-encoder ranks docs) + Generate (LLM synthesizes answer from context). The generation step requires a decoder-only or encoder-decoder model. Decoder-only (GPT, LLaMA) is the modern default for RAG because it can condition on arbitrarily long retrieved context and generate fluent responses. Encoder-only (BERT) cannot generate autoregressively. Encoder-decoder (T5) works but is less common for chat (decoder-only is simpler and scales better). (A) BERT is used for retrieval, not generation. (C) RAG is not seq2seq in the classical sense (the 'input sequence' is query+docs, not a fixed-length source). (D) in-context learning is not exclusive to decoder-only.",
          reference: "Learn: 02 Neural networks — Architecture families, RAG systems"
        }
      ]
    },
    {
      name: "Section 2 — LLM Decoding, APIs & Tooling",
      questions: [
        {
          id: "q7",
          stem: "A medical diagnosis assistant must produce identical outputs for the same patient symptoms to ensure reproducibility across multiple doctors. The team also wants the model to avoid hallucinations by sticking to high-probability tokens. Which decoding configuration is most appropriate?",
          options: {
            A: "Temperature = 0.0, top_p = 1.0, because temperature controls randomness and top_p allows all tokens.",
            B: "Temperature = 0.0, top_k = 1, because temperature=0 makes the distribution a spike at argmax, and top_k=1 enforces greedy decoding.",
            C: "Top-p = 0.95, top_k = 50, because these settings balance diversity and quality.",
            D: "Beam search with beam_size = 5, because beam search explores multiple hypotheses and selects the most probable sequence."
          },
          correct: "B",
          explanation: "Temperature = 0 makes softmax(logits/T) collapse to a one-hot at argmax → deterministic. top_k=1 explicitly keeps only the top token. Together they guarantee greedy decoding (same input → same output, always). (A) top_p=1.0 means 'include all tokens' which reintroduces randomness if temperature > 0. (C) top_p=0.95 and top_k=50 both allow sampling from multiple tokens → non-deterministic. (D) beam search is deterministic but not required here; greedy is simpler and faster.",
          reference: "Learn: 04 LLM decoding & APIs — Greedy decoding, temperature"
        },
        {
          id: "q8",
          stem: "A language model generates text with the following token probability distribution (sorted descending): A=0.35, B=0.25, C=0.18, D=0.14, E=0.04, F=0.02, G=0.01, H=0.005, I=0.003, J=0.002. The decoding parameters are top_k = 5 and top_p = 0.90. How many tokens are in the final sampling set, and which constraint is binding?",
          options: {
            A: "5 tokens; top_k is binding.",
            B: "4 tokens; top_p is binding.",
            C: "5 tokens; neither is binding.",
            D: "6 tokens; top_p is binding."
          },
          correct: "B",
          explanation: "Top-k=5 selects the 5 most probable tokens: {A, B, C, D, E}. Top-p=0.90 accumulates tokens until the cumulative probability first reaches ≥ 0.90: A=0.35, A+B=0.60, A+B+C=0.78, A+B+C+D=0.92 → stop (0.92 ≥ 0.90). Top-p yields {A, B, C, D} — a 4-token set. Intersection {A,B,C,D,E} ∩ {A,B,C,D} = {A,B,C,D} = 4 tokens. Top-p is the binding constraint because it truncates the sampling set to 4 while top-k would have allowed 5.",
          reference: "Learn: 04 LLM decoding & APIs — Top-k ∩ Top-p"
        },
        {
          id: "q9",
          stem: "A developer is building a multi-step agentic workflow using the OpenAI Chat Completions API. The workflow requires: (i) a system-level constraint that tools must never modify production databases, (ii) injecting the result of a previous tool call back into the conversation, and (iii) limiting the assistant's response to 150 tokens. Which API features directly satisfy all three?",
          options: {
            A: "system role (constraint) | tool role (inject result) | max_tokens",
            B: "developer role (constraint) | function schema | stop sequences",
            C: "system role | assistant role (tool call JSON) | frequency_penalty",
            D: "user role | tool schema | temperature=0"
          },
          correct: "A",
          explanation: "(i) System-level constraint → system role message sets behavior guardrails. (ii) Tool result → tool role message returns function output to the model. (iii) Hard token limit → max_tokens. (B) developer role is a newer alternative to system, but function schema defines available tools, not results. Stop sequences don't guarantee 150 tokens. (C) assistant role is for model responses or few-shot, not injecting tool results. (D) user role is for human input; tool schema defines tools but doesn't inject results; temperature doesn't cap length.",
          reference: "Learn: 04 LLM decoding & APIs — OpenAI API roles, parameters"
        },
        {
          id: "q10",
          stem: "A university student wants to run Qwen2.5-72B locally for a weekend hackathon project. She has a MacBook Pro with 64 GB RAM, no external GPU, and wants to avoid sending data to external APIs. She is comfortable with slower inference (5-10 tokens/sec) and can quantize the model. Which approach fits her constraints?",
          options: {
            A: "Hugging Face Hosted Inference API, because it requires no local GPU and is the fastest option.",
            B: "Ollama with a quantized GGUF model (Q4_K_M), because it runs entirely locally on CPU/RAM, supports 72B models with 64GB RAM via quantization, and requires no API keys.",
            C: "Local Transformers with torch, because it gives full control and privacy.",
            D: "Claude API, because it offers the best quality for agentic workflows."
          },
          correct: "B",
          explanation: "Constraints: 64GB RAM, no GPU, must be local (no API). Qwen2.5-72B in FP16 needs ~144GB VRAM. Quantized to 4-bit (Q4), it needs ~40GB, which fits in 64GB RAM. Ollama supports GGUF quantized models and runs on CPU. (A) HF Hosted API is cloud-based (violates 'no external APIs'). (C) Local Transformers needs GPU for 72B inference; CPU-only would be unusably slow without quantization, and torch quantization setup is complex. (D) Claude API is external.",
          reference: "Learn: 04 LLM decoding & APIs — Model access options, Ollama"
        },
        {
          id: "q11",
          stem: "A team implements a custom LogitsProcessor to constrain JSON generation. They replace the `regex` library's `partial=True` matching with Python's built-in `re.fullmatch()`. The generator starts producing truncated JSON objects that stop mid-key. What is the root cause?",
          code: "# Original (correct):\\nif not self.regex.match(partial_string, partial=True):\\n    mask_token()\\n\\n# Buggy replacement:\\nif not re.fullmatch(self.pattern, partial_string):\\n    mask_token()",
          options: {
            A: "`re.fullmatch` is slower than `regex.match`, causing timeout errors.",
            B: "`re.fullmatch` requires the ENTIRE pattern to match the partial string. At token 5, '\"nam' does not match the full JSON schema, so valid continuation tokens are masked out, halting generation prematurely.",
            C: "`re.fullmatch` does not support named capture groups required for JSON parsing.",
            D: "`re.fullmatch` uses a different regex dialect incompatible with the pattern."
          },
          correct: "B",
          explanation: "The key difference: `regex.match(partial=True)` returns True if the partial string is a VALID PREFIX of the pattern (even if incomplete). `re.fullmatch` only returns True if the ENTIRE pattern matches. When generating '\"name\":', at token 3 we have '\"na', which is a valid prefix of a JSON key but not a complete key. `re.fullmatch` returns None, so the processor masks out all continuation tokens, killing generation. The `regex` library's `partial=True` is essential for token-by-token constrained decoding.",
          reference: "Learn: 04 LLM decoding & APIs — LogitsProcessor, partial matching"
        },
        {
          id: "q12",
          stem: "A legal document summarization system evaluates two candidate summaries against a reference. Summary A has high n-gram overlap with the reference but reorders clauses. Summary B paraphrases heavily with different words but preserves the original clause order. The team needs a metric that rewards structural similarity (clause order) while still checking for content recall. Which metric aligns best, and what is its key limitation for legal text?",
          options: {
            A: "BLEU, because it measures n-gram precision and implicitly rewards order via higher-order n-grams (bigrams, trigrams).",
            B: "ROUGE-L, because it uses Longest Common Subsequence (LCS) to capture both recall of reference tokens and their sequential ordering. Its limitation is that n-gram overlap does not verify legal semantics — two clauses can share words but have opposite obligations (e.g., 'shall indemnify' vs 'shall not indemnify').",
            C: "BERTScore, because it uses contextual embeddings to capture semantic similarity and handles paraphrasing. Its limitation is that it ignores word order entirely.",
            D: "METEOR, because it aligns words using synonyms and stem matching. Its limitation is that it is designed for translation, not summarization."
          },
          correct: "B",
          explanation: "ROUGE-L computes the Longest Common Subsequence between candidate and reference, rewarding both token recall and ORDER (longer LCS = better order preservation). It is recall-oriented, fitting the goal. Its limitation: surface-form overlap ≠ legal meaning. 'Party A shall indemnify Party B' vs 'Party A shall not indemnify Party B' differ by one word but have opposite legal force. ROUGE-L sees 80% word overlap and gives a high score, missing the semantic flip. (A) BLEU is precision-oriented, wrong direction. (C) BERTScore captures semantics but indeed ignores order (it is a bag-of-embeddings cosine). (D) METEOR balances P/R (not recall-focused) and handles morphology, but the key issue is legal semantics, not stems.",
          reference: "Learn: 05 Structured outputs & evaluation — ROUGE-L, NLP metrics for legal text"
        }
      ]
    },
    {
      name: "Section 3 — Prompting Techniques & NLP Metrics",
      questions: [
        {
          id: "q13",
          stem: "Consider the following prompt structure. Which prompting technique does this example demonstrate?",
          code: "Q: A farmer has 15 apples. He sells 6 and buys 8 more. How many does he have?\\nReasoning path 1: 15 - 6 = 9, then 9 + 8 = 17.\\nReasoning path 2: Sells 6 → 15 - 6 = 9. Buys 8 → 9 + 8 = 17.\\nReasoning path 3: Start=15, change=-6+8=+2, final=15+2=17.\\nMajority answer: 17\\n\\nQ: A store has 42 items. 13 are sold, 7 are returned. How many now?\\nA:",
          options: {
            A: "Few-shot Chain-of-Thought, because intermediate reasoning is shown.",
            B: "Self-Consistency, because multiple reasoning paths are sampled for the same question and the final answer is determined by majority vote.",
            C: "Tree-of-Thoughts, because the prompt explores multiple branches of reasoning.",
            D: "Zero-shot Chain-of-Thought, because the prompt includes 'Let's think step by step.'"
          },
          correct: "B",
          explanation: "Self-Consistency samples MULTIPLE reasoning paths for the same question (path 1, 2, 3) and takes the majority vote (17 appears in all 3, so answer=17). This is the defining pattern of Self-Consistency. Few-shot CoT (A) shows reasoning but doesn't sample multiple paths or vote. Tree-of-Thoughts (C) would show a branching exploration with backtracking, not parallel paths voting. Zero-shot CoT (D) uses a trigger phrase, not multiple sampled paths.",
          reference: "Learn: 07 Advanced prompting — Self-Consistency"
        },
        {
          id: "q14",
          stem: "A planning task requires exploring multiple approaches, evaluating each approach's feasibility, backtracking if an approach hits a dead-end, and systematically refining the plan. The task is: 'Design a 5-day itinerary for a family vacation in Japan with a $5000 budget, including flights, hotels, and activities.' Which prompting technique is most appropriate?",
          options: {
            A: "Chain-of-Thought, because the task requires step-by-step reasoning.",
            B: "Self-Consistency, because we want multiple itineraries and select the best via voting.",
            C: "Tree-of-Thoughts, because the task requires exploring multiple branches (Tokyo vs Kyoto first?), evaluating feasibility at each step (budget remaining?), and backtracking if a branch violates constraints (e.g., no hotels under $100 in Tokyo).",
            D: "Few-shot prompting, because providing 2-3 example itineraries will guide the model."
          },
          correct: "C",
          explanation: "Tree-of-Thoughts (ToT) is designed for planning and search problems where you explore branches, evaluate each branch (feasibility, cost), and backtrack if needed. The vacation planning task has multiple decision points (city order, hotel tier, activities) and hard constraints (budget cap). ToT lets the model explore 'Tokyo first' vs 'Kyoto first', evaluate each branch, and backtrack if budget is exceeded. CoT (A) is linear, no backtracking. Self-Consistency (B) samples multiple full solutions but doesn't guide the search process. Few-shot (D) helps but doesn't structure the exploration.",
          reference: "Learn: 07 Advanced prompting — Tree-of-Thoughts"
        },
        {
          id: "q15",
          stem: "An attacker trains a shadow model on publicly available data similar to the target LLM's training set, then queries the target model and the shadow model with the same inputs. If the target model assigns higher confidence to a specific data point than the shadow model, the attacker infers that the data point was in the target's training set. Which attack is this, and which CIA triad component is violated?",
          options: {
            A: "Model Extraction; Integrity, because the attacker is stealing the model's functionality.",
            B: "Membership Inference; Confidentiality, because the attacker learns which specific data records (e.g., individuals' medical records) were in the training set, potentially violating privacy.",
            C: "Adversarial Attack; Availability, because the attacker disrupts the model's predictions.",
            D: "Data Poisoning; Integrity, because the attacker manipulates training data."
          },
          correct: "B",
          explanation: "Membership Inference attacks determine whether a specific record was in the training set by comparing the target model's confidence to a shadow model's. If the target overfits to training data, it assigns higher confidence to training samples. This violates CONFIDENTIALITY — knowing 'Alice's medical record was in the training set' leaks private information. Model Extraction (A) steals the model's weights or behavior, not training set membership. Adversarial (C) crafts inputs to cause misclassification. Poisoning (D) injects bad data during training.",
          reference: "Learn: 06 AI safety — Membership Inference, CIA triad"
        },
        {
          id: "q16",
          stem: "An Automatic Prompt Engineering system generates 6 candidate prompts and evaluates them by computing the AVERAGE log-probability of correct answers on a 100-question validation set, then LENGTH-NORMALIZES by dividing by the number of tokens in each answer. The scores are: Prompt A: -0.032, Prompt B: -0.018, Prompt C: -0.095, Prompt D: -0.041, Prompt E: -0.009, Prompt F: -0.068. Which prompt should the APE system select?",
          options: {
            A: "Prompt F, because the most negative value indicates the strongest signal.",
            B: "Prompt E, because log-probabilities are negative, and -0.009 is closest to zero, indicating the highest likelihood.",
            C: "Prompt A, because it has the second-best score and is more robust than the top-ranked prompt.",
            D: "Prompt C, because lower log-probability means higher confidence."
          },
          correct: "B",
          explanation: "Log-probabilities are negative. Closer to zero = higher probability. -0.009 is closest to zero → highest average token-level likelihood → best prompt. Length-normalization (dividing by token count) prevents the metric from penalizing longer answers. exp(-0.009) ≈ 0.991 (very high probability per token). exp(-0.095) ≈ 0.909 (much lower). Prompt E wins. (A) and (D) reverse the interpretation. (C) invents a 'robustness' criterion not in standard APE.",
          reference: "Learn: 08 Prompt security & APE — Log-prob scoring, length normalization"
        },
        {
          id: "q17",
          stem: "Using the standard Levenshtein distance (insertions, deletions, and substitutions each cost 1), what is the minimum edit distance between 'KITCHEN' and 'CHICKEN'?",
          options: {
            A: "2",
            B: "3",
            C: "4",
            D: "6"
          },
          correct: "C",
          explanation: "KITCHEN and CHICKEN share the suffix 'EN' (2 chars) and both have 7 letters. Building the DP table (or aligning character by character: K→C, I→H, T→I, C=C, H→K, E=E, N=N), the optimal path uses 4 substitutions at positions 0, 1, 2, and 4. No insertion/deletion shortcut helps because the only shared substring outside the suffix is a single C, which does not reduce the count below 4. (A) 2 is a common trap — it would only work for pairs like HEART/EARTH where one character shifts across the whole word. (B) 3 is the KITTEN→SITTING answer, not this pair. (D) 6 counts each substitution as two operations (delete + insert), which is not optimal.",
          reference: "Learn: 08 Prompt security & APE — Levenshtein distance"
        },
        {
          id: "q18",
          stem: "A research team wants to improve retrieval quality for a specialized medical QA system. They observe that standard RAG (embed query → retrieve top-k docs → generate answer) often retrieves documents that match query keywords but miss the underlying medical concept. For example, the query 'What causes sudden cardiac arrest in athletes?' retrieves documents about 'cardiac arrest' but misses key documents about 'hypertrophic cardiomyopathy' (the underlying cause). Which RAG variant addresses this by generating a more conceptually rich retrieval query?",
          options: {
            A: "Standard RAG with a better embedding model (e.g., switch from Sentence-BERT to BGE-Large).",
            B: "HyDE (Hypothetical Document Embeddings), where the LLM first generates a hypothetical answer (a detailed passage about hypertrophic cardiomyopathy causing arrest), then embeds that passage and retrieves documents similar to the hypothetical answer, not the original query.",
            C: "Self-RAG, where the model retrieves, generates, then critiques its own answer and retrieves again if confidence is low.",
            D: "Multi-Query Retrieval, where the original query is rewritten into 3-5 paraphrases and all are used for retrieval in parallel."
          },
          correct: "B",
          explanation: "HyDE generates a hypothetical answer (LLM imagines a detailed response) and embeds THAT, not the query. The hypothetical answer is richer in domain terms (e.g., 'hypertrophic cardiomyopathy, ventricular arrhythmia, sudden death') than the original query. Retrieving based on the hypothetical answer finds documents that match the CONCEPT, not just keywords. (A) A better embedding model helps but doesn't solve the keyword-vs-concept gap. (C) Self-RAG critiques and re-retrieves but doesn't enrich the query. (D) Multi-Query paraphrases the query but doesn't add domain concepts.",
          reference: "Learn: 09 RAG fundamentals, 12 Agentic RAG — HyDE"
        }
      ]
    },
    {
      name: "Section 4 — Vector Indexing & RAG Security",
      questions: [
        {
          id: "q19",
          stem: "A vector database stores 100,000,000 (100 million) document embeddings. Each embedding has 768 dimensions and is stored as float32 (4 bytes per value). Product Quantization is applied with m = 48 sub-vectors and k = 256 centroids per codebook. How much storage (in GB) is required for the PQ-compressed codes, and what is the compression ratio?",
          options: {
            A: "PQ storage = 4.8 GB; Compression ratio = 64×",
            B: "PQ storage = 4.8 GB; Compression ratio = 256×",
            C: "PQ storage = 48 GB; Compression ratio = 6.4×",
            D: "PQ storage = 307.2 GB; No compression applied"
          },
          correct: "A",
          explanation: "Raw storage = N × D × 4 = 100,000,000 × 768 × 4 = 307,200,000,000 bytes = 307.2 GB. PQ storage = N × m = 100,000,000 × 48 bytes = 4,800,000,000 bytes = 4.8 GB (since k=256 fits in 1 byte). Compression ratio = 307.2 / 4.8 = 64×. Note: m=48 is larger than typical (usually 8-16), which reduces compression ratio compared to m=8 (which would give 768×4/8 = 384× compression). With m=48, we get 768×4/48 = 64×.",
          reference: "Learn: 11 Vector indexing — Product Quantization storage"
        },
        {
          id: "q20",
          stem: "Continuing from Q19 (N=100M, D=768, m=48, k=256), at query time, how many entries are in the Product Quantization lookup table (LUT), and how many lookups and additions are required to compute the approximate distance from the query to ONE database vector?",
          options: {
            A: "LUT: 768 entries; per-vector: 768 lookups + 767 additions",
            B: "LUT: 12,288 entries; per-vector: 48 lookups + 47 additions",
            C: "LUT: 256 entries; per-vector: 256 lookups + 255 additions",
            D: "LUT: 48 entries; per-vector: 48 lookups + 0 additions"
          },
          correct: "B",
          explanation: "LUT size = m × k = 48 × 256 = 12,288 entries. The LUT stores the distance from each of the 48 query sub-vectors to each of the 256 centroids in the corresponding codebook. To score one database vector: look up 48 codes (one per sub-vector) in the LUT, then sum them (47 additions). Total: 48 lookups + 47 additions = 95 operations. Compare to brute-force dot product: 768 multiplications + 767 additions = 1535 operations. PQ is ~16× faster per vector.",
          reference: "Learn: 11 Vector indexing — PQ query mechanics"
        },
        {
          id: "q21",
          stem: "Continuing from Q19-Q20 (N=100M, D=768, m=48, k=256), the engineering team decides to INCREASE k from 256 to 1024 centroids per codebook to improve recall (finer quantization). How does this affect PQ storage per vector, LUT size, and compression ratio?",
          options: {
            A: "PQ storage per vector increases from 48 bytes to 96 bytes (2 bytes per code for k=1024); LUT size increases from 12,288 to 49,152 entries; compression ratio drops from 64× to 32×.",
            B: "PQ storage per vector remains 48 bytes; LUT size increases to 49,152 entries; compression ratio remains 64×.",
            C: "PQ storage per vector increases to 192 bytes; LUT size increases to 49,152 entries; compression ratio drops to 16×.",
            D: "k does not affect storage, only query speed."
          },
          correct: "A",
          explanation: "k=1024 = 2^10 requires 10 bits per code, rounded up to 2 bytes. PQ storage = N × m × 2 = 100M × 48 × 2 = 9.6 GB (vs 4.8 GB with k=256). Compression ratio = 307.2 GB / 9.6 GB = 32× (vs 64× with k=256). LUT size = m × k = 48 × 1024 = 49,152 entries (vs 12,288 with k=256). Larger k improves recall (finer quantization) but doubles memory and LUT size.",
          reference: "Learn: 11 Vector indexing — PQ with different k values"
        },
        {
          id: "q22",
          stem: "A company deploys a vector search system for 5 million product catalog embeddings (D=512). They need sub-50ms p99 query latency, can tolerate 5% recall drop, and have 32 GB RAM available. Which index should they choose, and why would IVF+PQ be overkill?",
          options: {
            A: "IndexFlatIP, because 5M vectors at D=512 fits in 10 GB and brute-force can achieve sub-50ms with SIMD optimizations.",
            B: "HNSW, because it offers O(log N) search with high recall, fits easily in 32 GB RAM at 5M scale, and provides sub-10ms p99 latency. IVF+PQ is overkill because it requires training, adds quantization error, and the memory savings (32× compression) are unnecessary when the index already fits in RAM.",
            C: "IVF+PQ, because it is the industry standard for all vector search deployments.",
            D: "ColBERT, because token-level embeddings provide the highest retrieval accuracy."
          },
          correct: "B",
          explanation: "5M vectors × 512 dims × 4 bytes = 10 GB raw. Add HNSW graph overhead (M=32 → ~128 bytes per vector) = 5M × 128 = 640 MB → total ~11 GB, fits in 32 GB easily. HNSW gives sub-10ms latency and 95%+ recall at this scale. IVF+PQ is designed for 50M+ vectors where RAM is the bottleneck. At 5M, the training cost, quantization error, and complexity of IVF+PQ are unnecessary. (A) IndexFlatIP is O(N), likely >50ms at 5M scale. (C) IVF+PQ is not always the right choice. (D) ColBERT stores per-token vectors (5M docs × 100 tokens = 500M vectors = 1 TB), blows RAM.",
          reference: "Learn: 11 Vector indexing — Index selection, HNSW vs IVF+PQ"
        },
        {
          id: "q23",
          stem: "A SaaS platform serves three customer segments with separate knowledge bases: Enterprise customers (KB_Enterprise), SMB customers (KB_SMB), and Free-tier users (KB_Public). An incoming query must be routed to the correct KB before retrieval. The CTO mandates: (1) no LLM call per query (too expensive), (2) semantic understanding (regex is too brittle), (3) must work offline (no external API). Which query routing strategy satisfies all constraints?",
          options: {
            A: "LLM-based routing (prompt the model to return JSON with the target KB).",
            B: "Embedding-based routing: for each KB, embed 30-50 representative historical queries offline. At query time, embed the incoming query, compute cosine similarity to each KB's sample embeddings, route to the KB with the highest average similarity. No LLM call, works offline, handles paraphrasing.",
            C: "Rule-based routing with hardcoded keyword lists (e.g., if query contains 'SSO' or 'SAML' → KB_Enterprise).",
            D: "Train a BERT classifier on historical query logs to predict the target KB."
          },
          correct: "B",
          explanation: "Embedding-based routing (semantic router) satisfies all constraints: (1) No LLM call (just embedding inference, can run locally or offline). (2) Semantic understanding (cosine similarity on embeddings handles paraphrasing). (3) Offline-capable (embeddings and sample profiles are precomputed). (A) violates constraint 1 (LLM call). (C) violates constraint 2 (regex/keywords are brittle). (D) is valid but requires training data and a trained model (not mentioned in the question), and the question asks for the strategy that matches the lecture's 'embedding-based' approach.",
          reference: "Learn: 11 Vector indexing — Query routing, semantic router"
        },
        {
          id: "q24",
          stem: "A financial services RAG system indexes internal documents tagged with clearance levels (clearance_level: 1=Public, 2=Internal, 3=Confidential, 4=Restricted). A junior analyst (clearance_level=2) submits a query: 'What is the bank's exposure to commercial real estate in Q3?' A senior security architect reviews the system design and identifies a critical flaw: the retrieval gateway is NOT injecting RBAC filters. What is the correct enforcement mechanism, and why are the alternatives insufficient?",
          options: {
            A: "Post-retrieval filtering: retrieve all documents, then filter by clearance level before sending to the LLM.",
            B: "Retrieval gateway reads the user's clearance level from the authentication token and injects a metadata filter (where: {clearance_level: {$lte: 2}}) into the vector database query BEFORE retrieval, physically preventing unauthorized documents from being returned. This enforces least privilege at the data layer.",
            C: "Prompt the LLM with: 'Only answer using documents accessible to clearance level 2. Refuse if sensitive data appears.'",
            D: "Use an LLM-based query classifier to detect sensitive queries (e.g., 'commercial real estate exposure') and block them before retrieval."
          },
          correct: "B",
          explanation: "Security must be enforced at the RETRIEVAL layer, not the LLM layer. The gateway injects a metadata filter into the DB query (where clearance_level ≤ 2) BEFORE retrieval. Documents with clearance_level=3 or 4 are physically never returned, never logged, never enter the LLM context. (A) Post-filtering is too late — sensitive docs have already been retrieved into memory and logs (data leak). (C) LLM prompt is guidance, not enforcement (the docs are already in the context window; prompt injection could bypass). (D) Query classifier guesses intent but has no document-level control (what if the query is innocent but retrieved docs are sensitive?).",
          reference: "Learn: 11 Vector indexing — RAG security, retrieval gateway RBAC"
        }
      ]
    }
  ]
};
