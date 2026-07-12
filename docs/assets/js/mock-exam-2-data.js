window.MOCK_EXAM_2 = {
  title: "Mock Exam 2 — Original Sample Paper",
  durationMinutes: 90,
  marksPerQuestion: 4,
  sections: [
    {
      name: "Section 1 — Evaluation Metrics & Model Architecture",
      questions: [
        {
          id: "q1",
          stem: "A hospital deploys a model to screen patients for a rare but life-threatening disease. The dataset contains 9,900 healthy patients and 100 diseased patients. A naive model that always predicts \"healthy\" achieves 99% accuracy. A data scientist argues this model is useless. Which metric should the team prioritise instead, and why?",
          options: {
            A: "Accuracy, because 99% is objectively high and the model is working correctly.",
            B: "Precision, because it measures how many of the model's positive predictions are actually correct.",
            C: "Recall, because missing a true positive (a real disease case) is the costliest error — the model must catch as many actual positives as possible.",
            D: "F1 Score, because it is always the best metric for any medical task."
          },
          correct: "C",
          explanation: "Recall = TP / (TP + FN) measures how many actual positives the model catches. A False Negative here means a diseased patient is sent home undiagnosed — potentially fatal. Maximising Recall directly minimises this worst-case error. (A) 99% accuracy is an illusion caused by class imbalance — a model that predicts 'healthy' for every patient gets 99% accuracy yet catches zero disease cases. (B) Precision matters when false alarms are costly; here false alarms mean extra tests (inconvenient, not fatal). (D) F1 is the harmonic mean of Precision and Recall — useful when both error types carry equal weight, which is not the case here.",
          reference: "Learn: 01 ML foundations — Recall for imbalanced data"
        },
        {
          id: "q2",
          stem: "A cancer-screening model is evaluated on a test set. Its confusion matrix shows: TP = 80, FP = 20, FN = 10, TN = 890. What is the model's precision, and what does it tell you about the model's behaviour?",
          options: {
            A: "Precision = 80 / (80+10) ≈ 0.89; 89% of actual cancer cases were caught.",
            B: "Precision = 80 / (80+20) = 0.80; 80% of the cases the model flagged as cancer were truly cancerous.",
            C: "Precision = (80+890) / 1000 = 0.97; the overall accuracy of the model.",
            D: "Precision = 80 / (80 + 20 + 10) ≈ 0.73; accounting for missed cases in the denominator."
          },
          correct: "B",
          explanation: "Precision = TP / (TP + FP) = 80 / (80 + 20) = 0.80. Of 100 flagged cases, 80 were genuinely cancerous and 20 were false alarms. Precision answers: 'When the model raises an alarm, how often is it right?' (A) TP / (TP + FN) = 80 / 90 ≈ 0.89 is Recall, not Precision. (C) (TP + TN) / Total = 970 / 1000 = 0.97 is Accuracy. (D) TP / (TP + FP + FN) does not correspond to any standard metric.",
          reference: "Learn: 01 ML foundations — Confusion matrix formulas"
        },
        {
          id: "q3",
          stem: "A computer vision team is building an image classifier. One engineer proposes using a fully connected MLP where every pixel connects to every neuron in the first layer. A second engineer argues CNNs are better. What is the single most important structural difference between a CNN's first layer and an MLP's first layer, and why does it matter for image tasks?",
          options: {
            A: "CNNs use ReLU activation while MLPs use sigmoid — this is the key structural difference.",
            B: "CNNs connect each neuron to only a small local region of the input rather than every pixel, allowing filters to learn spatially localised patterns like edges and textures while drastically reducing the number of parameters.",
            C: "CNNs process images in grayscale while MLPs require RGB — the channel handling is the key difference.",
            D: "CNNs use matrix multiplication while MLPs use element-wise operations — the mathematical operation differs fundamentally."
          },
          correct: "B",
          explanation: "Local connectivity is the defining structural property of a convolutional layer. Each filter connects to a small spatial patch (e.g., 3×3 pixels) rather than all pixels simultaneously. This reflects how visual patterns work — edges, corners, and textures are local structures. It also reduces parameters dramatically: a 3×3 filter needs only 27 weights regardless of image size, versus millions for a fully connected first layer. (A) Both can use ReLU. (C) Both handle grayscale or RGB. (D) Both use matrix multiplication as their core linear operation.",
          reference: "Learn: 02 Neural networks — CNN local connectivity"
        },
        {
          id: "q4",
          stem: "A team trains a Word2Vec model on a large corpus. After training, they run the analogy: vec(\"king\") − vec(\"man\") + vec(\"woman\") and find the nearest vector is vec(\"queen\"). A colleague asks: \"How does Word2Vec learn that king and queen are related without anyone telling it?\" Which explanation is correct?",
          options: {
            A: "Word2Vec reads a handcrafted thesaurus during training that lists king and queen as related royalty terms.",
            B: "Word2Vec learns that king and queen appear in similar contexts — both co-occur with words like \"rules,\" \"kingdom,\" and \"throne\" — so their vectors end up geometrically close, encoding the shared meaning through distributional statistics.",
            C: "Word2Vec assigns king and queen similar vectors because they share the same number of letters and similar spelling patterns.",
            D: "Word2Vec uses a pretrained grammar parser to identify that king and queen are both nouns, placing all nouns in the same region of vector space."
          },
          correct: "B",
          explanation: "Firth (1957): 'A word's meaning is defined by the words that frequently appear around it.' 'King' and 'queen' both appear near 'rules,' 'kingdom,' 'royal,' and 'throne.' Because their context distributions are similar, Word2Vec trains their vectors to be geometrically similar. The analogy king − man + woman ≈ queen works because the vector offset between male and female royalty terms reflects a learned gender direction — an emergent property of distributional patterns. (A) No thesaurus is required. (C) Word2Vec operates on word identities, not spelling. (D) No grammar parser is used.",
          reference: "Learn: 02 Neural networks — Word2Vec distributional hypothesis"
        },
        {
          id: "q5",
          stem: "What do the two print statements output?",
          code: "lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)\nx = torch.randn(8, 15, 64)  # (batch=8, seq=15, input=64)\noutput, (hn, cn) = lstm(x)\nprint(hn.shape)\nprint(cn.shape)",
          options: {
            A: "hn: (8, 15, 32), cn: (8, 15, 32)",
            B: "hn: (2, 8, 32), cn: (2, 8, 32)",
            C: "hn: (1, 8, 32), cn: (1, 8, 32)",
            D: "hn: (2, 15, 32), cn: (2, 15, 32)"
          },
          correct: "B",
          explanation: "LSTM returns hn and cn each with shape (num_layers, batch, hidden_size) = (2, 8, 32). num_layers=2 means both the hidden state and cell state have one entry per layer. (A) (8, 15, 32) is the shape of `output` — it contains hidden states across all 15 timesteps. hn and cn store only the final state. (C) (1, 8, 32) would be correct only if num_layers=1. (D) The 15 confuses seq_len with batch_size; seq_len does not appear in hn/cn.",
          reference: "Learn: 02 Neural networks — LSTM output shapes"
        },
        {
          id: "q6",
          stem: "A team is building a named entity recognition (NER) system to tag people, organisations, and locations in legal documents. They have labelled sentences and access to pretrained models. A colleague argues for a decoder-only model like GPT. Another argues for an encoder-only model like BERT. Who is right, and why?",
          options: {
            A: "Neither — NER requires an encoder-decoder because the output tag sequence is different from the input token sequence.",
            B: "Both are equally suitable — the architecture decision table is just a suggestion.",
            C: "The decoder-only advocate — decoder-only is the industry standard and always outperforms encoder-only regardless of task.",
            D: "The encoder-only advocate — NER is a classification task where every token needs full bidirectional context; BERT-style models are purpose-built for this with the lowest inference cost."
          },
          correct: "D",
          explanation: "Encoder-only wins for understanding/scoring tasks. Bidirectional self-attention gives every token full context in both directions — exactly what per-token classification needs. NER outputs one label per input token — same-length aligned sequences — so no encoder-decoder is required. Decoder-only excels at conversational, instruction-following use cases, not per-token classification. Architecture choice is a real cost decision — different families have genuinely different strengths.",
          reference: "Learn: 02 Neural networks — Architecture families"
        }
      ]
    },
    {
      name: "Section 2 — LLM Decoding, APIs & Tooling",
      questions: [
        {
          id: "q7",
          stem: "A legal-tech company is deploying an LLM to auto-complete contract clauses. The model must always pick the single most grammatically and legally correct continuation — the same output every time the same clause prefix is given. Which decoding strategy is most appropriate, and why?",
          options: {
            A: "Top-p sampling with p = 0.95, because it dynamically adjusts the candidate pool based on confidence.",
            B: "Top-k sampling with k = 50, because a fixed candidate pool improves legal terminology coverage.",
            C: "Greedy decoding or temperature ≈ 0 (near-deterministic), because the task demands the single highest-probability token at each step with no randomness.",
            D: "Temperature = 1.5 with top-k = 80, because legal language benefits from diverse phrasing options."
          },
          correct: "C",
          explanation: "Greedy decoding (or T → 0) picks the single highest-probability token at every step with no stochastic component, guaranteeing identical output for identical input — exactly what legal consistency demands. (A) Top-p is still a sampling strategy — it introduces randomness. (B) Top-k also samples randomly from the top-k tokens. (D) Temperature = 1.5 amplifies randomness far beyond the default, producing highly variable and potentially incoherent text.",
          reference: "Learn: 04 LLM decoding & APIs — Decoding strategies"
        },
        {
          id: "q8",
          stem: "A developer wants to use the OpenAI Chat Completions API for a multi-turn customer support bot. He needs to: (i) inject tool-call results back into the conversation, (ii) prime the model with a few worked examples before the live user query, and (iii) prevent the model from producing more than 150 tokens per reply. Which set of API features directly covers all three requirements?",
          options: {
            A: "tool role message | assistant role (few-shot examples) | max_tokens",
            B: "system role | top-p sampling | stop sequences",
            C: "function schema | temperature = 0 | frequency_penalty",
            D: "developer role | prompt caching | repetition_penalty"
          },
          correct: "A",
          explanation: "The tool role is the designated way to return function/tool results in OpenAI's API. Placing prior assistant turns in the messages array is the standard few-shot technique. max_tokens is the direct hard cap on output length. (B) system sets behavior but cannot inject tool results; stop sequences halt on a string pattern but cannot guarantee a token budget of 150. (C) function schema defines what tools exist but does not inject results; frequency_penalty reduces repetition. (D) prompt caching is a Claude API feature, not OpenAI Chat Completions.",
          reference: "Learn: 04 LLM decoding & APIs — OpenAI API roles"
        },
        {
          id: "q9",
          stem: "A developer wants to prototype an LLM-powered feature over a weekend. She has no GPU, wants to avoid installing PyTorch, but needs access to a large model like Qwen3-32B. She's fine with slightly slow responses and occasional rate limits. Which approach best fits her situation, and why does the alternative with full privacy not suit her?",
          options: {
            A: "Approach 1 (local Transformers), because it has access to the widest model variety.",
            B: "Approach 2 (HF Hosted Inference API), because it requires no local GPU or torch, at the cost of data being sent to HF servers.",
            C: "Approach 3 (Ollama), because it bundles its own runtime and needs only requests.",
            D: "Approach 2 (HF Hosted Inference API), because it is the fastest option and has no rate limits."
          },
          correct: "B",
          explanation: "The HF Hosted Inference API requires only `pip install requests` and no local GPU or PyTorch — ideal for quick prototyping. The privacy trade-off (data sent to HF servers) rules out Ollama as a privacy-equivalent alternative for her. Option D is wrong: the API is slower due to network latency and is rate-limited on the free tier. (A) Local Transformers requires torch and typically a GPU. (C) Ollama runs locally — still needs CPU/RAM for a 32B model.",
          reference: "Learn: 04 LLM decoding & APIs — Model access options"
        },
        {
          id: "q10",
          stem: "In the RegexMask LogitsProcessor, the line `if self.regex.match(partial + token_str, partial=True)` uses the regex library rather than Python's built-in re. A classmate replaces it with `re.match(partial + token_str)`. Which of the following best describes what goes wrong, and why?",
          options: {
            A: "re.match is slower, so generation becomes too slow to be practical.",
            B: "re.match requires a complete match, so tokens that could legally extend the output are masked out prematurely, causing the model to stop generating valid prefixes.",
            C: "re.match does not support character classes, so the URL regex breaks.",
            D: "re.match returns a match object instead of a boolean, causing a TypeError."
          },
          correct: "B",
          explanation: "The critical difference is `partial=True`: a partial match returns True if the string is consistent with the pattern even if incomplete. Standard `re.match` only returns a match when the full pattern is satisfied. At step 5 of generating a URL, 'http://' does not yet satisfy the full regex — `re.match` would return None, masking all legal next tokens and halting generation incorrectly.",
          reference: "Learn: 04 LLM decoding & APIs — LogitsProcessor"
        },
        {
          id: "q11",
          stem: "A team is building an AI-powered legal document assistant. They want to evaluate whether the system correctly captures all key clauses from a source contract in its generated summaries. Which metric is most theoretically aligned with this goal, and why might it still fall short in practice?",
          options: {
            A: "BLEU, because it measures precision of legal terminology.",
            B: "ROUGE-L, because it prioritizes recall of reference content and captures structural ordering via LCS.",
            C: "BERTScore, because legal terms are often synonymous and embeddings handle that.",
            D: "METEOR, because it handles morphological variants like \"indemnified\" vs \"indemnify\"."
          },
          correct: "B",
          explanation: "ROUGE-L's recall orientation aligns with the goal of not missing clauses, and LCS captures sequential structure. It falls short because 'recall of n-grams' doesn't verify legal meaning — two clauses can share words but differ in scope or obligation. BERTScore would actually be a strong complement here, not the primary metric.",
          reference: "Learn: 05 Structured outputs & evaluation — NLP metrics"
        },
        {
          id: "q12",
          stem: "A company deploys an LLM-powered internal assistant that has access to Slack, Gmail, and cloud infrastructure APIs. An attacker embeds a hidden instruction in an invoice PDF sent to an employee, which the assistant processes when asked to summarize it. The assistant then silently sends an email to an external address. Which attack type best describes this, and which component of the CIA Triad is primarily violated?",
          options: {
            A: "Model Poisoning; Integrity",
            B: "Indirect Prompt Injection; Confidentiality",
            C: "Membership Inference; Availability",
            D: "Adversarial Attack; Integrity"
          },
          correct: "B",
          explanation: "This is a classic Indirect Prompt Injection: the malicious instruction is hidden in external data (the PDF), not the user's direct prompt. Confidentiality is primarily violated because sensitive data is exfiltrated to an unauthorized party.",
          reference: "Learn: 06 AI safety — CIA triad + attack types"
        }
      ]
    },
    {
      name: "Section 3 — Prompting Techniques & NLP Metrics",
      questions: [
        {
          id: "q13",
          stem: "Consider the following prompt given to a language model. Which prompting technique does this example illustrate?",
          code: "Q: The odd numbers in this group add up to an even number: 17, 9, 10, 12, 13, 4, 2.\nA: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.\n\nQ: The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.\nA:",
          options: {
            A: "Zero-shot prompting",
            B: "Few-shot prompting (without Chain-of-Thought)",
            C: "Few-shot Chain-of-Thought (CoT) prompting",
            D: "Self-Consistency prompting"
          },
          correct: "C",
          explanation: "The provided example demonstrates few-shot CoT prompting because each in-context example includes an intermediate reasoning step ('Adding all the odd numbers (17, 9, 13) gives 39') before stating the final answer. Plain few-shot prompting would supply only the question and the final answer, without exposing the reasoning. Self-Consistency is not identifiable from a single prompt; it is an inference-time strategy that samples multiple CoT completions and aggregates them by majority vote.",
          reference: "Learn: 07 Advanced prompting — Few-shot CoT"
        },
        {
          id: "q14",
          stem: "Refer to the token probability table below. The decoding parameters are: top_k = 2, top_p = 0.95, temperature = 1.0, do_sample = True. Token probabilities: A=0.13, B=0.04, C=0.22, D=0.16, E=0.01, F=0.18, G=0.07, H=0.10, I=0.06, J=0.03. How many tokens can now be sampled from the intersection of top-k and top-p sampling, and which constraint is binding?",
          options: {
            A: "2 tokens; top-k is the binding constraint.",
            B: "2 tokens; top-p is the binding constraint.",
            C: "8 tokens; top-p is the binding constraint.",
            D: "10 tokens; neither constraint is binding."
          },
          correct: "A",
          explanation: "Top-k = 2 selects only the 2 most probable tokens: {C, F}. Top-p = 0.95 includes tokens until cumulative probability reaches at least 0.95: C+F=0.40, C+F+D+A=0.69, C+F+D+A+H+G+I=0.92, C+F+D+A+H+G+I+B=0.96 → stop (include B). So top-p yields {C, F, D, A, H, G, I, B}, an 8-token set. The intersection is {C, F} ∩ {C, F, D, A, H, G, I, B} = {C, F}, which has 2 tokens. Top-k is the binding constraint because the small k=2 truncates the sampling set far more aggressively than the permissive top-p=0.95.",
          reference: "Learn: 04 LLM decoding & APIs — Top-k ∩ Top-p"
        },
        {
          id: "q15",
          stem: "Using edit-distance (insertions, deletions, and substitutions each cost 1), what is the Levenshtein distance between HEART and EARTH?",
          options: {
            A: "1",
            B: "2",
            C: "4",
            D: "5"
          },
          correct: "B",
          explanation: "Position-by-position substitution gives 4 edits, but the optimal alignment is cheaper. Delete the leading H to obtain EART, then insert H at the end to obtain EARTH, for a total of 2 edits. The point is that optimal alignment can shift characters rather than overwrite them position by position.",
          reference: "Learn: 08 Prompt security & APE — Levenshtein distance"
        },
        {
          id: "q16",
          stem: "In an Automatic Prompt Engineering \"Instruction Search\" run, four candidate prompts receive the following log-probability scores: −0.41, −0.07, −2.13, −0.95. Which candidate is selected, and why?",
          options: {
            A: "−0.07, the value closest to zero corresponds to the highest likelihood.",
            B: "−2.13, because the largest magnitude indicates the strongest signal.",
            C: "−0.95, because it lies near the median of the four candidates.",
            D: "−0.41, because the second-best candidate avoids overfitting."
          },
          correct: "A",
          explanation: "Log-probabilities are negative, and the value closest to zero corresponds to the highest probability: exp(−0.07) ≈ 0.93, versus 0.66 for −0.41, 0.39 for −0.95, and 0.12 for −2.13.",
          reference: "Learn: 08 Prompt security & APE — Log-prob scoring"
        },
        {
          id: "q17",
          stem: "Which statement best describes the core mechanism of Retrieval-Augmented Generation (RAG)?",
          options: {
            A: "The model is fine-tuned on the target corpus so that all task knowledge is stored in its weights.",
            B: "Relevant information is fetched from an external knowledge source at inference time and supplied to the generator as additional context, which is then conditioned on to produce the answer.",
            C: "The model first generates an answer from its parameters and only afterwards retrieves documents to verify it.",
            D: "Retrieval replaces generation entirely; the system returns the single highest-ranked document verbatim."
          },
          correct: "B",
          explanation: "RAG augments the generator's input with passages retrieved from an external store at inference time, grounding the output in knowledge that need not live in the model's weights. Option A describes parametric fine-tuning (knowledge baked into weights); option C reverses the retrieve-then-generate order of standard RAG; option D describes plain search/IR with no generation step.",
          reference: "Learn: 09 RAG fundamentals"
        },
        {
          id: "q18",
          stem: "What most clearly distinguishes agentic RAG from a naive single-pass RAG pipeline?",
          options: {
            A: "Agentic RAG never retrieves documents and answers purely from parametric memory.",
            B: "Agentic RAG replaces the language model with a hand-written rule-based finite-state machine.",
            C: "Agentic RAG differs only in using a larger embedding dimension for the index.",
            D: "Agentic RAG can issue multiple retrieval steps, route across tools or sources, and iterate."
          },
          correct: "D",
          explanation: "Naive RAG is a fixed linear pipeline: retrieve once, then generate. Agentic RAG places an LLM-driven controller in the loop that plans, chooses retrieval actions, can perform multi-hop or iterative retrieval, select among sources or tools, and reflect on intermediate results.",
          reference: "Learn: 12 Agentic RAG"
        }
      ]
    },
    {
      name: "Section 4 — Vector Indexing & RAG Security",
      questions: [
        {
          id: "q19",
          stem: "You're storing 10,000,000 vectors, each 1024-dimensional float32. You apply Product Quantization with m = 16 sub-vectors, each encoded against a codebook of k = 256 centroids. What is the raw storage for all vectors before PQ?",
          options: {
            A: "10.24 GB",
            B: "10 MB",
            C: "40.96 GB",
            D: "1 GB"
          },
          correct: "C",
          explanation: "Each vector is 1024 × 4 bytes = 4096 bytes. Across 10M vectors: 4096 × 10,000,000 ≈ 40.96 GB.",
          reference: "Learn: 11 Vector indexing — PQ raw storage"
        },
        {
          id: "q20",
          stem: "(Continued from Q19 — same PQ setup) What is the storage after PQ (codes only)?",
          options: {
            A: "10 MB",
            B: "80 MB",
            C: "40.96 GB",
            D: "160 MB"
          },
          correct: "D",
          explanation: "256 centroids fit in 8 bits = 1 byte, and there are m = 16 sub-vectors, so 16 bytes per vector. Across 10M: 16 × 10,000,000 = 160 MB.",
          reference: "Learn: 11 Vector indexing — PQ compressed storage"
        },
        {
          id: "q21",
          stem: "(Continued from Q19 — same PQ setup) At search time, how big is the per-query distance lookup table?",
          options: {
            A: "10 Million entries",
            B: "1,024 entries",
            C: "4,096 entries",
            D: "16 entries"
          },
          correct: "C",
          explanation: "The lookup table is m × k = 16 × 256 = 4,096 entries, computed once per query.",
          reference: "Learn: 11 Vector indexing — PQ lookup table"
        },
        {
          id: "q22",
          stem: "(Continued from Q19 — same PQ setup) At search time, how many lookups and additions does it take to score one database vector? (Answer in the form #lookups, #additions.)",
          options: {
            A: "1024, 1023",
            B: "8, 8",
            C: "16, 16",
            D: "16, 15"
          },
          correct: "D",
          explanation: "Scoring one database vector requires m = 16 lookups and 15 additions (to sum the 16 partial distances together).",
          reference: "Learn: 11 Vector indexing — PQ query mechanics"
        },
        {
          id: "q23",
          stem: "How is Embedding-Based routing set up and matched?",
          options: {
            A: "Train a deep classifier on millions of labeled queries; match by softmax.",
            B: "Build profiles of 20–50 sample questions per database; embed the query and match by cosine similarity.",
            C: "Ask an LLM to emit a JSON route decision for every query.",
            D: "Map regex keywords to each database."
          },
          correct: "B",
          explanation: "Profiles of 20–50 samples per DB; cosine match on embedded query; no LLM, offline-capable. (C/D describe LLM- and rule-based routing.)",
          reference: "Learn: 11 Vector indexing — Query routing"
        },
        {
          id: "q24",
          stem: "A junior engineer asks \"What are the latest salary bands?\" To enforce Least Privilege so retrieval returns zero docs, where/how is AuthZ applied?",
          options: {
            A: "In the system prompt, instructing the LLM to refuse if salary data appears in context.",
            B: "In the retrieval gateway, by reading the user's role/token and injecting an RBAC metadata filter into the Vector DB query before retrieval.",
            C: "By post-filtering retrieved chunks with a cross-encoder for clearance level.",
            D: "By having the LLM router classify the query as \"sensitive\" and block generation."
          },
          correct: "B",
          explanation: "The gateway injects an RBAC metadata filter at the DB level before retrieval, so unauthorized docs physically can't be returned.",
          reference: "Learn: 11 Vector indexing — RAG security / Least Privilege"
        }
      ]
    }
  ]
};
