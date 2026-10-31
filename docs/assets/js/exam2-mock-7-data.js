window.EXAM2_MOCK_7 = {
  "title": "Mock Exam 7 — Scenarios and design choices",
  "durationMinutes": 90,
  "marksPerQuestion": 4,
  "sections": [
    {
      "name": "Section 1 — Fine-tuning & alignment",
      "questions": [
        {
          "id": "q1",
          "stem": "An insurer's support team wants a chatbot that answers policy questions and quotes the current clause. The policy wording is revised every week, and the base model's tone and format are already fine. Which approach should they try first?",
          "options": {
            "A": "Fine-tune the model every week on the new policy text so it memorises the changes",
            "B": "Paste the entire policy library into every prompt and skip retrieval",
            "C": "RAG over the policy documents, so answers use the latest text",
            "D": "Run full fine-tuning once on last year's policies to get the best accuracy"
          },
          "correct": "C",
          "explanation": "Correct: (C). The problem is knowledge that changes, not behaviour. RAG changes what the model knows at query time by fetching the current clause, and the model can quote it. The rule of thumb is to go prompting first, then RAG, then fine-tuning. (A) Fine-tuning changes how a model behaves. It is unreliable for adding many new facts, and the model would be out of date between runs. (B) Pasting a whole document set is not RAG. It is too long, costly and noisy, and it may not fit in the context window. (D) The policies change weekly, so last year's text is already stale, and the behaviour was already fine. Review: Chapter 01, Concept 2 — The three pathways: Prompting, RAG, Fine-tuning."
        },
        {
          "id": "q2",
          "stem": "A startup wants to fine-tune a 34B-parameter model on a single 48 GB GPU. FP16 weights take 2 bytes per parameter, and 4-bit NF4 weights take about 0.5 byte per parameter. Which plan is realistic?",
          "options": {
            "A": "QLoRA, because the frozen 4-bit base needs about 34 × 0.5 = 17 GB, leaving room for adapters and activations",
            "B": "LoRA on an FP16 base, because only the small adapters are trained and the base needs little memory",
            "C": "Full fine-tuning with gradient checkpointing, which removes the need for optimizer states",
            "D": "Full fine-tuning with a paged optimizer, which moves the weights to CPU RAM"
          },
          "correct": "A",
          "explanation": "Correct: (A). Work out the memory step by step. The 4-bit base is 34 billion × 0.5 byte = 17 GB. The LoRA matrices, their small optimizer states and the activations (kept low with gradient checkpointing) fit in the remaining 31 GB or so. QLoRA quantizes only the frozen base, and the small A and B matrices stay in higher precision. (B) The frozen base must still sit in GPU memory. In FP16 it needs 34 × 2 = 68 GB, which is already above 48 GB. (C) Checkpointing only lowers activation memory. Full fine-tuning needs about 12 bytes per parameter, so 34 × 12 = 408 GB. (D) A paged optimizer only spills optimizer states when memory spikes. Weights and gradients, about 136 GB, would still need the GPU. Review: Chapter 04, Putting it together: what QLoRA actually does."
        },
        {
          "id": "q3",
          "stem": "A team trains with the configuration below on 2 GPUs. Which statement about this run is correct?",
          "code": "lora = LoraConfig(r=16, lora_alpha=32,\n                  target_modules=[\"q_proj\", \"v_proj\"],\n                  lora_dropout=0.05)\nargs = TrainingArguments(\n    learning_rate=2e-4,\n    per_device_train_batch_size=4,\n    gradient_accumulation_steps=8,\n    warmup_ratio=0.03,\n    max_grad_norm=1.0,\n    bf16=True)",
          "options": {
            "A": "The update is scaled by r/α = 0.5, and the effective batch is 64",
            "B": "The update is scaled by α/r = 2, but the effective batch is only 32 because GPUs do not add to it",
            "C": "The update is scaled by 2, but 2e-4 is far too high for LoRA and will make training diverge",
            "D": "The update is scaled by α/r = 2, the effective batch is 4 × 8 × 2 = 64, and 2e-4 is a normal LoRA learning rate"
          },
          "correct": "D",
          "explanation": "Correct: (D). Read the config. lora_alpha / r = 32 / 16 = 2, so the low-rank update BA is multiplied by 2 before it is added to W₀x. The effective batch is per-device batch 4 × accumulation 8 × 2 GPUs = 64. The learning rate 2e-4 sits at the top of the normal LoRA range, and max_grad_norm=1.0 with bf16 is the standard safe setup. (A) The LoRA scale is α/r, which is 32/16 = 2, not r/α. (B) Effective batch = micro-batch × accumulation steps × number of GPUs, so 4 × 8 × 2 = 64. Each GPU processes its own micro-batch. (C) The course gives 1e-4 to 2e-4 as the usual LoRA range. Full fine-tuning is the one that needs about 1e-5 to 2e-5. Review: Chapter 04, Concept 4 — The two dials: rank r and alpha α."
        },
        {
          "id": "q4",
          "stem": "During a fine-tuning run, training loss falls steadily for 3 epochs. Validation loss reaches its minimum near step 600, then rises while training loss keeps falling, and sample answers on the validation set get worse. What is the best next step?",
          "options": {
            "A": "Raise the learning rate so validation loss can catch up with training loss, which closes the gap",
            "B": "Go back to the step-600 checkpoint (or train fewer epochs) and add more varied data",
            "C": "Train for more epochs, because training loss is still improving",
            "D": "Increase the LoRA rank so the model has more capacity to fit the validation set"
          },
          "correct": "B",
          "explanation": "Correct: (B). Training loss down and validation loss up is overfitting: the model memorises training examples instead of generalising. The standard fixes are to stop at the best validation checkpoint (early stopping), use fewer epochs or a lower learning rate, and add more diverse data. (A) A higher learning rate makes the gap worse and risks instability. The model is already fitting the training set too well. (C) Falling training loss with rising validation loss is the overfitting signature. More epochs would make it worse. (D) More capacity helps underfitting, but here the model is memorising. Extra capacity would overfit faster. Review: Chapter 02, Part 3 — Training."
        },
        {
          "id": "q5",
          "stem": "Which of the following is NOT a sound step when preparing a chat-style SFT dataset?",
          "options": {
            "A": "Formatting every example with the exact chat template the model expects at inference",
            "B": "Setting prompt-token labels to −100 so the loss is computed only on the answer",
            "C": "Removing near-duplicate examples before splitting into train and validation sets",
            "D": "Fitting normalisation statistics on all the data before splitting"
          },
          "correct": "D",
          "explanation": "Correct: (D). The rule is to split first, then fit any transform on the training set only. Fitting statistics on the full dataset lets information from validation and test data leak into training, which makes scores look better than they really are. This is the data-leakage trap. (A) This is sound. A template mismatch between training and inference quietly hurts quality. (B) This is sound. Label −100 is ignored by the loss, so the model learns to answer instead of copying prompts. (C) This is sound. Duplicates waste compute, raise memorisation, and leak into the validation set if they are split apart. Review: Chapter 02, Part 2 — The data."
        },
        {
          "id": "q6",
          "stem": "A team serves one frozen model with a separate soft prompt for each task. The hidden size is d = 1024, each prompt uses t = 50 virtual tokens, and the model has L = 24 layers. Roughly how many trainable parameters does one task need with prompt tuning, and with prefix tuning (2·L·t·d)?",
          "options": {
            "A": "Prompt tuning: 51,200. Prefix tuning: about 2.46 million (2,457,600)",
            "B": "Prompt tuning: 51,200. Prefix tuning: about 1.23 million, counting the keys but not the values",
            "C": "Prompt tuning: about 2.46 million. Prefix tuning: 51,200",
            "D": "Prompt tuning: 102,400. Prefix tuning: about 2.46 million, because prompt tuning also trains keys"
          },
          "correct": "A",
          "explanation": "Correct: (A). Prompt tuning trains t × d = 50 × 1,024 = 51,200 numbers at the input only. Prefix tuning trains a key prefix and a value prefix at every layer, 2 × L × t × d = 2 × 24 × 50 × 1,024 = 2,457,600. Both are still tiny compared with the full model, and the base weights stay frozen. (B) Prefix tuning trains both P_k and P_v at every layer, so the factor 2 is needed: 2 × 24 × 50 × 1,024. (C) The two are swapped. Prompt tuning trains vectors at the input only, so it is the smaller one. (D) Prompt tuning trains only t × d embeddings at the input layer. It does not train keys or values. Review: Chapter 03, Concept 8 — Prompt tuning vs prefix tuning."
        },
        {
          "id": "q7",
          "stem": "A team has 40,000 human preference pairs (chosen vs rejected) that were collected once and will not change. They have one GPU node and cannot afford to hold four models in memory or tune an RL loop. Which alignment method fits?",
          "options": {
            "A": "PPO-based RLHF, because only reinforcement learning can use preference data",
            "B": "Plain SFT on the chosen answers only, because preference data needs a reward model",
            "C": "DPO, which learns directly from the fixed pairs using only the policy and a frozen reference model",
            "D": "Train a reward model and then run PPO, because DPO needs online sampling from the live model"
          },
          "correct": "C",
          "explanation": "Correct: (C). DPO needs 2 models (the policy being trained and a frozen reference) and has no reward model and no RL loop. PPO needs 4 models. A fixed offline preference set with a limited budget is the case where DPO is the better choice. PPO is for online exploration or a changing reward. (A) DPO uses preference pairs directly with a simple classification-style loss. No RL loop is needed. (B) SFT on the chosen answers throws away the signal from the rejected ones. DPO uses both and needs no reward model. (D) DPO works offline on a fixed set. The need for a reward model and online sampling is a feature of PPO, not DPO. Review: Chapter 05, RLHF (PPO) vs DPO: which one when?"
        },
        {
          "id": "q8",
          "stem": "A bank must classify 5 million support messages a day into 12 categories. The data cannot leave its own servers, latency must be low, and it has 80,000 labelled examples. Which choice fits best?",
          "options": {
            "A": "Call a large hosted model with a few-shot prompt for every message",
            "B": "Fine-tune a small language model on the labelled data and host it on-premises",
            "C": "Pre-train a new small model from scratch on the 80,000 labelled examples",
            "D": "Use a large model plus RAG over the category descriptions, hosted by a cloud vendor"
          },
          "correct": "B",
          "explanation": "Correct: (B). SLMs fit narrow tasks with high volume, on-premises or on-device needs, and good labelled data. All four conditions hold here. A fine-tuned SLM gives low latency, low cost per message and full data control. (A) At 5 million requests a day this is expensive and slow, and the data would leave the bank's servers. (C) Pre-training from scratch needs far more data and compute. Fine-tuning an existing SLM reuses what it already knows. (D) The task is a narrow, high-volume classification. RAG adds cost, and a cloud host breaks the on-premises rule. Review: Chapter 06, Concept 1 — Small language models (SLMs)."
        },
        {
          "id": "q9",
          "stem": "A customer-service bot sometimes echoes card numbers back in its replies, and some users paste prompt-injection text into the chat. The team can add checks anywhere in the pipeline. Which placement is best?",
          "options": {
            "A": "An input check before the model, plus an output check after it",
            "B": "Only an output check after the model, because injection cannot be seen until the model replies",
            "C": "Only a stronger system prompt, because alignment training already makes the model refuse unsafe requests",
            "D": "Only an input check before the model, because redaction on outputs is not needed once inputs are clean"
          },
          "correct": "A",
          "explanation": "Correct: (A). Guardrails work in layers. The input guardrail comes before the model and blocks injection while saving cost. The output guardrail comes after it and handles PII redaction and content checks. Each placement catches a different failure. (B) Injection text is visible in the input. Blocking it early saves cost and keeps it away from the model. (C) Alignment is not a security boundary. A prompt is an instruction, not a guarantee. (D) Clean inputs do not stop the model from repeating card numbers it has seen. The output needs its own PII check. Review: Chapter 06, Concept 3 — Why alignment isn't enough: enter guardrails."
        }
      ]
    },
    {
      "name": "Section 2 — Vision-language models & multimodal RAG",
      "questions": [
        {
          "id": "q10",
          "stem": "An online marketplace wants to tag 2 million product photos into 40 category labels. It has no labelled images, needs a low cost per image, and only needs the tag, not a description. Which model fits?",
          "options": {
            "A": "LLaVA, asking it to write a sentence about each photo and then reading the category from it",
            "B": "SAM, prompting it with a point to produce a mask for each product",
            "C": "Qwen-VL, asking for a JSON caption that includes the category for every image",
            "D": "CLIP zero-shot, comparing each image embedding with text such as 'a photo of a {label}'"
          },
          "correct": "D",
          "explanation": "Correct: (D). If you need a sentence, use a generative VLM. If you need only a tag, CLIP is the cheaper tool. It embeds the image, embeds the 40 label prompts, and picks the closest one, with no training labels. (A) This is a generative model producing sentences. It is far more costly per image when only a tag is needed. (B) SAM knows where an object is, not what it is. It does not name categories. (C) This also generates text for every photo, which is heavier than needed for a simple tag. Review: Chapter 08, The application map."
        },
        {
          "id": "q11",
          "stem": "A firm indexes scanned annual reports where the meaning sits in charts, tables and page layout. Their OCR pipeline splits tables into jumbled text and loses chart labels. They can accept a larger index and more GPU memory. Which retrieval approach should they try?",
          "options": {
            "A": "Keep the OCR pipeline and add a bigger LLM at answer time to repair the jumbled text",
            "B": "ColPali, which embeds each page image as patch vectors and scores with late interaction (MaxSim)",
            "C": "Embed each full page with CLIP and rank pages by one pooled vector",
            "D": "Switch to BM25 over the OCR text, because keyword search ignores layout problems and needs no embeddings"
          },
          "correct": "B",
          "explanation": "Correct: (B). ColPali treats the page as an image, with about 1,000 patch vectors, and skips OCR. Late interaction (MaxSim) then matches each query token to its best patch. It indexes faster than parse-then-embed but needs a larger index and more GPU memory, which this firm accepts. (A) The damage happens before retrieval. A bigger LLM cannot recover chunks that were never retrieved. (C) CLIP is weak at reading text and fine layout, and a single pooled vector loses detail. (D) BM25 still depends on the same broken OCR text, so jumbled tables stay jumbled. Review: Chapter 09, Concept 5 — Two strategies: parse-then-embed vs retrieval in vision space."
        },
        {
          "id": "q12",
          "stem": "A chat product accepts up to 5 images per message, each 448 × 448 pixels. With patch size 16, a LLaVA-style projector keeps one token per patch, while Qwen-VL's adapter always outputs 256 tokens per image. How many visual tokens reach the LLM per message in each design?",
          "options": {
            "A": "LLaVA-style: 784. Qwen-VL: 256, because tokens are counted per message",
            "B": "LLaVA-style: 3,920. Qwen-VL: 256, because the adapter pools all five images together",
            "C": "LLaVA-style: 5 × 784 = 3,920. Qwen-VL: 5 × 256 = 1,280",
            "D": "LLaVA-style: 1,280. Qwen-VL: 3,920, because the adapter keeps every patch"
          },
          "correct": "C",
          "explanation": "Correct: (C). Patches per image = (448 ÷ 16)² = 28² = 784. A LLaVA-style projector keeps all of them, so 5 images give 3,920 tokens. Qwen-VL's cross-attention adapter uses 256 learnable queries per image, so 5 images give 1,280, whatever the resolution. (A) The product takes 5 images, and each one produces its own tokens. (B) The adapter works on each image separately, giving 256 tokens per image, so 5 × 256 = 1,280. (D) The two are swapped. Qwen-VL compresses to a fixed 256, and the LLaVA-style projector keeps every patch. Review: Chapter 07, Concept 5 — Qwen-VL: a fixed-size summary that remembers where."
        },
        {
          "id": "q13",
          "stem": "A team uses a vision-language model to read invoices with the prompt below. On invoices that have no purchase-order number, the model invents a plausible-looking one. What is the best fix?",
          "code": "Extract from the invoice image and return JSON:\n{\"invoice_no\": \"string\",\n \"po_number\": \"string\",\n \"total\": \"number\"}\nAll fields are required.",
          "options": {
            "A": "Allow null or 'not found' for each field, and send high-stakes values such as the total to a human check",
            "B": "Keep every field required but lower the image resolution to remove distracting detail",
            "C": "Add 'be creative and complete' so the model fills every field with its best guess",
            "D": "Fine-tune CLIP on invoice photos so the vision encoder stops producing a PO number"
          },
          "correct": "A",
          "explanation": "Correct: (A). 'All fields are required' pushes the model to fabricate a value (a presupposition-style hallucination). A schema that allows null or 'not found' gives it an honest way out, and routing high-stakes outputs to human verification catches the remaining errors. (B) Resolution is the main lever for reading documents, and lowering it makes errors more likely, not fewer. (C) This encourages invention. The problem is that the model is forced to fill fields that do not exist. (D) The vision encoder does not produce text fields. The behaviour comes from the prompt and schema that force an answer. Review: Chapter 08, Concept 4 — Document intelligence."
        },
        {
          "id": "q14",
          "stem": "A support site searches product manuals. Users sometimes type exact part codes such as 'XK-2291-B' and sometimes ask natural questions. Pure embedding search keeps missing the exact codes. What should the team do?",
          "options": {
            "A": "Switch to a larger embedding model, since semantic search will then match part codes exactly",
            "B": "Use BM25 only, since keyword search handles natural-language questions as well as embeddings",
            "C": "Increase the number of ANN clusters so approximate search becomes exact for codes",
            "D": "Hybrid search: BM25 plus dense retrieval, then rerank"
          },
          "correct": "D",
          "explanation": "Correct: (D). Keyword search (BM25) is strong on exact tokens like part codes. Semantic search is strong on meaning. Hybrid search combines both, and a reranker sorts the merged list. This is the production default. (A) A bigger embedding model is still semantic. Exact strings are what keyword matching is good at. (B) BM25 misses paraphrases and meaning, so natural questions would suffer. (C) ANN settings change speed and accuracy of the vector search. They do not add keyword matching. Review: Chapter 09, Concept 1 — Text search: keyword, semantic and hybrid."
        },
        {
          "id": "q15",
          "stem": "A RAG system's retriever finds the right passage in the top 5 for 60 of 100 test questions, and in the top 50 for 95 of 100. The generator is given only 5 passages. Which change most directly raises answer quality?",
          "options": {
            "A": "Fine-tune the generator LLM on the test questions, since the retrieved passages are already good enough",
            "B": "Pass all 50 passages to the generator so that nothing relevant is left out",
            "C": "Retrieve the top 50, then use a cross-encoder reranker to pick the best 5 for the generator",
            "D": "Replace the ANN index with exact search, which will raise Recall@5 to 95%"
          },
          "correct": "C",
          "explanation": "Correct: (C). Recall@5 = 60/100 = 60%, while Recall@50 = 95%. The passages are being found but ranked too low. A cross-encoder reranker reads query and passage together, so it can pull the best ones into the final 5. Fine-tuning the embedder or reranker, not the LLM, is the usual fix for weak retrieval. (A) Only 60% of questions get the right passage into the 5. Retrieval is the weak point, not the LLM. (B) Long irrelevant context raises cost and can distract the model. A reranker is the targeted fix. (D) The right passage is already within the top 50. The problem is its order, which exact search does not change. Review: Chapter 09, Concept 10 — Measuring retrieval quality (what the metrics mean)."
        }
      ]
    },
    {
      "name": "Section 3 — Agents, multi-agent systems & LangGraph",
      "questions": [
        {
          "id": "q16",
          "stem": "A finance team automates expense reports. Every report follows the same five steps in the same order: extract fields, validate totals, categorise, auto-approve under a limit, post to the ledger. Only extraction and categorisation need an LLM. What design fits?",
          "options": {
            "A": "An autonomous agent with all tools that decides the order of steps at run time",
            "B": "A fixed workflow with LLM calls at the two language steps",
            "C": "A supervisor with five specialist agents, one for each step",
            "D": "A ReAct agent that reasons freely and may skip validation when it feels confident"
          },
          "correct": "B",
          "explanation": "Correct: (B). The course rule is that if you can draw the flowchart in advance, you do not need an agent. Here the order never changes, so a workflow with LLM calls only where language understanding is needed is cheaper, safer and easier to test. (A) The order is known in advance. An agent adds cost and unpredictability for no benefit. (C) Multi-agent is only for structural needs. Five agents for a fixed flow adds handoffs and failure points. (D) Skipping a validation step on confidence is exactly the risk a fixed flow removes. Review: Chapter 10, Concept 5 — Workflow vs agent: who controls the order?"
        },
        {
          "id": "q17",
          "stem": "A company runs three specialist agents (research, drafting, compliance). Leadership wants one accountable decision point, a full trace of who did what, and the ability to stop a runaway job. They accept some extra latency. Which pattern fits?",
          "options": {
            "A": "A supervisor that assigns tasks, holds the shared state and logs every step, accepting the bottleneck",
            "B": "Peer-to-peer debate among the three agents until they reach agreement, with no central owner or shared log",
            "C": "A router that classifies the request once and hands it off with no further coordination",
            "D": "Fully independent agents that each write directly to the final report"
          },
          "correct": "A",
          "explanation": "Correct: (A). A supervisor is the governance-friendly pattern: it gives one decision point, central tracing and a place to enforce loop guards and escalation. The cost is that it can become a bottleneck or single point of failure, which the company has said it accepts. (B) Debate has no single accountable owner and is harder to trace or stop. (C) A router chooses a path once. It does not hold state or provide ongoing oversight. (D) Independent writers have no coordinator, so there is no single owner and updates can conflict. Review: Chapter 11, Concept 8 — Architectures: supervisor vs peer-to-peer."
        },
        {
          "id": "q18",
          "stem": "Both search nodes below run in the same super-step, and the graph fails with INVALID_CONCURRENT_GRAPH_UPDATE. What is the right fix?",
          "code": "class State(TypedDict):\n    question: str\n    notes: list[str]\n\ndef search_web(state):  return {\"notes\": [\"web fact\"]}\ndef search_docs(state): return {\"notes\": [\"doc fact\"]}\n\ng.add_edge(START, \"search_web\")\ng.add_edge(START, \"search_docs\")\ng.add_edge(\"search_web\", \"write\")\ng.add_edge(\"search_docs\", \"write\")",
          "options": {
            "A": "Replace the two fixed edges with add_conditional_edges so only one search runs",
            "B": "Compile the graph with a checkpointer so each node's write is saved separately",
            "C": "Give notes a reducer, such as Annotated[list[str], add]",
            "D": "Raise recursion_limit so the graph has enough super-steps to apply both writes"
          },
          "correct": "C",
          "explanation": "Correct: (C). With no reducer, a key is overwritten by default. Two nodes writing notes in the same super-step conflict, so LangGraph raises INVALID_CONCURRENT_GRAPH_UPDATE. A reducer such as Annotated[list[str], add] tells it to accumulate the updates instead. (A) This changes the design and removes a search. The conflict is about merging writes. (B) A checkpointer saves state for resuming. It does not decide how two writes to one key are merged. (D) recursion_limit only caps the number of super-steps. Both writes happen in the same one. Review: Chapter 12, Concept 5 — State and reducers: how updates get merged."
        },
        {
          "id": "q19",
          "stem": "A graph is paused with interrupt() and later resumed with Command(resume=True) on the same thread_id. Customers report receiving the email twice. Why, and what is the fix?",
          "code": "def approve_email(state):\n    send_email(state[\"draft\"])\n    ok = interrupt(\"Send this email?\")\n    return {\"approved\": ok}",
          "options": {
            "A": "Resume must use a new thread_id, otherwise the checkpoint replays the first run's side effects",
            "B": "interrupt() only works together with interrupt_before, so the call should be removed",
            "C": "A random UUID idempotency key on send_email will stop the second send",
            "D": "The node re-runs from its first line on resume, so send_email runs again; move the side effect after the interrupt"
          },
          "correct": "D",
          "explanation": "Correct: (D). On resume, LangGraph restarts the paused node from its first line, and interrupt() then returns the human's answer. Anything before interrupt() runs again. So the email was sent before approval and again after it. Put the send after the interrupt, or in a later node, and keep the part before the interrupt free of side effects. (A) Resuming needs the same thread_id to find the saved state. A new one would lose it. (B) interrupt() works inside a node. interrupt_before is another way to pause, not a requirement. (C) A random key changes every run. It must be derived, such as thread_id plus the item, to be recognised. Review: Chapter 12, Concept 9 — Human-in-the-loop: interrupt() and Command(resume=...)."
        },
        {
          "id": "q20",
          "stem": "A company wants agents built in different frameworks to read its ticketing database through one standard interface. Separately, its planning agent must delegate a sub-task to a partner company's agent and track its progress. Which standards match?",
          "options": {
            "A": "A2A for the ticketing tools; MCP for delegating tasks to the partner's agent",
            "B": "MCP for the ticketing tools and data; A2A for delegating the task to the partner's agent",
            "C": "MCP for both, because every agent can be treated as just another tool exposed behind an MCP server",
            "D": "A2A for both, since any request between systems counts as agent-to-agent"
          },
          "correct": "B",
          "explanation": "Correct: (B). MCP lets a client list and call tools from a server that owns their schema, which suits the ticketing data. A2A uses Agent Cards and Tasks so one agent can hand work to another, even across companies. They solve different problems and often appear together. (A) These are the wrong way round. MCP connects agents to tools and data, and A2A connects agents to agents. (C) MCP is for tools and data. Delegating and tracking a task between agents is the job of A2A. (D) A2A is for agent-to-agent tasks. A database is a tool or data source, which is the MCP job. Review: Chapter 11, Concept 12 — Standards for agent communication: MCP vs A2A."
        },
        {
          "id": "q21",
          "stem": "A booking agent's payment call timed out, the framework retried it automatically, and the customer was charged twice. The first call had actually succeeded. What is the best design fix?",
          "options": {
            "A": "Attach a fresh random UUID to every attempt so each request can be traced",
            "B": "Retry more times with a longer back-off until the server confirms a reply",
            "C": "Lower the model's temperature so the agent only makes the payment call once",
            "D": "Send a derived idempotency key (such as thread_id plus booking ID) so a retry is recognised"
          },
          "correct": "D",
          "explanation": "Correct: (D). A timeout does not tell you whether the first call worked. With a derived idempotency key, the payment service sees the same key on the retry and returns the first result instead of charging again. Writes that are not idempotent should not be blindly retried. (A) A fresh UUID makes every retry look new, so the duplicate charge still goes through. (B) More retries on a non-idempotent write can create more duplicates. (C) The retry came from the framework, not from the model's sampling. Review: Chapter 12, Concept 12 — Idempotency: never book twice."
        }
      ]
    },
    {
      "name": "Section 4 — Responsible AI",
      "questions": [
        {
          "id": "q22",
          "stem": "A prompt says 'never refund more than $100'. After a prompt edit to sound friendlier, the agent starts approving $400 refunds. What should the team change?",
          "options": {
            "A": "Enforce the limit in the refund tool's code or permissions, and re-run guardrail tests after every change",
            "B": "Repeat the $100 rule in capital letters at the top and the bottom of the prompt, so that it is hard to miss",
            "C": "Add a footer to every reply saying amounts may be wrong and users should double-check",
            "D": "Rely on alignment training, since a well-aligned model keeps limits across prompt edits"
          },
          "correct": "A",
          "explanation": "Correct: (A). A guardrail is a permission, not an instruction. A limit checked in code cannot be talked around. Because any change to prompts, models or tools can break behaviour that worked before, the guardrail tests must be re-run after every change. (B) A prompt is an instruction, and instructions can still be ignored or broken by later edits. (C) A disclaimer is not a control. The refund still happens. (D) Alignment is not a security boundary, and prompt edits can change the behaviour. Review: Chapter 13, Concept 6 — The five essential controls."
        },
        {
          "id": "q23",
          "stem": "A team set a budget alert at $500 for an agent that calls paid APIs. The alert emailed a shared inbox over a weekend, nobody acted, and the bill grew many times larger. What control was missing?",
          "options": {
            "A": "More alerts at lower thresholds, sent to more people",
            "B": "A hard spending cap that stops the agent, with a named owner",
            "C": "A lower temperature, so the agent makes fewer API calls per task",
            "D": "A longer system prompt telling the agent to be careful with money"
          },
          "correct": "B",
          "explanation": "Correct: (B). The guide's phrase is that an alert is not a cap. A cap is enforced automatically and stops spending, and a named owner is accountable for setting it and for what happens when it triggers. (A) More alerts are still alerts. If nobody acts, the spending continues. (C) Temperature controls randomness in wording, not how many paid calls the agent makes. (D) An instruction in a prompt does not limit spending. It is not enforced in code. Review: Chapter 13, Concept 6 — The five essential controls."
        },
        {
          "id": "q24",
          "stem": "A company buys a CV-screening tool. The vendor says it is compliant, and overall accuracy is high. An internal check finds that one group is shortlisted far less often than the others. Who answers for this, and what should the company do?",
          "options": {
            "A": "The vendor alone answers, because the company only used its product",
            "B": "Nobody, since high overall accuracy shows the tool is fair",
            "C": "The company, as deployer, still answers; it should measure outcomes per group, name an owner and keep evidence",
            "D": "The company is covered once it adds a disclaimer saying the tool may be biased"
          },
          "correct": "C",
          "explanation": "Correct: (C). The provider and deployer both have duties, and the deployer cannot hand its accountability to the vendor. The course's example is the Amazon recruiting tool. The company should check results per group, name an accountable owner and keep an evidence record. (A) Deployers still answer for how a system is used. Buying from a vendor does not move the responsibility away. (B) An overall number can hide a group that is treated worse. The five leader questions include harm per group. (D) A disclaimer is not a control, so the unequal outcomes remain. Review: Chapter 13, Concept 4 — The limits of existing rules: deployers still answer."
        }
      ]
    }
  ]
};
