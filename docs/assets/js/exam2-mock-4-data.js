window.EXAM2_MOCK_4 = {
  "title": "Mock Exam 4 — Official practice set (part 4 of 4)",
  "durationMinutes": 90,
  "marksPerQuestion": 4,
  "sections": [
    {
      "name": "Section 1 — Fine-tuning & alignment",
      "questions": [
        {
          "id": "q1",
          "stem": "When fully fine-tuning a 7B-parameter model with Adam in mixed precision, which component typically consumes the most GPU memory?",
          "options": {
            "A": "The input tokens",
            "B": "The learning-rate scheduler",
            "C": "The optimizer states (first and second moments) plus master weights",
            "D": "The tokenizer vocabulary"
          },
          "correct": "C",
          "explanation": "Correct: (C). Training keeps several numbers for every parameter. The FP16 weight takes 2 bytes and its gradient 2 bytes. Adam adds two FP32 running averages (momentum and variance), which is 8 bytes. Mixed precision usually adds an FP32 master copy of the weight, which is 4 bytes more. So the optimizer states plus master weights are the biggest share, and Adam's states alone are four times the size of the FP16 weights. (A) The input tokens are a small batch of integers. Activations take memory, but the tokens themselves do not. (B) The scheduler is a rule that outputs one learning-rate value per step, so it needs almost no memory. (D) The tokenizer vocabulary is a lookup table of at most a few hundred thousand entries and does not grow with training. Review: Chapter 04, Concept 6, Where the GPU memory actually goes."
        },
        {
          "id": "q2",
          "stem": "Gradient accumulation over 8 steps with a micro-batch size of 4 on a single GPU gives an effective batch size of:",
          "options": {
            "A": "4",
            "B": "8",
            "C": "12",
            "D": "32"
          },
          "correct": "D",
          "explanation": "Correct: (D). Effective batch = micro-batch × accumulation steps × number of GPUs = 4 × 8 × 1 = 32. The optimizer updates the weights only after all 8 micro-batches have been added up, so one update 'sees' 32 examples. (A) 4 is just the micro-batch, which is what one forward and backward pass handles. (B) 8 is the number of accumulation steps, not the number of examples. (C) 12 comes from adding 4 and 8 instead of multiplying them. Remember that accumulation raises the effective batch without raising memory, because memory depends on the micro-batch. Review: Chapter 02, Concept 11, Batch size."
        },
        {
          "id": "q3",
          "stem": "According to the Prompt Tuning paper, how does its performance compare with full fine-tuning as model size grows?",
          "options": {
            "A": "The gap closes, and it becomes competitive with full fine-tuning at around 10B+ parameters",
            "B": "The gap widens as models get larger",
            "C": "It is always better than full fine-tuning",
            "D": "It only works for models under 1B parameters"
          },
          "correct": "A",
          "explanation": "Correct: (A). Prompt tuning trains only a small matrix of virtual-token embeddings while the model stays frozen. The paper's finding is that it lags full fine-tuning on small models but catches up as the model gets bigger, becoming competitive at roughly 10B parameters and above. The exact 10B figure and the paper result are from outside the course slides (beyond the course slides). The course itself stresses that soft prompts are PEFT methods with under 0.1% trainable parameters. (B) This reverses the finding: bigger models narrow the gap, they do not widen it. (C) 'Always better' is too strong. It matches full fine-tuning only for large models. (D) There is no such limit. Small models are where prompt tuning is weakest, not the only place it works. Review: Chapter 03, Concept 8, Prompt tuning vs prefix tuning."
        },
        {
          "id": "q4",
          "stem": "Which statement about QLoRA is correct?",
          "options": {
            "A": "It removes the need for LoRA adapters",
            "B": "It increases inference latency because adapters cannot be merged",
            "C": "It trains the base model weights in 4-bit precision",
            "D": "It keeps the base model frozen in 4-bit NormalFloat (NF4) and trains LoRA adapters in higher precision, using double quantization and paged optimizers"
          },
          "correct": "D",
          "explanation": "Correct: (D). QLoRA is LoRA plus three ingredients. The frozen base weights are stored in 4-bit NF4, the scaling constants are quantized again (double quantization), and a paged optimizer spills optimizer states to CPU RAM when memory spikes. Only the small LoRA matrices A and B are trained, and they stay in higher precision (BF16). (A) QLoRA is built on LoRA, so the adapters are still the part that learns. (B) Nothing about QLoRA forces extra latency. The adapters are separate small matrices that can be merged after training. (C) The 4-bit base weights are frozen and never trained. Each layer is dequantized temporarily for the forward pass, and gradients flow only into A and B. Review: Chapter 04, Putting it together, What QLoRA actually does."
        },
        {
          "id": "q5",
          "stem": "What is the key advantage of Direct Preference Optimization (DPO) over PPO-based RLHF?",
          "options": {
            "A": "It removes the need for a reference model in every formulation",
            "B": "It trains a larger, separate reward model",
            "C": "It needs no preference data at all",
            "D": "It optimizes the policy directly on preference pairs with a simple classification-style loss, removing the explicit reward model and RL sampling loop"
          },
          "correct": "D",
          "explanation": "Correct: (D). DPO uses the same chosen-versus-rejected data as RLHF, but it trains the policy directly with a simple classification-style loss. That drops the separate reward model and the RL loop, so it needs 2 models (policy and frozen reference) instead of PPO's 4, and it is simpler and more stable. (A) The reference model stays. DPO measures the chosen and rejected answers relative to a frozen reference copy, which plays the role of RLHF's KL penalty. (B) DPO trains no separate reward model at all. The reward is implicit. (C) DPO still needs preference data, which is its whole training signal. Review: Chapter 05, Concept 8, DPO."
        },
        {
          "id": "q6",
          "stem": "(Added question) A single attention weight matrix has shape 2048 × 4096. LoRA with rank r = 16 is applied to it. How many trainable parameters does LoRA add for this matrix?",
          "options": {
            "A": "65,536",
            "B": "98,304",
            "C": "8,388,608",
            "D": "134,217,728"
          },
          "correct": "B",
          "explanation": "Correct: (B). LoRA trains B (d × r) and A (r × k), which is r × (d + k) numbers. Here d = 2048, k = 4096 and r = 16, so 16 × (2048 + 4096) = 16 × 6144 = 98,304. That is about 1.2% of the 8,388,608 weights in the full matrix. (A) 65,536 is 16 × 4096, which counts only one of the two thin matrices. (C) 8,388,608 is 2048 × 4096, the size of the full matrix, which is what full fine-tuning would train. (D) 134,217,728 is 16 × 2048 × 4096, which multiplies all three numbers together instead of adding d and k. Check the order of magnitude: LoRA must come out far smaller than the full matrix. Review: Chapter 04, Concept 2, Low rank, and Numbers & formulas, section 11."
        }
      ]
    },
    {
      "name": "Section 2 — Vision-language models & multimodal RAG",
      "questions": [
        {
          "id": "q7",
          "stem": "How does CLIP classify an image into a category it was never explicitly trained on?",
          "options": {
            "A": "It fine-tunes a new classification head on the fly",
            "B": "It embeds class names as text prompts and picks the closest to the image embedding",
            "C": "It runs OCR on the image",
            "D": "It retrieves the nearest training image"
          },
          "correct": "B",
          "explanation": "Correct: (B). This is zero-shot transfer. Each class name is turned into a sentence such as 'a photo of a giraffe' and embedded by the text encoder. The class whose text embedding is closest to the image embedding wins. No classification head is trained, and any text can act as a label. (A) Nothing is trained at classification time, which is the point of zero-shot. (C) OCR reads characters in an image. CLIP does not need to read anything to match an image to a caption. (D) CLIP compares the image with text prompts, not with stored training images. Review: Chapter 07, Concept 3, Zero-shot classification with CLIP."
        },
        {
          "id": "q8",
          "stem": "In LLaVA's two-stage training, what is trained in Stage 1?",
          "options": {
            "A": "The vision encoder only",
            "B": "The LLM only",
            "C": "Everything end to end",
            "D": "Only the projection layer"
          },
          "correct": "D",
          "explanation": "Correct: (D). Stage 1 freezes both the vision encoder and the LLM and trains only the projector, using about 595K caption pairs, so image features are aligned with the LLM's word space. Stage 2 then unfreezes the LLM for instruction tuning on about 158K GPT-4-generated samples, while the vision encoder stays frozen. (A) The vision encoder is frozen in both stages. (B) The LLM is frozen in Stage 1. It is trained only in Stage 2. (C) End-to-end training of everything never happens in LLaVA, because the encoder is always frozen. Review: Chapter 07, Concept 4, LLaVA."
        },
        {
          "id": "q9",
          "stem": "How does Qwen-VL represent bounding boxes for grounding?",
          "options": {
            "A": "As normalized coordinates written as ordinary text inside special box tokens",
            "B": "As image tokens",
            "C": "As pixel masks",
            "D": "Through a separate regression head"
          },
          "correct": "A",
          "explanation": "Correct: (A). Grounding becomes next-token prediction. The LLM writes the coordinates as normal text (numbers normalised to 0 to 1000) between special tags such as <box> and </box>, and <ref> wraps the phrase being referred to. The LLM's existing decoding machinery produces them, with no extra output head. (B) Image tokens carry the picture into the LLM. They are not how boxes come out. (C) Pixel masks are what SAM produces. Qwen-VL outputs text and boxes. (D) There is no regression head. Adding one is exactly what Qwen-VL avoids by emitting coordinates as text. Review: Chapter 07, Concept 5, Qwen-VL."
        },
        {
          "id": "q10",
          "stem": "What problem does dice loss address in SAM's mask training?",
          "options": {
            "A": "Pixel-level class imbalance only",
            "B": "Region-level overlap and object-size imbalance",
            "C": "Prompt ambiguity",
            "D": "Slow convergence"
          },
          "correct": "B",
          "explanation": "Correct: (B). Dice loss measures overlap between whole regions, so a tiny object and a huge object matter equally and small objects are fairly rewarded. Focal loss handles the pixel-level imbalance, and SAM combines them at 20:1 (focal:dice). (A) Pixel-level imbalance, where easy background pixels dominate, is the job of focal loss. (C) Prompt ambiguity is handled differently, by returning 3 candidate masks and scoring each with an IoU-prediction head. (D) Dice loss is not a speed trick. Convergence is not what it is there for. Review: Chapter 08, Concept 1, SAM recap and its loss function."
        },
        {
          "id": "q11",
          "stem": "How do you reduce presupposition hallucination in VQA (e.g., \"What color is the cat?\" when there is no cat)?",
          "options": {
            "A": "Use a larger image",
            "B": "Lower the temperature to zero",
            "C": "Explicitly allow \"not present\" as a valid answer",
            "D": "Remove the question"
          },
          "correct": "C",
          "explanation": "Correct: (C). Without a permitted way to say 'no', the model defaults to answering confidently, because the question itself planted the false assumption. Grounding the prompt and explicitly allowing 'not present' gives it a safe way out. (A) A larger image gives more detail but does not remove the false assumption in the question. (D) Removing the question avoids the problem instead of fixing it, and the model would still fail on similar questions. (B) Temperature zero makes the output deterministic, so the model would give the same wrong confident answer every time. Review: Chapter 08, Concept 3, Visual Question Answering."
        },
        {
          "id": "q12",
          "stem": "What does DocVLM's query-compression module do?",
          "options": {
            "A": "Uses 64 learnable queries to distill OCR and layout signal into a compact, instruction-aware representation",
            "B": "Compresses the image resolution",
            "C": "Removes the need for OCR",
            "D": "Replaces the LLM"
          },
          "correct": "A",
          "explanation": "Correct: (A). DocVLM encodes OCR text together with its layout positions and compresses them with 64 learnable queries, the same idea as Qwen-VL's 256-query adapter. The compressed outputs are projected to the VLM's hidden size and concatenated with the visual tokens, and the base VLM stays frozen. (B) The module compresses OCR and layout information, not the image resolution. (C) It uses OCR as an input, so it does not remove the need for it. (D) The base VLM and its LLM stay frozen and in place. Review: Chapter 08, Concept 4, Document intelligence."
        },
        {
          "id": "q13",
          "stem": "Why does CLIP work well for text-to-image search?",
          "options": {
            "A": "It uses OCR internally",
            "B": "It requires no index",
            "C": "Its shared embedding space lets a text query and an image be compared directly",
            "D": "It generates captions for every image"
          },
          "correct": "C",
          "explanation": "Correct: (C). The text and image encoders are different networks, but contrastive training places their outputs in the same space. A text query can therefore be compared directly with image vectors. Only what is embedded changes, and the ANN indexing (FAISS or HNSW) is the same as for text search. (D) CLIP cannot generate text. It only scores matches. (B) At scale an approximate nearest-neighbour index is still needed. CLIP does not remove it. (A) CLIP does not rely on OCR. The match comes purely from where the query and images sit in the shared space. Review: Chapter 09, Concept 3, Image search with CLIP."
        },
        {
          "id": "q14",
          "stem": "In the EM-style view of training a multimodal RAG system, what is the latent (hidden) variable?",
          "options": {
            "A": "The tokenizer vocabulary",
            "B": "Which documents should have been retrieved",
            "C": "The learning rate",
            "D": "The image resolution"
          },
          "correct": "B",
          "explanation": "Correct: (B). The 'right' documents are never directly observed, so they are the hidden variable. The E-step-like pass retrieves and scores documents with the current model, and the M-step-like pass updates the model so the correct answer becomes more likely. This traces back to the original RAG and REALM papers. (A) The vocabulary is fixed and fully known. (C) The learning rate is a setting that we choose, so nothing about it is hidden. (D) Image resolution is known from the input. Review: Chapter 09, Concept 9, Training a multimodal RAG system: the EM-style view."
        }
      ]
    },
    {
      "name": "Section 3 — Agents & safe agent design",
      "questions": [
        {
          "id": "q15",
          "stem": "Why is planning important in agentic AI?",
          "options": {
            "A": "It allows the model to increase its vocabulary",
            "B": "It guarantees that every action is correct",
            "C": "It decomposes a high-level goal into executable intermediate steps",
            "D": "It eliminates the need for environmental feedback"
          },
          "correct": "C",
          "explanation": "Correct: (C). A goal such as 'book a trip to Paris within policy' is too big to do in one move. Planning breaks it into steps the agent can actually run, such as retrieving the policy, searching flights and checking compliance. Plan-and-Execute drafts the whole plan first, and ReAct plans one step at a time. (A) Planning has nothing to do with the model's vocabulary. (B) No plan guarantees correct actions. Believing a plan is compliant is not the same as checking it. (D) Feedback is still needed. After each action the agent observes the result, and a failed check can force a re-plan. Review: Chapter 10, Concept 7, Three shapes of the loop."
        },
        {
          "id": "q16",
          "stem": "What is the role of tool feedback in an agentic loop?",
          "options": {
            "A": "It replaces the language model",
            "B": "It provides information about the consequences/results of an action",
            "C": "It prevents the agent from taking further actions",
            "D": "It permanently changes the model parameters"
          },
          "correct": "B",
          "explanation": "Correct: (B). The agent loop is Think, Act, Observe, Remember. The tool result (the observation) is appended to the context and the LLM is called again, so the next decision is based on what actually happened. A failed tool is an observation too, not a crash. (A) The tool never replaces the model. The model decides, and your code runs the tool. (C) Feedback is what lets the loop continue or re-plan. Only limits such as a step budget stop it. (D) The model's weights do not change during the loop. It 'remembers' only because the history is re-sent each time. Review: Chapter 10, Concept 6, The agent loop."
        },
        {
          "id": "q17",
          "stem": "In a ReAct-style agent, the sequence below is useful because:",
          "code": "Thought → Action → Observation",
          "options": {
            "A": "The thought is always guaranteed to be correct",
            "B": "The action does not depend on the environment",
            "C": "It eliminates planning",
            "D": "The observation can inform subsequent decisions"
          },
          "correct": "D",
          "explanation": "Correct: (D). ReAct interleaves one thought and one action at a time. Each step is decided from the last result, so the observation feeds the next thought. In the Paris trace, a failed policy check was exactly what made the agent move the date. (A) A thought is only the model's reasoning and can be wrong, which is why checks are run in code. (B) The action runs in the environment, and its result depends on that environment. (C) ReAct does not remove planning. It plans one step at a time instead of drafting everything up front. Review: Chapter 10, Concept 7, Three shapes of the loop."
        },
        {
          "id": "q18",
          "stem": "Why are vector databases commonly used in agent memory systems?",
          "options": {
            "A": "They replace the LLM",
            "B": "They enable similarity-based retrieval of relevant stored information",
            "C": "They guarantee factual correctness",
            "D": "They perform model training"
          },
          "correct": "B",
          "explanation": "Correct: (B). Long-term and episodic memories, such as past episodes or facts about a user, are usually stored as embeddings in a vector database so that the most similar ones can be fetched for the current task. The framing of vector databases as agent memory and the word 'episodic' come from the practice set (beyond the course slides). Note that similar does not always mean useful. (A) A vector database only stores and searches vectors. The LLM still does the reasoning. (C) Stored items can be wrong or outdated, so retrieval gives no factual guarantee. (D) Training changes model weights. A vector store only holds data for lookup. Review: Chapter 10, Concept 9, Three things an agent must remember."
        },
        {
          "id": "q19",
          "stem": "Why doesn't self-reflection necessarily guarantee better agent performance?",
          "options": {
            "A": "Reflection eliminates access to observations",
            "B": "Reflection cannot generate text",
            "C": "A critic can itself produce incorrect evaluations",
            "D": "Reflection prevents iteration"
          },
          "correct": "C",
          "explanation": "Correct: (C). Reflection produces a draft, critiques it and revises it. But the critique is another model output, so it can be wrong or falsely confident. It also doubles or triples token spend. The course's lesson is that a claim of success must be checked by something executable, such as a policy check in code, not just by the model's own opinion. (D) Reflection is itself an iteration loop over the draft. (B) Critiquing and revising are done by generating text. (A) Reflection still has the observations. It adds a critique on top of them. Review: Chapter 10, Concept 7, Three shapes of the loop."
        },
        {
          "id": "q20",
          "stem": "Why is closed-loop behavior generally more robust than open-loop planning in dynamic environments?",
          "options": {
            "A": "It allows the agent to react to changes and unexpected outcomes",
            "B": "It requires no observations",
            "C": "It always produces the globally optimal policy",
            "D": "It eliminates planning"
          },
          "correct": "A",
          "explanation": "Correct: (A). An agent is a closed loop: the plan is regenerated after every observation, like a GPS that recalculates after each turn. An open-loop plan is fixed in advance and cannot adapt when a fare disappears or a check fails. (B) Closed-loop behaviour depends on observations, since they are what it reacts to. (C) Reacting to feedback does not make the policy globally optimal. It only makes it more adaptive. (D) Planning still happens. It is repeated as new information arrives. Review: Chapter 10, Concept 6, The agent loop."
        },
        {
          "id": "q21",
          "stem": "What is the primary challenge in evaluating agentic AI compared with ordinary question-answering systems?",
          "options": {
            "A": "Agents do not use language models",
            "B": "Agents only generate text",
            "C": "Agents cannot be evaluated quantitatively",
            "D": "Agents perform multi-step actions whose intermediate decisions and outcomes matter"
          },
          "correct": "D",
          "explanation": "Correct: (D). A fluent final answer is not enough. You evaluate the whole run: selection accuracy (right agent and tool), argument correctness, task completion, groundedness, conflicts caught, steps, tokens, latency and safety. Failures can come from delegation, tool choice, arguments, execution or the final write-up. (A) Most agents are built around a language model. (B) Agents act through tools, so they do more than generate text. (C) Agents can be measured, for example with task completion, steps and cost. It just takes more metrics than a single answer score. Review: Chapter 11, Concept 11, Logging, evaluation and cost."
        },
        {
          "id": "q22",
          "stem": "Consider the two systems below. The key distinction is:",
          "code": "System A: Question → Search → Generate Answer\nSystem B: Question → Search → Evaluate evidence → Decide whether more search is needed → Search/Answer",
          "options": {
            "A": "System B has adaptive control over its action sequence",
            "B": "System A uses retrieval whereas B does not",
            "C": "System B cannot hallucinate",
            "D": "System A cannot use an LLM"
          },
          "correct": "A",
          "explanation": "Correct: (A). System A is a fixed pipeline: retrieval runs once and then generation. System B is agentic RAG: it checks its evidence and decides whether to search again, rewrite the query or answer, so it has decision-making plus iteration. (B) Both systems search, so retrieval is not the difference. (C) Any system that uses an LLM can still hallucinate. Evaluating evidence reduces the risk but does not remove it. (D) System A generates its answer, normally with an LLM. Review: Chapter 10, Concept 8, Retrieval becomes one tool among many (Agentic RAG)."
        },
        {
          "id": "q23",
          "stem": "An agent has access to a database containing confidential patient information. Which design principle is most appropriate?",
          "options": {
            "A": "Give the agent least-privilege access and enforce authorization outside the LLM",
            "B": "Give the LLM unrestricted database access",
            "C": "Allow the agent to decide its own permissions",
            "D": "Put the database credentials in the prompt"
          },
          "correct": "A",
          "explanation": "Correct: (A). 'The LLM decides what it wants to do; the runtime decides what it is allowed to do.' Authorization must sit outside the LLM, so permissions are checked before any tool returns data, and the agent gets only the minimum access it needs. Write actions also need human approval. (B) Unrestricted access means any prompt injection or mistake can expose everything. (C) An agent must never approve its own high-impact actions, and the same goes for choosing its own permissions. (D) Anything in the prompt can be leaked or manipulated by the model's output. Review: Chapter 11, Concept 10, Safety: permissions, untrusted messages, fail closed."
        },
        {
          "id": "q24",
          "stem": "Which is the most important conceptual limitation of current LLM-based agents?",
          "options": {
            "A": "They cannot perform multi-step tasks",
            "B": "They cannot access external information",
            "C": "Their planning, reasoning, tool use, and self-evaluation can all fail, so autonomy does not imply reliability",
            "D": "They cannot generate language"
          },
          "correct": "C",
          "explanation": "Correct: (C). Agents can plan badly, reason wrongly, call tools with bad arguments and judge their own work wrongly. That is why the course wraps agents in limits and checks: a bounded, observable, recoverable loop, validation in code, least privilege and human approval for risky writes. 'You cannot fix a false claim of success with a better prompt.' (A) Multi-step tasks are exactly what agents are built for. (B) Tools, retrieval and MCP give agents access to outside information. (D) Generating language is what the underlying model does best. Review: Chapter 10, Concept 6, The agent loop."
        }
      ]
    }
  ]
};
