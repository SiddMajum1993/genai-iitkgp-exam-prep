window.EXAM2_MOCK_3 = {
  "title": "Mock Exam 3 — Official practice set (part 3 of 4)",
  "durationMinutes": 90,
  "marksPerQuestion": 4,
  "sections": [
    {
      "name": "Section 1 — Fine-tuning & alignment",
      "questions": [
        {
          "id": "q1",
          "stem": "Which of the following is a major drawback of full fine-tuning?",
          "options": {
            "A": "It needs high GPU memory and a full model copy per task, and risks catastrophic Forgetting",
            "B": "It updates only a small fraction of parameters",
            "C": "It cannot change the model’s output style",
            "D": "It cannot be used with labelled data"
          },
          "correct": "A",
          "explanation": "Correct: (A). Full fine-tuning updates every weight, so the GPU must hold weights, gradients and Adam states, you keep one full model copy per task, and narrow training can overwrite general skills (catastrophic forgetting). (B) This describes PEFT, which trains only a small set of parameters. Full fine-tuning updates all of them. (C) Full fine-tuning adapts style and format very strongly. It is one of its strengths. (D) Full fine-tuning is normally done with labelled data, as in SFT. Review: Chapter 01, Why full fine-tuning is costly and three ways to make it cheaper."
        },
        {
          "id": "q2",
          "stem": "Which technique trades extra computation for lower activation memory by recomputing activations during the backward pass?",
          "options": {
            "A": "Gradient accumulation",
            "B": "Learning-rate warmup",
            "C": "Label smoothing",
            "D": "Gradient checkpointing"
          },
          "correct": "D",
          "explanation": "Correct: (D). Gradient checkpointing keeps only some activations during the forward pass and recomputes the missing ones during the backward pass. This trades extra compute for lower activation memory. (A) Gradient accumulation adds up gradients from several small micro-batches before one update. It gives a larger effective batch but does not recompute anything. (B) Warm-up ramps the learning rate up at the start of training and has no effect on activation memory. (C) Label smoothing softens the target labels in the loss. It does not change how activations are stored (beyond the course slides). Review: Chapter 04, Gradient checkpointing: save some, recompute the rest."
        },
        {
          "id": "q3",
          "stem": "How does Prefix Tuning differ from Prompt Tuning (Lester et al.)?",
          "options": {
            "A": "Prompt Tuning uses discrete text tokens; Prefix Tuning uses LoRA matrices",
            "B": "There is no difference; they are the same method",
            "C": "Prefix Tuning adds trainable vectors to the keys and values at every layer; Prompt Tuning adds trainable embeddings only at the input layer",
            "D": "Prefix Tuning updates all model weights; Prompt Tuning updates none"
          },
          "correct": "C",
          "explanation": "Correct: (C). Both methods freeze the model and learn virtual tokens. Prompt tuning adds trainable embeddings at the input layer only. Prefix tuning adds trainable key and value prefixes (P_k and P_v) at every attention layer, which gives it more direct influence. (A) This is wrong on both sides. Prompt tuning uses learned soft vectors rather than discrete text, and prefix tuning does not use LoRA matrices. (B) They are different methods. The difference is where the virtual tokens go. Note the stem's wording: Lester et al. is the prompt-tuning paper, and prefix tuning comes from Li & Liang (2021). (D) Both freeze all model weights, and only the virtual-token vectors are trained. Review: Chapter 03, Prompt tuning vs prefix tuning."
        },
        {
          "id": "q4",
          "stem": "How is LoRA typically initialized so that training starts exactly from the pre-trained model’s behaviour?",
          "options": {
            "A": "A and B are copied from W₀",
            "B": "Both A and B are set to zero",
            "C": "A is initialized randomly (Gaussian) and B is initialized to zero, so BA = 0 at the start",
            "D": "Both A and B are initialized randomly"
          },
          "correct": "C",
          "explanation": "Correct: (C). In LoRA the output is W₀x + (α/r)·BAx. A starts as small random numbers and B starts at exactly zero, so BA = 0 and the model behaves exactly like the pre-trained one at step 0. (A) W₀ is the frozen original weight. A and B are new thin matrices and are not copied from it. (B) Setting both to zero is not the course recipe. The course sets only B to zero and starts A with random values. With both at zero the side path would also start with nothing to learn from (beyond the course slides). (D) If both were random, BA would be a non-zero random correction that damages the model before training begins. Review: Chapter 04, How LoRA actually works (why B starts at zero)."
        },
        {
          "id": "q5",
          "stem": "The reward model in RLHF is usually trained on pairwise comparisons using which loss formulation?",
          "options": {
            "A": "Cross-entropy over the next token",
            "B": "The Bradley–Terry model: maximize the log-sigmoid of the reward difference between the preferred and rejected response",
            "C": "Contrastive loss between images and text",
            "D": "Mean squared error against a human-written score"
          },
          "correct": "B",
          "explanation": "Correct: (B). Humans find it easier to say which of two answers is better than to give a score. The reward model is trained with the Bradley–Terry loss, L = −log σ(r_chosen − r_rejected), which is small when the chosen answer scores well above the rejected one. (A) Next-token cross-entropy is the language-modelling loss used in pre-training and SFT. It is not used to rank answers. (C) A contrastive loss between images and text is CLIP's loss. (D) The course notes that people struggle to agree on an absolute score such as 7.3 out of 10, which is why comparisons are used instead of regression on human scores. Review: Chapter 05, The reward model: learning to score from comparisons."
        },
        {
          "id": "q6",
          "stem": "(Added question) A team applies LoRA with rank r = 8 to one weight matrix of size 2048 × 4096 (d = 2048, k = 4096). How many trainable parameters does LoRA add for this one matrix, and what share of the full matrix is that?",
          "options": {
            "A": "49,152, which is about 0.59% of the 8,388,608 weights in the full matrix",
            "B": "98,304, because both A and B must be counted twice for the update",
            "C": "16,384, counting only the B matrix (d × r)",
            "D": "32,768, counting only the A matrix (r × k)"
          },
          "correct": "A",
          "explanation": "Correct: (A). For one d × k matrix, full fine-tuning trains d × k weights: 2048 × 4096 = 8,388,608. LoRA trains B (d × r) and A (r × k), which is r × (d + k) = 8 × (2048 + 4096) = 8 × 6144 = 49,152. As a share of the full matrix that is 49,152 ÷ 8,388,608 ≈ 0.59%. (B) 98,304 is double the right number. Nothing is counted twice. (C) 16,384 is r × d = 8 × 2048, which counts only B and forgets A. (D) 32,768 is r × k = 8 × 4096, which counts only A and forgets B. Review: Chapter 04, Low rank: building a big matrix from two thin ones."
        }
      ]
    },
    {
      "name": "Section 2 — Vision-language models & multimodal RAG",
      "questions": [
        {
          "id": "q7",
          "stem": "What does CLIP's learnable temperature parameter (τ) control?",
          "options": {
            "A": "The length of the text sequence",
            "B": "The sharpness of the softmax over similarity scores",
            "C": "The learning rate",
            "D": "The size of the image patches"
          },
          "correct": "B",
          "explanation": "Correct: (B). CLIP divides the image–text similarities by the learnable temperature τ before the softmax. A lower τ makes the softmax sharper and more confident, and a higher τ makes it softer. For example, cosines of 0.30 and 0.25 are nearly a coin flip at τ = 1, but at τ = 0.01 the first caption gets about 99%. (A) Text length is set by the tokenizer and context size, not by τ. (C) τ is not the learning rate. It is a parameter learned inside the loss. (D) Patch size is a fixed property of the vision encoder. Review: Chapter 07, Concept 2 — CLIP: teaching images and captions to meet in one space."
        },
        {
          "id": "q8",
          "stem": "Why does LLaVA need a projection layer between the vision encoder and the LLM?",
          "options": {
            "A": "To convert images to grayscale",
            "B": "The image is too large to fit in memory",
            "C": "CLIP image features and LLM word embeddings live in different geometries, even if they have the same dimension",
            "D": "The LLM cannot process any vectors"
          },
          "correct": "C",
          "explanation": "Correct: (C). CLIP image features and the LLM's word embeddings live in unrelated geometries, even when the vectors have the same size. The projector learns where each visual feature should land so the LLM sees it as meaningful content. Without it, an image vector would be an arbitrary point in the LLM's space. (A) Grayscale conversion has nothing to do with the projector. (B) The projector is not there for memory reasons. Its job is to translate between two spaces. (D) The LLM processes vectors perfectly well. The issue is that these vectors mean nothing to it until they are mapped. Review: Chapter 07, Concept 4 — LLaVA: plugging an image into a chatbot."
        },
        {
          "id": "q9",
          "stem": "What does Qwen-VL's position-aware vision-language adapter output, regardless of input resolution?",
          "options": {
            "A": "A single pooled vector",
            "B": "A variable number of tokens proportional to resolution",
            "C": "One token per image patch",
            "D": "A fixed 256 visual tokens"
          },
          "correct": "D",
          "explanation": "Correct: (D). Qwen-VL's position-aware adapter is one cross-attention layer with 256 learnable queries. Each query attends over all patch features, and 2D positional encodings keep the spatial layout, so exactly 256 visual tokens come out whatever the resolution. (A) A single pooled vector would throw away the spatial layout. (B) A count that grows with resolution is how LLaVA behaves, because its projector keeps every patch. (C) One token per patch is also the LLaVA-style projector. Qwen-VL compresses the patches instead. Review: Chapter 07, Concept 5 — Qwen-VL: a fixed-size summary that remembers where."
        },
        {
          "id": "q10",
          "stem": "Why does SAM run its image encoder only once per image?",
          "options": {
            "A": "It is the expensive step, and its embedding is cached and reused for every prompt",
            "B": "It reduces training loss",
            "C": "The encoder cannot be run twice",
            "D": "To save disk space"
          },
          "correct": "A",
          "explanation": "Correct: (A). SAM's heavy ViT-H image encoder is the expensive step, at about 0.15 s. Its embedding is cached and reused, and each new prompt only runs the light mask decoder (about 50 ms). That makes clicking feel instant. (B) Running the encoder once has nothing to do with training loss. (C) The encoder can be run again. It is simply wasteful to do so. (D) The reason is compute time and interactivity, not disk space. Review: Chapter 07, Concept 6 — SAM: cut out whatever you point at."
        },
        {
          "id": "q11",
          "stem": "What is the best practice for generating captions consistently at scale?",
          "options": {
            "A": "Ask for the most creative caption possible",
            "B": "Fix one prompt template with explicit scope and length constraints",
            "C": "Remove all constraints",
            "D": "Vary the prompt for each image"
          },
          "correct": "B",
          "explanation": "Correct: (B). For captioning at scale, fix one prompt template with an explicit scope and length, for example 'one sentence, only what is visually present'. Predictability matters more than per-caption creativity, and the scope limits hallucination and length drift. (A) Asking for the most creative caption makes the output vary and invites invented details. (C) Removing constraints leads to over-generic captions, hallucination or length drift. (D) Changing the prompt for each image breaks consistency across the catalogue. Review: Chapter 08, Image understanding and captioning (how to prompt)."
        },
        {
          "id": "q12",
          "stem": "Why does DAVE reconstruct raw pixels instead of per-patch normalized targets in its MAE stage?",
          "options": {
            "A": "To save memory",
            "B": "Because normalization is not differentiable",
            "C": "Document and web images have very low inter-patch variance, which destabilizes the normalized target",
            "D": "To speed up inference"
          },
          "correct": "C",
          "explanation": "Correct: (C). The official explanation is that document and web pages are mostly uniform, so their patches have very low variance between them. Normalising such small-variance patches destabilises training, and reconstructing raw pixels fixed this. DAVE is an optional case study in the guide, and this detail comes from the official practice-set explanation rather than the main lecture text. (A) Memory saving is not the stated reason. (B) Per-patch normalisation is differentiable. The problem is stability, not differentiability. (D) The choice concerns a training target, so it does not speed up inference. Review: Chapter 08, Document intelligence (DAVE case study)."
        },
        {
          "id": "q13",
          "stem": "Why do production systems use approximate nearest neighbor (ANN) indexes such as FAISS or HNSW?",
          "options": {
            "A": "They are always more accurate than exact search",
            "B": "They replace the embedding model",
            "C": "They need no vectors",
            "D": "Exact search does not scale to millions of documents, and ANN trades a little accuracy for large speed gains"
          },
          "correct": "D",
          "explanation": "Correct: (D). Exact nearest-neighbour search does not scale to millions of documents. ANN indexes such as FAISS or HNSW trade a little accuracy for a large speed gain, because a search only touches a small subset of candidates. (A) ANN is not more accurate than exact search. It gives up a little accuracy for speed. (B) The embedding model is still needed to create the vectors. (C) ANN indexes search over vectors, so they need them. Review: Chapter 09, How dense retrieval works: bi-encoders and ANN indexes."
        },
        {
          "id": "q14",
          "stem": "How is the late-interaction score computed in ColBERT and ColPali?",
          "options": {
            "A": "The cosine of the first tokens only",
            "B": "The average of all token embeddings",
            "C": "For each query token take its maximum similarity over document tokens or patches, then sum these maxima",
            "D": "A single dot product between two pooled vectors"
          },
          "correct": "C",
          "explanation": "Correct: (C). This is MaxSim. Each query token takes its single best similarity over the document's tokens or patches, and these maxima are added up. The result is richer than comparing two pooled vectors. (A) The score uses every query token, not only the first. (B) Averaging all token embeddings squeezes everything into one blurry vector, which is exactly what late interaction avoids. (D) A single dot product between two pooled vectors is ordinary single-vector retrieval, not late interaction. Review: Chapter 09, Late interaction and MaxSim."
        }
      ]
    },
    {
      "name": "Section 3 — Agents & safe agent design",
      "questions": [
        {
          "id": "q15",
          "stem": "An agent receives the goal: “Find the cheapest suitable flight to Delhi and book it.” Which behavior is most characteristic of an agent?",
          "options": {
            "A": "Searching flights, comparing options, asking for confirmation if necessary, and executing the booking",
            "B": "Generating a fixed flight itinerary from its training data",
            "C": "Generating a paragraph describing possible flights",
            "D": "Predicting the next token describing a flight"
          },
          "correct": "A",
          "explanation": "Correct: (A). An agent is given a goal and decides its own steps: it searches, compares options, asks for confirmation when needed and then executes the booking. The test is task completion, not just text. (B) Producing a fixed itinerary from training data involves no tool use or decisions. (C) A paragraph describing possible flights is plain text generation, with no searching or booking. (D) Predicting the next token is what every LLM does. On its own it is not agent behaviour. Review: Chapter 10, The ladder: each rung adds exactly one thing."
        },
        {
          "id": "q16",
          "stem": "An agent decides to call the function shown below. The LLM's decision to invoke this function is best viewed as:",
          "code": "search(query=\"recent multimodal RAG papers\")",
          "options": {
            "A": "Embedding generation",
            "B": "Backpropagation",
            "C": "Reward modeling",
            "D": "Tool selection/action generation"
          },
          "correct": "D",
          "explanation": "Correct: (D). The model writes out a structured request to call search with some arguments. This is tool selection, or action generation. Your code then executes the call and returns the result. (A) Embedding generation is a retrieval step that turns text into vectors. A search call is not that. (B) Backpropagation is used during training and is not part of an agent's tool call. (C) Reward modelling is part of RLHF alignment and is unrelated to choosing a tool at run time. Review: Chapter 10, Tool calling: the model asks, your code acts."
        },
        {
          "id": "q17",
          "stem": "What is ReAct primarily known for?",
          "options": {
            "A": "Combining retrieval with convolutional networks",
            "B": "Interleaving reasoning and actions/tool calls",
            "C": "Replacing transformers with recurrent networks",
            "D": "Training agents exclusively with reinforcement learning"
          },
          "correct": "B",
          "explanation": "Correct: (B). ReAct (Reasoning + Acting) interleaves a thought with an action or tool call, observes the result, and then decides the next step. (A) ReAct is about the reasoning–action loop, not about convolutional networks. (C) It does not replace transformers. It is a pattern for using an LLM in a loop. (D) ReAct does not require reinforcement learning. It works by prompting the model in a Think → Act → Observe loop. Review: Chapter 10, Three shapes of the loop: ReAct, Plan-and-Execute, Reflection."
        },
        {
          "id": "q18",
          "stem": "What is the key distinction between working memory and long-term memory in an agent?",
          "options": {
            "A": "Working memory is stored only in model weights",
            "B": "Working memory always requires a vector database",
            "C": "Long-term memory cannot contain text",
            "D": "Working memory concerns the current task/context, whereas long-term memory persists across interactions or tasks"
          },
          "correct": "D",
          "explanation": "Correct: (D). Working memory holds what matters for the current task or context, close to the course's task state. Long-term memory persists across interactions and tasks. (A) Working memory is not stored in the model weights. (B) Working memory does not require a vector database. Long-term memory is the one often stored in a vector database. (C) Long-term memory can hold text such as facts, preferences and past episodes. Working memory and episodic memory are practice-set terms that go beyond the course slides. Review: Chapter 10, Three things an agent must remember."
        },
        {
          "id": "q19",
          "stem": "Consider the flow below. This architecture primarily introduces:",
          "code": "Generate solution → Critic evaluates solution → Agent revises solution",
          "options": {
            "A": "Token pruning",
            "B": "Retrieval-only generation",
            "C": "Self-reflection/critique",
            "D": "Knowledge distillation"
          },
          "correct": "C",
          "explanation": "Correct: (C). The flow generates a solution, has a critic evaluate it and then revises it. This is Reflection, a self-critique loop on a draft. (A) Token pruning removes tokens from a sequence and is unrelated. (B) Retrieval-only generation answers from retrieved text without any critique step. (D) Knowledge distillation trains a smaller model from a larger one. This flow does not train anything. Review: Chapter 10, Three shapes of the loop (Reflection)."
        },
        {
          "id": "q20",
          "stem": "Which scenario most clearly requires closed-loop control?",
          "options": {
            "A": "Robot navigation where each action changes the subsequent observations",
            "B": "Generating a poem",
            "C": "Translating a paragraph",
            "D": "Summarizing a document"
          },
          "correct": "A",
          "explanation": "Correct: (A). Closed-loop control means each action changes what is observed next, so the system must keep observing and adjusting. A robot navigating is the classic case. (B) A poem is a single generation with no feedback from the world. (C) Translating a paragraph is a one-shot transformation. (D) Summarising a document is also a one-shot task. Review: Chapter 10, The agent loop: Think → Act → Observe → Remember."
        },
        {
          "id": "q21",
          "stem": "Which action has the highest need for human confirmation?",
          "options": {
            "A": "Searching a public webpage",
            "B": "Sending a legally binding contract to a third party",
            "C": "Summarizing a document",
            "D": "Calculating an average"
          },
          "correct": "B",
          "explanation": "Correct: (B). Sending a legally binding contract is an irreversible write action with real consequences. Costly or irreversible actions need human confirmation placed at the execution boundary. (A) Searching a public webpage is a low-risk read. (C) Summarising a document changes nothing in the world. (D) Calculating an average is harmless and reversible. Review: Chapter 10, Tool calling (the execution boundary)."
        },
        {
          "id": "q22",
          "stem": "An agent has the following policy. What makes this agentic rather than simply a fixed pipeline?",
          "code": "If search results are insufficient → search again with a refined query.",
          "options": {
            "A": "The system uses embeddings",
            "B": "The model dynamically determines whether another action is necessary based on the observation",
            "C": "The search operation exists",
            "D": "The system uses an LLM"
          },
          "correct": "B",
          "explanation": "Correct: (B). The rule is that if search results are insufficient, the agent searches again with a refined query. The model decides, from the observation, whether another action is needed. In a fixed pipeline the developer fixes the order in advance. (A) Using embeddings does not decide who controls the next step. (C) A fixed pipeline can contain a search step too. (D) A workflow can also contain an LLM. What makes it agentic is the run-time decision. Review: Chapter 10, Workflow vs agent: who controls the order?"
        },
        {
          "id": "q23",
          "stem": "An agent uses another LLM as a critic. Both models make the same systematic mistake. What limitation does this reveal?",
          "options": {
            "A": "Independent verification is not guaranteed merely by using multiple agents",
            "B": "Multi-agent systems cannot communicate",
            "C": "Critics cannot use language",
            "D": "Tool use causes the error"
          },
          "correct": "A",
          "explanation": "Correct: (A). If the critic and the worker are the same kind of model, they can share the same blind spot, so the critic agrees with the mistake. Independent verification is not guaranteed merely by having several agents or by using an LLM as critic. The course's reason to split work includes verification, but it needs genuine independence, such as a separate check in code or a different source (the shared-blind-spot reasoning is beyond the course slides). (B) Multi-agent systems can communicate. The issue here is shared error. (C) Critics can use language. That is how they work. (D) The tool is not the cause of the error. Review: Chapter 11, When to split the work (verification)."
        },
        {
          "id": "q24",
          "stem": "Which statement best captures the difference between an agent and an LLM workflow?",
          "options": {
            "A": "An LLM workflow cannot call tools",
            "B": "Agents cannot use predefined workflows",
            "C": "An agent necessarily uses reinforcement learning",
            "D": "An agent can dynamically select actions based on observations and goals, rather than merely executing a predetermined sequence"
          },
          "correct": "D",
          "explanation": "Correct: (D). An agent chooses its next action at run time from observations and goals. A workflow executes a sequence the developer fixed in advance. (A) A workflow can call tools. The tool-using assistant is exactly that. (B) Agents can include predefined workflows. In practice you ship both. (C) An agent does not need reinforcement learning. It can be an LLM with tools in a loop. Review: Chapter 10, Workflow vs agent: who controls the order?"
        }
      ]
    }
  ]
};
