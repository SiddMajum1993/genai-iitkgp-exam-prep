window.EXAM2_MOCK_1 = {
  "title": "Mock Exam 1 — Official practice set (part 1 of 4)",
  "durationMinutes": 90,
  "marksPerQuestion": 4,
  "sections": [
    {
      "name": "Section 1 — Fine-tuning & alignment",
      "questions": [
        {
          "id": "q1",
          "stem": "A team needs its LLM to answer questions using a company knowledge base that changes every week. Which approach is usually the best first choice?",
          "options": {
            "A": "Full fine-tuning on the knowledge base every week",
            "B": "RLHF on the knowledge base",
            "C": "Training a new model from scratch",
            "D": "Retrieval-Augmented Generation (RAG)"
          },
          "correct": "D",
          "explanation": "Correct: (D). The knowledge base changes every week, so the model needs fresh facts at question time. RAG retrieves the current documents for each question, so the team only re-indexes the new files. (A) Weekly full fine-tuning is slow and costly, and facts learned this way are frozen at training time and soon stale. (B) RLHF teaches preferred behaviour from human comparisons. It does not load a knowledge base. (C) Training from scratch is the most expensive option and does not solve weekly change at all. Review: Chapter 01, The three pathways: Prompting, RAG, Fine-tuning."
        },
        {
          "id": "q2",
          "stem": "In instruction fine-tuning, why is the loss usually computed only on the response tokens and not on the prompt tokens?",
          "options": {
            "A": "Prompt tokens have no gradients by design",
            "B": "So the model learns to generate answers rather than to reproduce the instructions",
            "C": "Prompt tokens cannot be tokenized",
            "D": "It doubles the effective batch size"
          },
          "correct": "B",
          "explanation": "Correct: (B). In instruction fine-tuning the model should learn to write good answers, not to copy the instruction back. So the loss is counted only on the response tokens. The prompt tokens are still read as context. (A) Prompt tokens do pass through the model and can carry gradients. Masking them from the loss is a training choice, not a built-in property. (C) Prompts are tokenized like any other text. (D) Loss masking does not change the batch size. Review: Chapter 01, Instruction Fine-Tuning (IFT)."
        },
        {
          "id": "q3",
          "stem": "In the Adapter method (Houlsby et al.), what is inserted into each transformer layer?",
          "options": {
            "A": "An extra attention head trained from scratch",
            "B": "A small bottleneck module (down-projection, non-linearity, up-projection) with a residual connection",
            "C": "A learned positional embedding",
            "D": "A copy of the full feed-forward network"
          },
          "correct": "B",
          "explanation": "Correct: (B). A Houlsby adapter is a small bottleneck module: down-project to a small size, apply a non-linearity, up-project back, and add a residual (skip) connection. Only these new layers are trained. (A) Adapters are not extra attention heads, and attention is not what they add. (C) A positional embedding tells the model where a token sits. It is not the adapter. (D) Copying the whole feed-forward network would be large and would defeat the point of parameter-efficient tuning. Review: Chapter 03, What an adapter looks like."
        },
        {
          "id": "q4",
          "stem": "In LoRA, a frozen weight matrix W₀ (d × k) is updated as W₀ + BA. What are the shapes of B and A for rank r?",
          "options": {
            "A": "B is k × d and A is d × k",
            "B": "B is r × r and A is d × k",
            "C": "B is d × r and A is r × k, with r much smaller than d and k",
            "D": "B is d × k and A is k × k"
          },
          "correct": "C",
          "explanation": "Correct: (C). The update BA must have the same shape as W₀, which is d × k. So B is d × r and A is r × k, and (d × r) times (r × k) gives d × k. The rank r is small, so LoRA trains only r × (d + k) numbers. (A) With B as k × d and A as d × k, the product is k × k, not d × k, and nothing is low rank. (B) A B of size r × r cannot give d rows. (D) These shapes are full-size, so there is no saving, and the product does not match W₀. Review: Chapter 04, How LoRA actually works."
        },
        {
          "id": "q5",
          "stem": "What is the correct order of the three stages in the classic RLHF pipeline (InstructGPT)?",
          "options": {
            "A": "Reward modelling → PPO → supervised fine-tuning",
            "B": "Supervised fine-tuning → reward model training on human preference comparisons → RL optimization with PPO",
            "C": "PPO → supervised fine-tuning → reward modelling",
            "D": "Pre-training → DPO → PPO"
          },
          "correct": "B",
          "explanation": "Correct: (B). Classic RLHF runs in three stages. First, supervised fine-tuning (SFT) on good example answers. Second, train a reward model from human comparisons of answers. Third, optimise the model with PPO against the reward model, with a KL penalty to stay near the SFT model. (A) Reward modelling needs an SFT model to produce answers first, and PPO comes after the reward model. (C) PPO needs a reward model to score answers, so it cannot come first. (D) Pre-training comes before all three. DPO is an alternative to the reward model plus PPO, not a step before PPO. Review: Chapter 05, The RLHF pipeline in three stages."
        },
        {
          "id": "q6",
          "stem": "(Added question) A team trains with a micro-batch of 4 examples per GPU, gradient accumulation of 8 steps and 2 GPUs. The training set has 12,800 examples. How many optimizer steps make up one epoch?",
          "options": {
            "A": "200 steps",
            "B": "400 steps",
            "C": "3,200 steps",
            "D": "1,600 steps"
          },
          "correct": "A",
          "explanation": "Correct: (A). Effective batch size = micro-batch × accumulation steps × GPUs = 4 × 8 × 2 = 64. Steps per epoch = 12,800 ÷ 64 = 200. One optimizer step happens only after all 64 examples are seen. (B) 400 divides by 4 × 8 = 32 and forgets the second GPU. (C) 3,200 divides by the micro-batch of 4 only, ignoring both accumulation and GPUs. (D) 1,600 divides by 4 × 2 = 8 and ignores gradient accumulation. Review: Chapter 02, Batch size, and Chapter 02 section on steps and epochs."
        }
      ]
    },
    {
      "name": "Section 2 — Vision-language models & multimodal RAG",
      "questions": [
        {
          "id": "q7",
          "stem": "Which of the three VLM paradigms produces only similarity scores and never free-form text?",
          "options": {
            "A": "Generative (instruction-following)",
            "B": "Contrastive",
            "C": "Autoregressive captioning",
            "D": "Promptable dense prediction"
          },
          "correct": "B",
          "explanation": "Correct: (B). Contrastive models such as CLIP compare image and text embeddings and return similarity scores. They have no text decoder, so they cannot write free-form text. (A) Generative models such as LLaVA are built to produce text. (C) Captioning models decode a caption word by word. (D) Promptable dense prediction, such as SAM, returns pixel masks rather than similarity scores. Review: Chapter 07, The big picture: one template, four variations."
        },
        {
          "id": "q8",
          "stem": "What is the \"modality gap\" in CLIP's embedding space?",
          "options": {
            "A": "Missing training data for some modalities",
            "B": "The gap between zero-shot and fine-tuned accuracy",
            "C": "The text encoder being smaller than the image encoder",
            "D": "Image and text embeddings occupy two distinct, roughly parallel regions rather than fully overlapping"
          },
          "correct": "D",
          "explanation": "Correct: (D). Even after contrastive training, image embeddings and text embeddings sit in two separate, roughly parallel regions (cones) of the shared space. Similarity search still works because it is relative: the matching pair is still the closest. (A) Missing data is a data problem, not an embedding-space property. (B) Zero-shot versus fine-tuned accuracy is a different comparison. (C) The relative size of the two encoders is unrelated to the gap. Review: Chapter 07, CLIP: teaching images and captions to meet in one space."
        },
        {
          "id": "q9",
          "stem": "How was LLaVA's visual instruction-tuning data created without manual annotation?",
          "options": {
            "A": "The model generated its own data from scratch",
            "B": "Crowdsourced human conversations",
            "C": "Scraped from web forums",
            "D": "Text-only GPT-4 was given COCO captions and bounding boxes and generated conversations"
          },
          "correct": "D",
          "explanation": "Correct: (D). No image-instruction dataset existed, so the LLaVA team gave COCO captions and bounding boxes, written out as text, to a text-only GPT-4. GPT-4 never saw the images. It wrote conversations, detailed descriptions and reasoning questions, about 158K samples in total. (A) The model did not invent its own data from scratch. (B) The data was not crowdsourced human conversation. (C) It was not scraped from web forums. Review: Chapter 07, LLaVA: plugging an image into a chatbot."
        },
        {
          "id": "q10",
          "stem": "In Qwen-VL's three-stage training, what is trained in Stage 1?",
          "options": {
            "A": "The vision encoder and adapter, with the LLM frozen",
            "B": "Everything at 448px",
            "C": "The LLM only",
            "D": "Only the text tokenizer"
          },
          "correct": "A",
          "explanation": "Correct: (A). Qwen-VL Stage 1 trains the vision encoder and the adapter on about 1.4B weakly labelled image-text pairs at 224 px, with the LLM frozen. The LLM is unfrozen in Stage 2. (B) Training everything at 448 px describes Stage 2, not Stage 1. (C) Training only the LLM is the opposite of Stage 1, where the LLM is frozen. (D) A tokenizer is not trained in any of the three stages. Review: Chapter 07, Qwen-VL: three training stages."
        },
        {
          "id": "q11",
          "stem": "What did the third (fully automatic) stage of SAM's data engine do?",
          "options": {
            "A": "Annotators corrected SAM's mistakes",
            "B": "Annotators only added missed objects",
            "C": "SAM was prompted with a 32×32 point grid, and confident, de-duplicated masks were kept",
            "D": "Annotators drew masks by hand"
          },
          "correct": "C",
          "explanation": "Correct: (C). In the fully automatic stage, SAM is prompted with a regular 32 × 32 grid of points. Confident, stable masks are kept and near-duplicates are removed. No human is needed in this stage, and it produced most of the 1.1B masks in SA-1B. (A) Annotators correcting SAM belongs to the earlier, human-in-the-loop stages. (B) Annotators adding missed objects is the semi-automatic stage. (D) Annotators drawing masks, with SAM's help, is the first stage. Review: Chapter 07, SAM: the data engine."
        },
        {
          "id": "q12",
          "stem": "What did the 2026 interpretability work on LLaVA-1.5 show about visual question answering?",
          "options": {
            "A": "Many response tokens barely change when the image is blurred, and correct answers can occur with wrong attention",
            "B": "Blurring the image always destroys the answer",
            "C": "Correct answers always mean the model attended to the right region",
            "D": "The vision encoder is never used"
          },
          "correct": "A",
          "explanation": "Correct: (A). Blurring the image barely changed many answer words, which shows much of what looks like visual reasoning is the language prior at work. Also, the attention could be on the wrong region while the answer was right, and the reverse. So accuracy alone can hide how an answer was produced. (B) Blurring did not always destroy the answer. (C) Correct answers do not always mean the right region was attended. (D) The vision encoder is used. The finding is that it is often not decisive. Review: Chapter 08, Visual Question Answering: Are VLMs even looking at the image?"
        },
        {
          "id": "q13",
          "stem": "How does DAVE combine encoders trained against different LLM decoders into one decoder-agnostic encoder?",
          "options": {
            "A": "Learned merge coefficients, with the encoder weights kept frozen",
            "B": "Simple weight averaging",
            "C": "Majority voting",
            "D": "Retraining from scratch"
          },
          "correct": "A",
          "explanation": "Correct: (A). DAVE trains one encoder per LLM decoder, then merges them using learned merge coefficients to get one decoder-agnostic encoder. Learned coefficients beat simple averaging and Fisher merging, and merging lifted document accuracy from 55.6% to 63.4% in the ablations. Nuance: the option's extra phrase about frozen encoder weights is not spelled out in the course digest, so focus on 'learned coefficients' as the key idea. (B) Simple weight averaging was tried and was weaker. (C) Majority voting combines outputs, not weights. (D) Retraining from scratch is not the method. Review: Chapter 08, Document intelligence: case studies (DAVE, DocVLM)."
        },
        {
          "id": "q14",
          "stem": "What does Recall@5 measure?",
          "options": {
            "A": "Image resolution",
            "B": "The number of images in the corpus",
            "C": "Query latency",
            "D": "Whether the correct result appears in the top 5 retrieved"
          },
          "correct": "D",
          "explanation": "Correct: (D). Recall@K asks whether the correct item appears anywhere in the top K results. Recall@5 is therefore always at least as high as Recall@1. It is a standard retrieval metric on Flickr30K and MS COCO. (A) Image resolution is not what a retrieval metric measures. (B) Corpus size is a property of the dataset. (C) Query latency is a speed measure, not a quality measure. Review: Chapter 09, Measuring retrieval quality."
        }
      ]
    },
    {
      "name": "Section 3 — Agents & safe agent design",
      "questions": [
        {
          "id": "q15",
          "stem": "What most fundamentally distinguishes an AI agent from a conventional LLM-based chatbot?",
          "options": {
            "A": "The agent must have access to the internet",
            "B": "The agent always uses a larger language model",
            "C": "The agent can autonomously decide and execute a sequence of actions toward a goal",
            "D": "The agent must use reinforcement learning"
          },
          "correct": "C",
          "explanation": "Correct: (C). The dividing line is task completion: an agent decides and executes a sequence of actions towards a goal, instead of just answering. (A) Internet access is not required. An agent may use only local tools. (B) A bigger model is not the point. The same model can sit inside a chatbot or an agent. (D) Agents do not need reinforcement learning. Review: Chapter 10, The ladder: each rung adds exactly one thing (Common misunderstandings)."
        },
        {
          "id": "q16",
          "stem": "An agent decomposes the goal below into the listed steps. This is an example of:",
          "code": "“Prepare a literature review on multimodal RAG”\ninto:\nSearch papers\nFilter relevant papers\nExtract methods\nCompare approaches\nGenerate synthesis",
          "options": {
            "A": "Task decomposition",
            "B": "Parameter-efficient fine-tuning",
            "C": "Knowledge distillation",
            "D": "Tokenization"
          },
          "correct": "A",
          "explanation": "Correct: (A). Splitting one big goal into smaller bounded sub-tasks is task decomposition. It is the first step of both Plan-and-Execute and the multi-agent coordination cycle. (B) Parameter-efficient fine-tuning is a way to train a model cheaply, not a way to break up a goal. (C) Knowledge distillation trains a small model to copy a large one. (D) Tokenization splits text into tokens. Review: Chapter 11, The coordination cycle and loop (Decompose)."
        },
        {
          "id": "q17",
          "stem": "An agent repeatedly calls a web-search tool without making progress. Which problem is most directly occurring?",
          "options": {
            "A": "Underfitting",
            "B": "Agentic loop failure",
            "C": "Vanishing gradient",
            "D": "Mode collapse"
          },
          "correct": "B",
          "explanation": "Correct: (B). An agent that keeps repeating the same tool call without progress is stuck in a loop. This is why a loop needs a step budget, logging and detection of repeated identical calls. (A) Underfitting is a training problem, not a run-time loop problem. (C) Vanishing gradient happens during training of deep networks. (D) Mode collapse is a generative-training problem. Review: Chapter 10, The agent loop: Think, Act, Observe, Remember (three properties every loop needs)."
        },
        {
          "id": "q18",
          "stem": "What is the major limitation of using an LLM's generated reasoning as the sole basis for an agent's action?",
          "options": {
            "A": "LLMs cannot generate text",
            "B": "Generated reasoning may be incorrect even when it appears plausible",
            "C": "LLMs cannot call tools",
            "D": "Reasoning always requires supervised labels"
          },
          "correct": "B",
          "explanation": "Correct: (B). Generated reasoning can sound plausible and still be wrong, so an agent should verify with an executable check instead of trusting its own claim. (A) LLMs generate text very well. That is not the limitation. (C) LLMs can request tool calls. (D) Reasoning does not require supervised labels. Review: Chapter 10, The agent loop (the Verified guard: check the history, not the model's claim)."
        },
        {
          "id": "q19",
          "stem": "An agent retrieves five previous experiences based solely on embedding similarity. What is a potential problem?",
          "options": {
            "A": "Embeddings cannot represent text",
            "B": "Retrieval prevents tool use",
            "C": "Similarity does not necessarily imply usefulness for the current task",
            "D": "The agent cannot perform reasoning after retrieval"
          },
          "correct": "C",
          "explanation": "Correct: (C). Embedding similarity only says that two things look alike. A past experience that is similar may still be unhelpful or misleading for the current task, so retrieval should consider usefulness too. (A) Embeddings can represent text well. (B) Retrieval does not prevent tool use. (D) The agent can still reason after retrieval. Review: Chapter 10, Three things an agent must remember (Other names you may see; this vocabulary is beyond the course slides)."
        },
        {
          "id": "q20",
          "stem": "What is goal-conditioned behavior in an agent?",
          "options": {
            "A": "The agent always performs the same action",
            "B": "The agent ignores environmental observations",
            "C": "The agent chooses actions based on the desired objective and current state",
            "D": "The agent generates random actions"
          },
          "correct": "C",
          "explanation": "Correct: (C). A goal-conditioned agent picks each action using both the desired objective and the current state, and it re-decides as observations come in. This term is general agent vocabulary used by the practice set (beyond the course slides). (A) Always doing the same action is a fixed script, not goal-directed behaviour. (B) Ignoring observations would make it an open-loop script. (D) Random actions have no relation to a goal. Review: Chapter 10, The agent loop: Think, Act, Observe, Remember."
        },
        {
          "id": "q21",
          "stem": "What is the main purpose of guardrails in agentic systems?",
          "options": {
            "A": "Increase model vocabulary",
            "B": "Increase context length",
            "C": "Replace planning",
            "D": "Constrain potentially unsafe, invalid, or unauthorized actions"
          },
          "correct": "D",
          "explanation": "Correct: (D). Guardrails are limits that the system enforces on what an agent may say or do. They block unsafe, invalid or unauthorized actions. (A) Guardrails do not change vocabulary size. (B) They do not change context length. (C) They do not replace planning. They sit around planning and acting as checks. Review: Chapter 13, The five essential controls (Guardrails), and Chapter 06, Why alignment isn't enough: enter guardrails."
        },
        {
          "id": "q22",
          "stem": "An agent obtains the correct final answer but takes 40 unnecessary tool calls. What does this illustrate?",
          "options": {
            "A": "Perfect agentic behavior",
            "B": "Retrieval failure",
            "C": "The difference between outcome correctness and process efficiency",
            "D": "Model underfitting"
          },
          "correct": "C",
          "explanation": "Correct: (C). A correct final answer shows outcome correctness. Forty unnecessary tool calls show poor process efficiency, which costs time and tokens. Both are measured separately. (A) It is not perfect behaviour because the process wasted cost. (B) Retrieval failure means the right information was never found. (D) Underfitting is a training issue. Review: Chapter 11, Logging, evaluation and cost (rounds and cost to finish)."
        },
        {
          "id": "q23",
          "stem": "An agent plans the sequence shown below, but the optimal strategy is to perform one carefully targeted search followed by synthesis. What does this primarily demonstrate?",
          "code": "Search → summarize → search → summarize → search → summarize",
          "options": {
            "A": "Planning can introduce unnecessary action costs",
            "B": "Retrieval always causes hallucination",
            "C": "Reflection eliminates inefficiency",
            "D": "More actions always improve reliability"
          },
          "correct": "A",
          "explanation": "Correct: (A). A plan with repeated search and summarize cycles spends extra actions and tokens where one targeted search would do. So planning can add unnecessary action cost. (B) Retrieval does not always cause hallucination. (C) Reflection does not remove inefficiency. It adds more passes and costs more tokens. (D) More actions do not always improve reliability. Each action adds cost and chances to fail. Review: Chapter 10, Three shapes of the loop: ReAct, Plan-and-Execute, Reflection."
        },
        {
          "id": "q24",
          "stem": "Why should authorization generally be enforced outside the LLM?",
          "options": {
            "A": "LLMs cannot generate structured outputs",
            "B": "LLMs cannot process permissions",
            "C": "Authorization is unnecessary for agents",
            "D": "An LLM's generated decision is not a reliable security boundary"
          },
          "correct": "D",
          "explanation": "Correct: (D). An LLM's generated decision can be changed by a clever prompt, so it is not a reliable security boundary. The runtime must check permissions in code, outside the model. As the course puts it, the LLM decides what it wants to do and the runtime decides what it is allowed to do. (A) LLMs can produce structured outputs. (B) LLMs can read permission text. The issue is trust, not ability. (C) Authorization is essential for agents, especially for write actions. Review: Chapter 12, Where students got confused (payroll example), and Chapter 11, Safety: permissions, untrusted messages, fail closed."
        }
      ]
    }
  ]
};
