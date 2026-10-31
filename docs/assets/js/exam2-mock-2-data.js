window.EXAM2_MOCK_2 = {
  title: "Mock Exam 2 — Official practice set (part 2 of 4)",
  durationMinutes: 90,
  marksPerQuestion: 4,
  sections: [
    {
      name: "Section 1 — Fine-tuning & alignment",
      questions: [
        {
          id: "q1",
          stem: "In a typical decision framework for adapting LLMs, what is the recommended order of escalation?",
          options: {
            A: "RLHF → full fine-tuning → RAG",
            B: "Full fine-tuning → PEFT → prompt engineering",
            C: "Pre-training → prompt engineering → RAG",
            D: "Prompt engineering → RAG → PEFT → full fine-tuning"
          },
          correct: "D",
          explanation: "Correct: (D). Start with the cheapest, most reversible step and move on only when the current one fails. Prompting changes the instructions, RAG changes what the model sees, and fine-tuning changes the weights. Within fine-tuning, parameter-efficient methods (PEFT) are the lighter option and full fine-tuning is the heaviest. (A) RLHF is an alignment technique, not a first escalation step, and RAG comes before any weight change. (B) This reverses the order and starts with the most expensive step. (C) Pre-training is a huge, from-scratch effort that is almost never the starting point. Note: the course states the order prompting → RAG → fine-tuning; placing PEFT before full fine-tuning follows from PEFT being the cheaper way to change weights. Review: Chapter 01, Concept 2 — The three pathways."
        },
        {
          id: "q2",
          stem: "What is the main purpose of applying a chat template (e.g., system/user/assistant role tags) during data preparation?",
          options: {
            A: "To remove duplicate examples",
            B: "To increase the vocabulary size",
            C: "To compress the dataset",
            D: "To format examples consistently with the structure the model will see at inference"
          },
          correct: "D",
          explanation: "Correct: (D). The model must be trained on text laid out exactly the way it will receive prompts at inference, with the same role tags. A mismatch can silently ruin a fine-tune even when the content is perfect. (A) Removing duplicates is a separate cleaning step in the data curation pipeline. (B) A chat template does not change the vocabulary size. (C) A template adds a few tag tokens; it does not compress anything. Review: Chapter 02, The golden rule of formatting."
        },
        {
          id: "q3",
          stem: "What is a known drawback of adapter layers compared with LoRA at inference time?",
          options: {
            A: "They require full fine-tuning first",
            B: "They cannot be trained with backpropagation",
            C: "They add extra sequential layers, which increases inference latency",
            D: "They only work with encoder-only models"
          },
          correct: "C",
          explanation: "Correct: (C). A sequential adapter sits inside the layer as an extra step that every token must pass through, so inference is slower. LoRA's update can be merged into the original weights, which adds zero latency. (A) Adapters are trained while the base model stays frozen, so no full fine-tuning is needed. (B) Adapters are trained with ordinary backpropagation; only their small weights get gradients. (D) Adapters were shown on many transformer models, not only encoder-only ones. Review: Chapter 04, Concept 5 — Merging: why LoRA adds zero latency at inference."
        },
        {
          id: "q4",
          stem: "For a 4096 × 4096 weight matrix, how many trainable parameters does LoRA add with rank r = 8?",
          options: {
            A: "8,192",
            B: "32,768",
            C: "65,536",
            D: "16,777,216"
          },
          correct: "C",
          explanation: "Correct: (C). LoRA trains B (d × r) and A (r × k), so the count is r × (d + k). Here that is 8 × (4096 + 4096) = 8 × 8,192 = 65,536. (A) 8,192 is just d + k; it forgets to multiply by r. (B) 32,768 counts only one of the two matrices (4096 × 8), forgetting the other. (D) 16,777,216 is 4096 × 4096, the full matrix that full fine-tuning would train. Review: Chapter 04, Concept 3 — Worked example: the size of a LoRA adapter."
        },
        {
          id: "q5",
          stem: "Why is a KL-divergence penalty against the reference (SFT) policy included in the RLHF objective?",
          options: {
            A: "To speed up tokenization",
            B: "To stop the policy drifting too far from the reference model and exploiting flaws in the reward model (reward hacking)",
            C: "To increase the reward model's accuracy",
            D: "To make the model forget its pre-training data"
          },
          correct: "B",
          explanation: "Correct: (B). The reward model has blind spots. Without a leash, the policy can find odd outputs that score high but are useless (reward hacking). The KL penalty subtracts a cost for drifting away from the frozen reference, keeping the policy close to it. (A) The penalty has nothing to do with tokenization. (C) The reward model is trained separately; the KL term does not change its accuracy. (D) The penalty does the opposite: it keeps the model close to what it already knew. Review: Chapter 05, Concept 6 — Reward hacking and the KL penalty."
        },
        {
          id: "q6",
          stem: "(Added question) You fine-tune on 10,000 training examples with an effective batch size of 50 for 3 epochs, using warmup_ratio = 0.1. How many warm-up steps is that?",
          options: {
            A: "20",
            B: "30",
            C: "600",
            D: "60"
          },
          correct: "D",
          explanation: "Correct: (D). Steps per epoch = 10,000 ÷ 50 = 200. Total steps = 200 × 3 epochs = 600. Warm-up steps = 0.1 × 600 = 60. (A) 20 is 0.1 × 200: it uses only one epoch instead of all three. (B) 30 comes from dividing by a batch of 100 (10,000 ÷ 100 = 100 steps, × 3 = 300, × 0.1 = 30), which uses the wrong batch size. (C) 600 is the total number of steps; it forgets to apply the warm-up ratio. Warm-up is counted in optimizer steps, using the effective batch. Review: Chapter 02, Concept 9 — Warm-up and LR schedules; Numbers & formulas, section 4."
        }
      ]
    },
    {
      name: "Section 2 — Vision-language models & multimodal RAG",
      questions: [
        {
          id: "q7",
          stem: "In CLIP, for a batch of N image-text pairs, how many negative pairs does each image have?",
          options: {
            A: "N",
            B: "N − 1",
            C: "N²",
            D: "1"
          },
          correct: "B",
          explanation: "Correct: (B). Each image has one correct caption (the positive). The other N − 1 captions in the batch act as negatives. (A) N would count the correct caption as a negative too. (C) N² is the size of the whole similarity grid, not the negatives per image. (D) 1 is the number of positives per image, not negatives. Review: Chapter 07, Concept 2 — How one training step works (the grid picture)."
        },
        {
          id: "q8",
          stem: "From which position does CLIP's text encoder take the final sentence embedding?",
          options: {
            A: "The average of all tokens",
            B: "The first token",
            C: "The end-of-sequence ([EOS]) token position",
            D: "The longest token"
          },
          correct: "C",
          explanation: "Correct: (C). The activations at the [EOS] position are linearly projected into the shared embedding space. (A) CLIP does not average over all tokens. (B) The first token is not used for the sentence embedding. (D) There is no notion of a longest token here; position, not length, decides. Review: Chapter 07, Concept 2 — Two small details that get asked."
        },
        {
          id: "q9",
          stem: "What problem does LLaVA-NeXT's \"AnyRes\" address?",
          options: {
            A: "Handling high-resolution images by splitting them into tiles encoded separately",
            B: "Speeding up training",
            C: "Removing the need for a projector",
            D: "Reducing model size"
          },
          correct: "A",
          explanation: "Correct: (A). Each tile is encoded by CLIP, and a downsampled global view is added. This improves OCR and fine detail at the cost of more tokens. (B) AnyRes adds more visual tokens, so it does not speed training. (C) The projector is still needed to map vision features into the language model's space. (D) AnyRes does not shrink the model. Review: Chapter 07, Concept 4 — Upgrades worth recognising."
        },
        {
          id: "q10",
          stem: "Why does SAM output three masks for a single point prompt?",
          options: {
            A: "To triple its speed",
            B: "One for each color channel",
            C: "To resolve whole / part / sub-part ambiguity",
            D: "To average them into one"
          },
          correct: "C",
          explanation: "Correct: (C). A click on a shirt could mean the shirt, the person, or a button. An IoU-prediction head scores each mask so the best can be selected. (A) Producing three masks does not make SAM faster. (B) The masks are not tied to colour channels. (D) The masks are alternatives to choose from, not something to average. Review: Chapter 07, Concept 6 — Handling ambiguity: three masks."
        },
        {
          id: "q11",
          stem: "When is CLIP preferable to a generative captioner?",
          options: {
            A: "When you need chain-of-thought reasoning",
            B: "When you need a full descriptive sentence",
            C: "When you only need a tag or category label",
            D: "When you need OCR"
          },
          correct: "C",
          explanation: "Correct: (C). CLIP's zero-shot tagging is cheaper and faster than running a full generative model. Use a generative model only when a sentence is needed. (A) Chain-of-thought reasoning needs a generative model. (B) A full descriptive sentence also needs a generative captioner. (D) OCR needs a model that reads and generates text, which CLIP is weak at. Review: Chapter 08, Concept 2 — Image understanding and captioning (Which model?)."
        },
        {
          id: "q12",
          stem: "What is the single biggest lever for document-task accuracy in the Qwen-VL progression?",
          options: {
            A: "Higher input resolution",
            B: "Removing the vision encoder",
            C: "A larger tokenizer vocabulary",
            D: "Shorter prompts"
          },
          correct: "A",
          explanation: "Correct: (A). Moving from 224px to 448px to dynamic resolution tracked almost directly with DocVQA gains (about 65% to 94%+). Small text on a page needs enough pixels to be readable. (B) Removing the vision encoder would remove the model's ability to see. (C) Vocabulary size is not what limited document accuracy. (D) Prompt length is not the lever shown in the progression. Review: Chapter 08, Concept 4 — Document intelligence (Why documents need special care)."
        },
        {
          id: "q13",
          stem: "What is the main weakness of keyword search such as BM25?",
          options: {
            A: "It is too slow for small corpora",
            B: "It misses synonyms and paraphrases",
            C: "It requires a GPU",
            D: "It cannot handle English"
          },
          correct: "B",
          explanation: "Correct: (B). BM25 matches exact words. A query about \"canceling a subscription\" will not match \"end my plan.\" Dense embeddings match meaning instead. (A) Keyword search is fast, not slow. (C) BM25 runs on a normal CPU. (D) It handles English and many other languages. Review: Chapter 09, Concept 1 — Text search: keyword, semantic and hybrid."
        },
        {
          id: "q14",
          stem: "Why is retrieving in vision space (ColPali) gaining ground over parse-then-embed pipelines?",
          options: {
            A: "OCR and layout analysis are brittle and lossy on charts, tables, and unusual layouts, and embedding the page image avoids that failure point",
            B: "It needs no training data",
            C: "It only works on plain text",
            D: "It produces smaller indexes"
          },
          correct: "A",
          explanation: "Correct: (A). ColPali also indexed pages in about 0.39 s versus about 7.22 s for a standard pipeline. The trade-off is larger indexes, since every patch gets its own vector. (B) ColPali was trained on data; it does not avoid training. (C) It is designed for visually rich pages, not plain text only. (D) Its indexes are larger, not smaller, because every patch is stored as a vector. Review: Chapter 09, Concept 5 — Two strategies: parse-then-embed vs retrieval in vision space."
        }
      ]
    },
    {
      name: "Section 3 — Agents & safe agent design",
      questions: [
        {
          id: "q15",
          stem: "Consider an agent with the loop shown below. What is the primary purpose of repeating this loop?",
          code: "Observe → Think → Act → Observe → …",
          options: {
            A: "To eliminate the need for tools",
            B: "To continuously update the agent's state based on the consequences of its actions",
            C: "To increase the number of model parameters",
            D: "To generate multiple stylistic versions of the same answer"
          },
          correct: "B",
          explanation: "Correct: (B). After each action the agent sees what happened, updates what it knows, and decides the next step. The loop is how results feed back into decisions. (A) The loop is what lets the agent use tools and learn from their results. (C) Looping at run time never changes the model's parameters. (D) The loop is for progress toward a goal, not for style variations. Review: Chapter 10, Concept 6 — The agent loop: Think → Act → Observe → Remember."
        },
        {
          id: "q16",
          stem: "What is the main advantage of giving an LLM access to external tools?",
          options: {
            A: "Tools eliminate hallucination",
            B: "Tools increase the LLM's parameter count",
            C: "Tools make training unnecessary",
            D: "Tools allow the model to perform operations beyond its intrinsic knowledge and generation capability"
          },
          correct: "D",
          explanation: "Correct: (D). Tools let the model look things up, calculate and take actions it cannot do from its trained knowledge alone. (A) Tools reduce some errors but do not eliminate hallucination. (B) Tools are external and do not change the parameter count. (C) The model still needs training; tools add abilities on top. Review: Chapter 10, Concept 3 — Tool calling: the model asks, your code acts."
        },
        {
          id: "q17",
          stem: "Which mechanism is most useful for preventing an agent from endlessly repeating the same actions?",
          options: {
            A: "Maximum step/iteration limits and state/action tracking",
            B: "Removing observations",
            C: "Increasing temperature",
            D: "Increasing vocabulary size"
          },
          correct: "A",
          explanation: "Correct: (A). A loop needs an end condition that does not depend on the model choosing to stop. A step cap plus tracking of past actions catches repeats. (B) Removing observations would blind the agent and make loops more likely. (C) Higher temperature adds randomness but gives no guarantee of stopping. (D) Vocabulary size has no effect on looping. Review: Chapter 10, Concept 6 — Three properties every agent loop needs."
        },
        {
          id: "q18",
          stem: "An agent stores previous successful solutions and retrieves them when solving similar tasks. This is an example of:",
          options: {
            A: "Episodic/experience memory",
            B: "Gradient descent",
            C: "Prompt tokenization",
            D: "Beam search"
          },
          correct: "A",
          explanation: "Correct: (A). Remembering past experiences and reusing them for similar tasks is episodic or experience memory, a form of long-term memory. (B) Gradient descent updates weights during training; nothing is trained here. (C) Tokenization splits text into tokens and is not memory. (D) Beam search is a decoding method for generating text. Review: Chapter 10, Concept 9 — Three things an agent must remember."
        },
        {
          id: "q19",
          stem: "What does reflection generally mean in an agentic system?",
          options: {
            A: "The agent observes only itself rather than the environment",
            B: "The agent evaluates its previous output/action and uses that evaluation to improve subsequent behavior",
            C: "The agent copies its previous answer",
            D: "The agent permanently modifies its weights after every interaction"
          },
          correct: "B",
          explanation: "Correct: (B). Reflection is critique and revise: the agent checks its own earlier output and uses the critique to do better next time. It costs more tokens than other loop shapes. (A) Reflection is about judging outputs, not ignoring the environment. (C) Copying an answer is not an evaluation. (D) Reflection happens in the prompt and context at run time, not through weight changes. Review: Chapter 10, Concept 7 — Three shapes of the loop: ReAct, Plan-and-Execute, Reflection."
        },
        {
          id: "q20",
          stem: "Suppose an agent must navigate a maze. Which information is most important for selecting its next action?",
          options: {
            A: "Only the language model's pretraining corpus",
            B: "Only the original instruction",
            C: "Only the previous action",
            D: "Current state, goal, and relevant observations/history"
          },
          correct: "D",
          explanation: "Correct: (D). To pick a good next move the agent needs to know where it is, where it wants to go, and what it has already seen or tried. In an assistant, context is state. (A) The pretraining corpus does not describe this particular maze. (B) The original instruction alone does not tell the agent where it is now. (C) The previous action alone loses the wider history, so it could walk in circles. Review: Chapter 10, Concept 2 — The assistant: context is state."
        },
        {
          id: "q21",
          stem: "An agent is allowed to send emails autonomously. Which is the most appropriate safety mechanism?",
          options: {
            A: "Disable observations",
            B: "Require confirmation for high-impact actions",
            C: "Allow unlimited tool calls",
            D: "Increase temperature"
          },
          correct: "B",
          explanation: "Correct: (B). Actions that are hard to undo, such as sending an email, should need human approval before they run. The check lives at the execution boundary. (A) Disabling observations would leave the agent blind, not safe. (C) Unlimited tool calls remove a safety bound. (D) Higher temperature makes behaviour less predictable. Review: Chapter 10, Concept 3 — Why this matters: the execution boundary."
        },
        {
          id: "q22",
          stem: "Which metric would be particularly useful for evaluating an agent that performs multi-step tasks?",
          options: {
            A: "Final task success rate",
            B: "Vocabulary size",
            C: "Number of model parameters",
            D: "BLEU alone"
          },
          correct: "A",
          explanation: "Correct: (A). The question for an agent is whether it finished the job. Final task success rate measures that directly. In practice you also look at steps, cost and logged handoffs, so (A) is the best single choice here. (B) Vocabulary size describes the tokenizer, not task performance. (C) Parameter count describes the model's size, not whether the task succeeded. (D) BLEU compares text overlap and says little about whether a multi-step task was completed. Review: Chapter 11, Concept 11 — Logging, evaluation and cost."
        },
        {
          id: "q23",
          stem: "In a multi-agent system, one agent retrieves evidence while another critiques the evidence. What is the main motivation for this architecture?",
          options: {
            A: "The critic does not require observations",
            B: "Retrieval becomes unnecessary",
            C: "Separation of roles can allow specialized processing and cross-checking",
            D: "Multiple agents guarantee correctness"
          },
          correct: "C",
          explanation: "Correct: (C). Splitting roles lets each agent specialise and lets one agent verify another's work, which is one of the good reasons to split. (A) A critic still needs the evidence to review. (B) Retrieval is still needed; the second agent checks it. (D) More agents add cost and failure points and never guarantee correctness. Review: Chapter 11, Concept 2 — When to split the work (and when not to)."
        },
        {
          id: "q24",
          stem: "An agent is asked the request shown below. It identifies duplicates and immediately deletes them. A safer design is:",
          code: "“Delete all duplicate files.”",
          options: {
            A: "Ask the LLM to think longer",
            B: "Use a larger context window",
            C: "Increase retrieval depth",
            D: "Separate identification from irreversible execution and require confirmation before deletion"
          },
          correct: "D",
          explanation: "Correct: (D). Finding duplicates is harmless; deleting them is irreversible. Keep those steps apart and ask a human to confirm before the delete runs. Write actions need approval that sits outside the agent. (A) Thinking longer does not add a safety check. (B) A larger context window does not stop a wrong deletion. (C) Retrieving more does not make the deletion reversible. Review: Chapter 11, Concept 10 — Safety: permissions, untrusted messages, fail closed."
        }
      ]
    }
  ]
};
