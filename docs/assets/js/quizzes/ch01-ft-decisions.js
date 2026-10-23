// Practice questions for Chapter 01 · When to fine-tune.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch01"] = {
  "title": "When to fine-tune",
  "questions": [
    {
      "id": "q1",
      "type": "Scenario",
      "stem": "An HR team wants a chatbot that answers employee questions about leave and travel policies. The policies are revised every few weeks, and answers should point to the policy document they came from. What is the most suitable approach?",
      "options": {
        "A": "Curriculum fine-tuning, starting with simple policies and moving to complex ones.",
        "B": "Full fine-tuning of an open-weight model on the current policy documents.",
        "C": "RAG combined with good prompt engineering.",
        "D": "Instruction fine-tuning on a set of HR question–answer pairs."
      },
      "correct": "C",
      "why": {
        "A": "Curriculum learning is about the order of training examples. It doesn't solve the changing-knowledge problem.",
        "B": "Fine-tuned knowledge is frozen at training time. You'd have to retrain every time a policy changes, and the answers can't cite sources.",
        "C": "Correct. RAG fetches the latest policy text at question time, is cheap to update, and lets you trace answers to sources. The course called this the typical HR-chatbot design.",
        "D": "IFT improves instruction-following. It doesn't keep frequently changing facts up to date."
      },
      "takeaway": "Private, changing or citable knowledge → RAG. Fine-tuning is for stable behaviour."
    },
    {
      "id": "q2",
      "type": "Concept",
      "stem": "According to the course's decision matrix, which approach gives the strongest control over output style and format?",
      "options": {
        "A": "Prompt engineering",
        "B": "Fine-tuning",
        "C": "RAG",
        "D": "All three are equal on style control."
      },
      "correct": "B",
      "why": {
        "A": "Prompting is 'medium'. It can steer style but can't guarantee it consistently.",
        "B": "Correct. The matrix rates fine-tuning 'extremely high' on style/format control, because the behaviour is baked into the weights.",
        "C": "RAG is rated 'low' on style control. Its strength is external knowledge.",
        "D": "The matrix clearly separates them: fine-tuning > prompting > RAG for style."
      },
      "takeaway": "Matrix in one line: RAG wins knowledge, fine-tuning wins style/format, prompting wins setup cost."
    },
    {
      "id": "q3",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT one of the reasons the course gives for fine-tuning a model?",
      "options": {
        "A": "To get lower cost and latency through shorter prompts and smaller specialised models.",
        "B": "To enforce a consistent tone or output format that prompting cannot guarantee.",
        "C": "To give the model domain expertise and terminology.",
        "D": "To keep the model's knowledge of fast-changing facts up to date every day."
      },
      "correct": "D",
      "why": {
        "A": "This is a stated reason: lower cost and latency.",
        "B": "This is a stated reason: consistent behaviour.",
        "C": "This is a stated reason: domain expertise (medicine, law, finance).",
        "D": "Correct, this is NOT a reason. Fine-tuned knowledge is static, so fast-changing facts belong in RAG."
      },
      "takeaway": "Four reasons to fine-tune: domain expertise, consistent behaviour, higher accuracy, lower cost/latency. Fresh facts isn't one."
    },
    {
      "id": "q4",
      "type": "Compare",
      "stem": "What best describes the relationship between Supervised Fine-Tuning (SFT) and Instruction Fine-Tuning (IFT)?",
      "options": {
        "A": "IFT is a type of alignment fine-tuning that uses human preference rankings.",
        "B": "IFT is a specialised form of SFT: it trains on instruction–response pairs across many tasks, aiming to improve instruction-following in general.",
        "C": "IFT uses unlabelled text, while SFT uses labelled pairs.",
        "D": "SFT always trains on many tasks; IFT trains on exactly one."
      },
      "correct": "B",
      "why": {
        "A": "Preference rankings belong to alignment (RLHF/DPO). IFT learns from demonstrations.",
        "B": "Correct. This is the slide's comparison: SFT may be one task; IFT is many tasks expressed as instructions, measured on unseen instructions.",
        "C": "Both use labelled (input → output) data. IFT's inputs are just phrased as instructions.",
        "D": "It's the other way round. SFT may be a single task; IFT spans many tasks."
      },
      "takeaway": "IFT ⊂ SFT. Same training mechanics, but instruction-formatted, multi-task data."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "The course splits fine-tuning approaches into 'task/style adaptation' (unsupervised, supervised, alignment) and 'parameter-update strategy' (full, PEFT). What is the key point about these two groups?",
      "options": {
        "A": "PEFT can only be used for alignment fine-tuning.",
        "B": "Unsupervised fine-tuning can only be done with full fine-tuning.",
        "C": "They are independent axes: any learning signal can be combined with either full fine-tuning or PEFT.",
        "D": "They are alternatives: you choose either a learning signal or an update strategy."
      },
      "correct": "C",
      "why": {
        "A": "PEFT works with any objective, including SFT.",
        "B": "Unsupervised fine-tuning can also be done with PEFT.",
        "C": "Correct. The professor stressed they are orthogonal: e.g. SFT + LoRA, or unsupervised continued pre-training as full FT.",
        "D": "Every fine-tune has both a learning signal and an update strategy, so they're not alternatives."
      },
      "takeaway": "'What signal?' and 'how many weights?' are two separate questions."
    },
    {
      "id": "q6",
      "type": "Scenario",
      "stem": "A team pastes an entire 20-page product manual into every prompt and asks the model questions about it. There is no search or retrieval step. How should this approach be classified?",
      "options": {
        "A": "Agentic orchestration, because it uses a document as a tool.",
        "B": "Fine-tuning, because the model learns from the manual.",
        "C": "RAG, because external documents are being used.",
        "D": "Prompt engineering with in-context information (long-context prompting)."
      },
      "correct": "D",
      "why": {
        "A": "There is no tool calling or orchestration here.",
        "B": "No weights change, so it isn't fine-tuning.",
        "C": "RAG requires a retrieval step that selects relevant content. Pasting everything isn't retrieval.",
        "D": "Correct. The TAs classified this as prompt engineering with document-grounded, in-context information."
      },
      "takeaway": "No retrieval step = not RAG. No weight update = not fine-tuning."
    },
    {
      "id": "q7",
      "type": "Compare",
      "stem": "What is the key difference between layer-wise and progressive (gradual-unfreezing) fine-tuning?",
      "options": {
        "A": "Progressive fine-tuning only trains the embedding layer.",
        "B": "There is no difference; they are two names for the same method.",
        "C": "Layer-wise trains all layers at once; progressive trains one layer only.",
        "D": "Layer-wise unfreezes one layer per phase and freezes it again; progressive starts at the top and keeps earlier-unfrozen layers trainable as it adds deeper ones."
      },
      "correct": "D",
      "why": {
        "A": "Progressive starts from the top layers, not the embeddings, and can end with the whole network trainable.",
        "B": "The course treats them as distinct strategies, and so does the next session's notebook.",
        "C": "Neither trains everything at once from the start. Both limit what trains per phase.",
        "D": "Correct. Unfreezing is cumulative in progressive (ULMFiT-style) and one-at-a-time in layer-wise."
      },
      "takeaway": "Layer-wise = one at a time, re-freeze. Progressive = top-down, cumulative unfreezing."
    },
    {
      "id": "q8",
      "type": "Concept",
      "stem": "What is curriculum fine-tuning?",
      "options": {
        "A": "Training several tasks at the same time with a shared encoder.",
        "B": "Repeating the hardest examples many times so the model memorises them.",
        "C": "Presenting training examples in a meaningful order, typically easy → hard, using a difficulty measure and a schedule.",
        "D": "Fine-tuning a model on educational datasets such as school textbooks."
      },
      "correct": "C",
      "why": {
        "A": "That's multi-task fine-tuning.",
        "B": "Memorising hard examples would be overfitting. Even the 'anti-curriculum' variant just orders hard examples first.",
        "C": "Correct. Two ingredients: a difficulty measure and a pacing schedule. Benefits: faster convergence, stability, better generalisation.",
        "D": "'Curriculum' refers to the ordering of examples, not school-related data."
      },
      "takeaway": "Curriculum = easy → hard ordering (difficulty measure + schedule)."
    },
    {
      "id": "q9",
      "type": "Scenario",
      "stem": "You need one deployed model that handles sentiment analysis, topic classification and similarity scoring for the same customer reviews. In the course's multi-task setup, how is this usually built?",
      "options": {
        "A": "One shared encoder plus a separate small head for each task, each with its own loss (e.g. cross-entropy for classification, MSE for similarity).",
        "B": "One head shared by all tasks with a single cross-entropy loss.",
        "C": "Train on each task in sequence and throw away the earlier tasks' data.",
        "D": "Three completely separate fine-tuned models, one per task."
      },
      "correct": "A",
      "why": {
        "A": "Correct. This matches the notebook: DistilBERT encoder shared, one linear head per task, per-task loss, tasks interleaved.",
        "B": "Different tasks have different output shapes (2 classes, 4 classes, a real number), so they need separate heads and losses.",
        "C": "Sequential training without old data causes forgetting. Multi-task mixes tasks during training.",
        "D": "That's the opposite of multi-task fine-tuning, which exists to avoid separate models."
      },
      "takeaway": "Multi-task FT = shared body + per-task head + per-task loss."
    },
    {
      "id": "q10",
      "type": "Concept",
      "stem": "Which statement about full fine-tuning is correct?",
      "options": {
        "A": "It updates only the output head of the model.",
        "B": "It updates every parameter, gives the most capacity to adapt, but needs much more memory and risks catastrophic forgetting.",
        "C": "It cannot cause the model to forget its general skills.",
        "D": "It needs less GPU memory than PEFT because nothing new is added."
      },
      "correct": "B",
      "why": {
        "A": "Full FT updates everything: embeddings, attention, feed-forward, norms and the head.",
        "B": "Correct. These are the slide's three points: maximum adaptation, high cost, forgetting.",
        "C": "Aggressive updates to all weights are the main cause of catastrophic forgetting.",
        "D": "Full FT needs far more memory: every parameter carries a gradient and optimizer states."
      },
      "takeaway": "Full FT: best adaptation, highest memory, highest forgetting risk."
    }
  ]
};
