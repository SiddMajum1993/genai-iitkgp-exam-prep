// Practice questions for Chapter 02 · Data preparation & training mechanics.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch02"] = {
  "title": "Data preparation & training mechanics",
  "questions": [
    {
      "id": "q1",
      "type": "Symptom → diagnosis",
      "stem": "During fine-tuning, training loss keeps falling steadily, but after the second epoch validation loss starts to rise. What is happening, and what should you do?",
      "options": {
        "A": "Catastrophic forgetting: add general-purpose replay data.",
        "B": "Overfitting: stop early and keep the checkpoint at the validation-loss minimum (and consider fewer epochs, regularisation or more data).",
        "C": "No learning: check for frozen parameters or a bug.",
        "D": "Underfitting: train for more epochs with a higher learning rate."
      },
      "correct": "B",
      "why": {
        "A": "Forgetting shows up on general benchmarks, not as a rising validation loss on the same task.",
        "B": "Correct. Train down + validation up is the textbook overfitting pattern. The model is memorising the training set.",
        "C": "'No learning' means both curves are flat at the start value. Here training loss is clearly moving.",
        "D": "Underfitting means both losses stay high. Here training loss is falling fine."
      },
      "takeaway": "Train ↓ + validation ↑ = overfitting → early stopping at the validation minimum."
    },
    {
      "id": "q2",
      "type": "Scenario",
      "stem": "A company fully fine-tunes a model on its legal contracts. Contract-review accuracy improves a lot. Weeks later, users notice it has become worse at general reasoning and following ordinary instructions. Nobody saw this during training. Which statement best explains it?",
      "options": {
        "A": "This is domain shift caused by the deployment hardware.",
        "B": "This is catastrophic forgetting, which is 'silent' because the target-task metric improved while general skills, which weren't being measured, degraded.",
        "C": "This is a loss spike that will repair itself.",
        "D": "This is overfitting, visible as rising validation loss on the contract data."
      },
      "correct": "B",
      "why": {
        "A": "Hardware doesn't cause this. The weights were overwritten by narrow-domain training.",
        "B": "Correct. Narrow full fine-tuning overwrote general knowledge. The fix is to always run a general eval suite before and after.",
        "C": "Loss spikes are short-term training events, not a lasting loss of general skills.",
        "D": "Overfitting is about the target task's validation loss. Here the target task actually improved."
      },
      "takeaway": "Catastrophic forgetting is silent: measure general skills before and after every fine-tune."
    },
    {
      "id": "q3",
      "type": "Which is best",
      "stem": "You must fully fine-tune a model and want to reduce catastrophic forgetting. Which single option does the course call the cheapest baseline that should always be used?",
      "options": {
        "A": "Weight averaging (WiSE-FT) after training.",
        "B": "Elastic Weight Consolidation (EWC) with Fisher information.",
        "C": "Conservative hyperparameters: a low learning rate, only 1–2 epochs, warm-up + cosine decay, and early stopping on general evals.",
        "D": "Collecting a new general-purpose dataset for replay."
      },
      "correct": "C",
      "why": {
        "A": "WiSE-FT is a repair step for a model that has already degraded.",
        "B": "EWC works, but it costs memory for the original weights and importance scores. It's for when full FT is required and no general data exists.",
        "C": "Correct. The mitigation table lists conservative hyperparameters as 'always, the default baseline' with no extra cost. 'Most forgetting horror stories are just aggressive schedules.'",
        "D": "Replay is effective but needs data curation, so it isn't the free baseline."
      },
      "takeaway": "First defence against forgetting: a low LR and few epochs (free). Then add replay, regularisation or freezing as needed."
    },
    {
      "id": "q4",
      "type": "Compare",
      "stem": "What is the difference between a loss spike and divergence?",
      "options": {
        "A": "A loss spike only happens with bf16, and divergence only with fp16.",
        "B": "They are the same thing; divergence is just a bigger spike.",
        "C": "A loss spike recovers by itself after some wasted steps; divergence does not recover, because the weights have left every good region and the loss keeps rising.",
        "D": "Divergence happens at the start of training, and spikes only at the very end."
      },
      "correct": "C",
      "why": {
        "A": "Precision affects the risk of overflow, but neither event is tied to one format.",
        "B": "The key difference is recovery: a spike self-repairs, divergence doesn't.",
        "C": "Correct. Spike → recovers; gradient explosion → the warning stage; divergence → terminal.",
        "D": "Spikes tend to cluster just after warm-up, when the LR is highest. Divergence can happen any time the feedback loop takes over."
      },
      "takeaway": "Spike = recovers. Explosion = growing gradients. Divergence = no recovery."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "What does gradient clipping (for example max_grad_norm = 1.0) actually do?",
      "options": {
        "A": "It rescales any gradient whose size exceeds the cap, keeping its direction but limiting its magnitude.",
        "B": "It deletes any batch that produces a large gradient.",
        "C": "It freezes the layers whose gradients are largest.",
        "D": "It permanently lowers the learning rate after a spike."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Clipping is 'a seatbelt, not a steering wheel'. If more than ~20% of steps clip, lower the LR.",
        "B": "No data is thrown away. The gradient is just scaled down.",
        "C": "No layers are frozen.",
        "D": "The learning rate is unchanged. Clipping only affects the size of that step's gradient."
      },
      "takeaway": "Clipping caps the gradient's size (direction kept). Frequent clipping means the LR is too high."
    },
    {
      "id": "q6",
      "type": "Scenario",
      "stem": "A team is doing continual training and doesn't know how long the run will last. They want to be able to stop at any point and still finish with a well-converged model. Which learning-rate schedule fits best?",
      "options": {
        "A": "Cosine decay planned for a fixed total number of steps.",
        "B": "Constant learning rate with no warm-up.",
        "C": "Cosine with restarts, to re-initialise the model each cycle.",
        "D": "Warmup–Stable–Decay (WSD): hold the peak LR for most of the run, then decay sharply whenever they decide to stop."
      },
      "correct": "D",
      "why": {
        "A": "Cosine needs to know the total length in advance to shape the decay.",
        "B": "Constant without warm-up risks early instability and never decays to settle the model.",
        "C": "Restarts don't re-initialise weights (only the LR jumps). Their purpose is getting several candidate checkpoints.",
        "D": "Correct. The course matches WSD to 'run length unknown / continual training', because decay can be started on demand."
      },
      "takeaway": "Known length → cosine (the default). Unknown length → WSD. Short runs → constant + warm-up."
    },
    {
      "id": "q7",
      "type": "Scenario",
      "stem": "A fraud-detection model reports 99.8% accuracy on a dataset where 0.2% of transactions are fraud. On closer inspection it never flags any fraud. What concept does this illustrate, and what is the right fix?",
      "options": {
        "A": "Domain shift; add retrieval.",
        "B": "Overfitting; reduce the number of epochs.",
        "C": "Data leakage; split the data before preprocessing.",
        "D": "The accuracy paradox; evaluate with precision, recall, F1 or PR-AUC, and handle the imbalance with resampling or class weights."
      },
      "correct": "D",
      "why": {
        "A": "There is no training vs deployment mismatch here.",
        "B": "Nothing here says the model memorised the training data. The issue is the metric on skewed data.",
        "C": "Leakage inflates scores through test information in training. That isn't what's described.",
        "D": "Correct. Predicting 'never fraud' gets high accuracy on imbalanced data while catching nothing."
      },
      "takeaway": "Skewed classes → never trust plain accuracy. Use recall/precision/F1/PR-AUC plus resampling or class weights."
    },
    {
      "id": "q8",
      "type": "Which is NOT",
      "stem": "Which of these is NOT a data-leakage risk when splitting data?",
      "options": {
        "A": "Letting the same patient's records appear in both train and test sets.",
        "B": "Randomly splitting time-series data so later records end up in training.",
        "C": "Reshuffling the training set into new mini-batches at every epoch.",
        "D": "Normalising features using statistics computed on the full dataset before splitting."
      },
      "correct": "C",
      "why": {
        "A": "This is leakage: grouped data needs a group-aware split.",
        "B": "This is leakage: the model 'sees the future'. Use time-based splits.",
        "C": "Correct, this is NOT leakage. Reshuffling only reorders training data; no validation or test information enters training.",
        "D": "This is leakage: test statistics influence training. Split first and fit transforms on train only."
      },
      "takeaway": "Leakage = validation or test information entering training. Reshuffling batches is safe."
    },
    {
      "id": "q9",
      "type": "Concept",
      "stem": "The course's 'golden rule' of formatting fine-tuning data says what?",
      "options": {
        "A": "Always convert data to plain text, with no roles or structure.",
        "B": "Match the base model's chat template exactly, because a mismatch can silently ruin a fine-tune even when the content is perfect.",
        "C": "Format only the validation set; training data can be raw.",
        "D": "Use as many different formats as possible to increase diversity."
      },
      "correct": "B",
      "why": {
        "A": "Modern chat fine-tuning expects structured messages with system/user/assistant roles.",
        "B": "Correct. This is why the notebooks used apply_chat_template.",
        "C": "Training data must be in the exact format too. It's what the model learns from.",
        "D": "Inconsistent formatting is listed as a data-quality problem, not a strength."
      },
      "takeaway": "Format = the base model's exact chat template. A mismatch fails silently."
    },
    {
      "id": "q10",
      "type": "Compare",
      "stem": "To reduce forgetting, one option adds a penalty for moving away from the pre-trained weights (L2-SP/EWC); another freezes the lower layers. Which statement is correct?",
      "options": {
        "A": "Regularisation allows weights to change but penalises drift (EWC holds important weights tighter); freezing allows no change to those weights, though gradients still flow through them to the trainable layers.",
        "B": "Both methods stop gradients from flowing through the affected layers.",
        "C": "Freezing is more expensive in memory than EWC.",
        "D": "EWC treats every weight as equally important."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Penalise vs forbid. Backprop still passes through frozen layers so the upper layers learn correctly.",
        "B": "Gradients still pass through frozen layers. The weights just aren't updated.",
        "C": "It's the opposite. Freezing has no extra cost; EWC must store the original weights and importance scores.",
        "D": "EWC weights the penalty by Fisher information, i.e. each parameter's importance to old tasks."
      },
      "takeaway": "Regularisation = 'change, but stay close'. Freezing = 'don't change'. EWC = stay closest on the important weights."
    }
  ]
};
