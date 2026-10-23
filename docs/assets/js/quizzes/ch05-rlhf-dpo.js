// Practice questions for Chapter 05 · Alignment: RLHF & DPO.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch05"] = {
  "title": "Alignment: RLHF & DPO",
  "questions": [
    {
      "id": "q1",
      "type": "Concept",
      "stem": "Compared with RLHF using PPO, what does DPO remove from the training process?",
      "options": {
        "A": "The need for human preference data.",
        "B": "The supervised fine-tuning stage, which DPO forbids.",
        "C": "The separate reward model and the reinforcement-learning loop.",
        "D": "The frozen reference model."
      },
      "correct": "C",
      "why": {
        "A": "DPO learns from exactly the same kind of preference comparisons that RLHF uses.",
        "B": "SFT on the chosen answers is often done first. The course's DPO workflow shows an optional SFT step before DPO.",
        "C": "Correct. DPO optimises the policy directly on (prompt, chosen, rejected) triples. There's no reward model to fit and no RL loop to run.",
        "D": "DPO still needs a frozen reference. Chosen and rejected probabilities are measured relative to it."
      },
      "takeaway": "DPO = no reward model, no RL. It keeps a policy and a frozen reference."
    },
    {
      "id": "q2",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT one of the four models involved in the PPO stage of RLHF?",
      "options": {
        "A": "The value function (critic) that estimates expected reward.",
        "B": "The frozen reference (SFT) model used for the KL penalty.",
        "C": "The policy model being trained.",
        "D": "A retriever model that fetches documents for each answer."
      },
      "correct": "D",
      "why": {
        "A": "The value function is one of the four. It's needed to compute the advantage.",
        "B": "The reference model is the frozen SFT copy. It's one of the four.",
        "C": "The policy is the LLM being aligned. It's one of the four.",
        "D": "Correct. Retrievers belong to RAG, not to PPO. The fourth model is the reward model."
      },
      "takeaway": "PPO's four models: policy, reference, reward, value."
    },
    {
      "id": "q3",
      "type": "Scenario",
      "stem": "After RL fine-tuning, a chatbot answers 'Why is the sky blue?' with 'What a brilliant question! Your intellect is remarkable…' and never explains anything, yet the reward model scores it very highly. What is this, and what is the standard guard against it?",
      "options": {
        "A": "Hallucination; add retrieval over a knowledge base.",
        "B": "Reward hacking; add a KL penalty that keeps the policy close to the frozen reference model.",
        "C": "Over-refusal; lower the harmlessness weight.",
        "D": "Catastrophic forgetting; fix it by training for more epochs."
      },
      "correct": "B",
      "why": {
        "A": "No false facts were stated. The problem is gaming the scorer, not making things up.",
        "B": "Correct. The policy found a blind spot in the reward model (it loves flattery). The KL penalty against the reference model limits that kind of drift.",
        "C": "The model didn't refuse anything. It produced empty flattery.",
        "D": "This isn't loss of old knowledge, and more epochs would let the model exploit the reward model even further."
      },
      "takeaway": "High score, useless answer = reward hacking. The KL penalty to the reference model is the guard."
    },
    {
      "id": "q4",
      "type": "Why",
      "stem": "Why does RLHF train its reward model on pairwise comparisons ('A is better than B') instead of asking people for absolute scores like 7 out of 10?",
      "options": {
        "A": "People are much more consistent at comparing two answers than at assigning absolute scores.",
        "B": "Comparisons need no human labelers at all.",
        "C": "The PPO algorithm only accepts binary inputs.",
        "D": "Absolute scores can't be stored in a dataset."
      },
      "correct": "A",
      "why": {
        "A": "Correct. 'People are good at comparison.' Absolute ratings vary wildly between people, while 'which is better?' is reliable.",
        "B": "Comparisons still come from human labelers (or a reviewed LLM judge).",
        "C": "PPO consumes a scalar reward from the reward model. It doesn't require binary inputs.",
        "D": "Storing numbers is trivial. Storage isn't the issue."
      },
      "takeaway": "Reward models learn from comparisons because humans compare well and score badly."
    },
    {
      "id": "q5",
      "type": "Concept",
      "stem": "During reward-model training, the loss is small when the chosen answer's score is well above the rejected answer's, and large when the rejected answer scores higher. What does this push the reward model to learn?",
      "options": {
        "A": "To give higher scores to the kinds of answers humans prefer.",
        "B": "To give every answer the same score.",
        "C": "To copy the reference model's probabilities.",
        "D": "To generate better answers itself."
      },
      "correct": "A",
      "why": {
        "A": "Correct. That's the Bradley-Terry idea: the only way to keep the loss low across many pairs is to score preferred answers higher.",
        "B": "Equal scores would give a middling loss on every pair. The loss rewards a gap in the right direction.",
        "C": "Matching the reference is the KL penalty's job in PPO, not the reward model's.",
        "D": "The reward model only scores. The policy is what generates answers."
      },
      "takeaway": "Reward model = a learned judge that turns 'A beats B' into a score."
    },
    {
      "id": "q6",
      "type": "Compare",
      "stem": "A team has a fixed, offline dataset of 50,000 (prompt, chosen, rejected) records and limited GPU budget. They want a stable training run. Which approach fits best?",
      "options": {
        "A": "RLHF with PPO, because it always gives better alignment.",
        "B": "DPO, because it trains directly on static preference pairs with just a policy and a frozen reference.",
        "C": "Unsupervised fine-tuning on the chosen answers only, with no rejected answers.",
        "D": "Pre-training a new model from scratch on the preference data."
      },
      "correct": "B",
      "why": {
        "A": "PPO isn't always better, and it needs four models plus an unstable RL loop, which is a poor fit for a tight budget.",
        "B": "Correct. A static preference dataset, a small budget and a need for stability are exactly DPO's sweet spot.",
        "C": "That throws away the rejected answers, which are the signal that makes preference tuning work.",
        "D": "Pre-training is by far the most expensive option and isn't how preference data is used."
      },
      "takeaway": "Fixed preference pairs + limited compute + need for stability → DPO."
    },
    {
      "id": "q7",
      "type": "Scenario",
      "stem": "In which situation would the course suggest still choosing RLHF with PPO over DPO?",
      "options": {
        "A": "When you already have a clean, static preference dataset.",
        "B": "When you want the fewest models in GPU memory.",
        "C": "When a coding agent must explore online and is scored by a complex or changing outcome, such as whether tests pass.",
        "D": "When you want to avoid tuning any hyperparameters."
      },
      "correct": "C",
      "why": {
        "A": "A static preference dataset is DPO's natural setting.",
        "B": "PPO needs four models. DPO needs two.",
        "C": "Correct. Online exploration with a measurable, sequential or changing reward is where RL still shines.",
        "D": "PPO is known to be very sensitive to hyperparameters."
      },
      "takeaway": "PPO is still useful for online exploration and complex or changing rewards."
    },
    {
      "id": "q8",
      "type": "Concept",
      "stem": "In DPO, what does the hyperparameter β control?",
      "options": {
        "A": "The fraction of the dataset held out for testing.",
        "B": "The LoRA rank used for the adapter.",
        "C": "How strongly the preference signal pushes the policy, and how far it may drift from the frozen reference.",
        "D": "The number of reward models trained."
      },
      "correct": "C",
      "why": {
        "A": "Data splits are set separately. β is part of the loss.",
        "B": "Rank is a LoRA setting (r), unrelated to β.",
        "C": "Correct. β scales the preference signal relative to the reference. In the hands-on, 'higher = more aggressive'; the course used β = 0.1.",
        "D": "DPO trains no reward model."
      },
      "takeaway": "β = strength of the push / leash to the reference model."
    },
    {
      "id": "q9",
      "type": "Mapping",
      "stem": "When RLHF treats a language model as a reinforcement-learning agent, which mapping is correct?",
      "options": {
        "A": "State = the KV cache; policy = the tokenizer.",
        "B": "State = the reward model; action = the KL penalty.",
        "C": "State = the model's weights; action = a full answer.",
        "D": "State = the prompt plus the tokens generated so far; action = the next token; policy = the LLM."
      },
      "correct": "D",
      "why": {
        "A": "The KV cache is an implementation detail, and the tokenizer isn't the policy.",
        "B": "The reward model provides the reward signal. The KL penalty is part of the objective, not an action.",
        "C": "Weights are the policy's parameters, not the state, and each action is a single token.",
        "D": "Correct. This is the course's mapping. A trajectory is one full generated answer."
      },
      "takeaway": "State = text so far, action = next token, policy = LLM, trajectory = one full answer."
    },
    {
      "id": "q10",
      "type": "Symptom → read",
      "stem": "While running DPO, the loss starts at about 0.69 and steadily falls below it, and the margin between the chosen and rejected implicit rewards grows positive. How should you read this?",
      "options": {
        "A": "The data is mislabelled: chosen and rejected are swapped.",
        "B": "Training is working: the policy increasingly prefers chosen over rejected answers relative to the reference.",
        "C": "The model is reward hacking, so training should stop.",
        "D": "The reference model is being updated by mistake."
      },
      "correct": "B",
      "why": {
        "A": "Swapped labels would push the margin negative and the loss above 0.69.",
        "B": "Correct. At step 0 the policy equals the reference, so the loss is ln 2 ≈ 0.69. Going below it with a positive margin means the model is separating chosen from rejected.",
        "C": "There's no reward model to hack in DPO. These are the healthy signs.",
        "D": "A rising margin is expected when only the policy changes. Nothing here suggests the reference moved."
      },
      "takeaway": "Healthy DPO: loss falls below ln 2, chosen reward rises, rejected falls, margin is positive."
    }
  ]
};
