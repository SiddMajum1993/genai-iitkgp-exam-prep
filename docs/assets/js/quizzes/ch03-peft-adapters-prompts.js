// Practice questions for Chapter 03 · PEFT: adapters & soft prompts.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch03"] = {
  "title": "PEFT: adapters & soft prompts",
  "questions": [
    {
      "id": "q1",
      "type": "Concept",
      "stem": "During full fine-tuning with Adam, which item takes the MOST GPU memory per trained parameter?",
      "options": {
        "A": "The FP16 gradient.",
        "B": "Adam's two optimizer states (momentum and variance), kept in FP32.",
        "C": "The FP16 weight itself.",
        "D": "The tokenizer vocabulary."
      },
      "correct": "B",
      "why": {
        "A": "The gradient is also 2 bytes, the same as the weight.",
        "B": "Correct. Two FP32 states at 4 bytes each make 8 bytes per parameter, about four times the weights. Freezing a weight removes this cost.",
        "C": "The weight takes 2 bytes. That's a quarter of what Adam's states need.",
        "D": "The vocabulary is tiny and fixed. It isn't a per-parameter training cost."
      },
      "takeaway": "Adam states dominate training memory. Frozen weights need no gradients and no Adam states, which is why PEFT saves so much."
    },
    {
      "id": "q2",
      "type": "Scenario",
      "stem": "A company fine-tuned one model sequentially on HR, then legal, then IT-ticket data. It is now great at IT tickets but much worse at HR questions. What is this called, and what does PEFT do about it?",
      "options": {
        "A": "Hallucination. PEFT fixes it by retrieving documents at inference.",
        "B": "Overfitting. PEFT fixes it by adding dropout to every layer.",
        "C": "Quantization error. PEFT fixes it by storing weights in 16-bit.",
        "D": "Catastrophic forgetting. PEFT freezes the shared backbone and keeps a separate small adapter or prompt per task."
      },
      "correct": "D",
      "why": {
        "A": "Retrieving documents is RAG, and the symptom here isn't made-up facts.",
        "B": "Overfitting is poor generalisation on one task. Losing an earlier task after training on new ones is forgetting.",
        "C": "Nothing was quantized, so quantization error doesn't apply.",
        "D": "Correct. Full fine-tuning overwrites general and earlier knowledge. With a frozen backbone and task-specific modules, nothing is overwritten."
      },
      "takeaway": "Catastrophic forgetting comes from full fine-tuning; task-specific adapters on a frozen backbone are the PEFT fix."
    },
    {
      "id": "q3",
      "type": "Classify",
      "stem": "BitFit fine-tunes only the bias terms of an existing model and adds no new parameters. Which PEFT family does it belong to?",
      "options": {
        "A": "Soft prompt",
        "B": "Re-parameterization",
        "C": "Additive",
        "D": "Selective"
      },
      "correct": "D",
      "why": {
        "A": "Soft-prompt methods add learned virtual tokens. BitFit doesn't.",
        "B": "Re-parameterization (LoRA) learns a low-rank form of the weight change in new matrices.",
        "C": "Additive methods insert new modules, such as adapters. BitFit adds nothing.",
        "D": "Correct. Selective PEFT unfreezes a chosen subset of existing parameters, such as bias terms or layer norms."
      },
      "takeaway": "Selective = train some existing weights, add none. Additive = add modules. Soft prompt = add virtual tokens. Re-param = low-rank update."
    },
    {
      "id": "q4",
      "type": "Why",
      "stem": "In an adapter, the up-projection is initialised to (near) zero so that the adapter starts close to an identity function. Why is this done?",
      "options": {
        "A": "Because zero weights make inference faster.",
        "B": "To make the bottleneck dimension smaller.",
        "C": "So the newly inserted adapter initially changes nothing, letting the model start from its pre-trained behaviour and adapt gradually.",
        "D": "To force the adapter to remain an identity function throughout training."
      },
      "correct": "C",
      "why": {
        "A": "Initial values don't affect inference speed, and the weights won't stay zero anyway.",
        "B": "The bottleneck size is a separate design choice (e.g. 32 or 64), not set by initial values.",
        "C": "Correct. With f(h) ≈ 0, h′ = h + f(h) ≈ h, so inserting the adapter doesn't disturb the pre-trained model at step 0.",
        "D": "Near-identity is only the starting condition. The adapter learns freely afterwards."
      },
      "takeaway": "Near-identity initialisation is 'start by doing nothing'. It's the same idea as B = 0 in LoRA."
    },
    {
      "id": "q5",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT a limitation of sequential (Houlsby) adapters listed in the lecture?",
      "options": {
        "A": "They add inference latency because data must pass through them step by step.",
        "B": "Their parameter cost grows with the depth of the network.",
        "C": "They cause catastrophic forgetting of the frozen backbone.",
        "D": "They cannot be parallelised with the existing computation."
      },
      "correct": "C",
      "why": {
        "A": "This is a listed limitation. The pre-read calls latency the primary practical objection.",
        "B": "This is a listed limitation. More layers means more adapters.",
        "C": "Correct, this is NOT a limitation. The backbone is frozen, so adapters actually help avoid forgetting.",
        "D": "This is a listed limitation. They sit in series with the sub-layers."
      },
      "takeaway": "Houlsby adapters' weakness is latency (they're sequential), not forgetting. Lin's residual adapters run in parallel."
    },
    {
      "id": "q6",
      "type": "Compare",
      "stem": "What is the key difference between Houlsby-style sequential adapters and Lin et al.'s residual adapters?",
      "options": {
        "A": "Residual adapters are a soft-prompt method.",
        "B": "Houlsby adapters have no bottleneck, while residual adapters do.",
        "C": "Residual adapters update the frozen backbone weights, while Houlsby adapters don't.",
        "D": "Residual adapters run in parallel with the attention/feed-forward sub-layer and merge outputs, avoiding the extra sequential step of Houlsby adapters."
      },
      "correct": "D",
      "why": {
        "A": "Both are additive PEFT. Soft prompts add virtual tokens, not layers.",
        "B": "Both use the down-project → non-linearity → up-project bottleneck.",
        "C": "Both keep the backbone frozen. Only the adapter weights train.",
        "D": "Correct. Placement is the difference: series (Houlsby) vs parallel (Lin), which affects latency."
      },
      "takeaway": "Same bottleneck shape, different placement: Houlsby = in series (slower), Lin = in parallel."
    },
    {
      "id": "q7",
      "type": "Compare",
      "stem": "A student says: 'Prompt tuning and prefix tuning are the same thing.' Which correction is accurate?",
      "options": {
        "A": "Prefix tuning trains the W_k and W_v projection matrices, while prompt tuning trains nothing.",
        "B": "Both learn virtual tokens with a frozen model, but prompt tuning adds them only at the input layer, while prefix tuning prepends learned key/value vectors in every attention layer.",
        "C": "Prompt tuning uses human-written words, while prefix tuning uses learned vectors.",
        "D": "Prefix tuning changes the queries Q, while prompt tuning changes the keys."
      },
      "correct": "B",
      "why": {
        "A": "In prefix tuning W_k and W_v stay frozen. Only the extra P_k and P_v vectors are trained. Prompt tuning trains its virtual-token embeddings.",
        "B": "Correct. Location is the difference: input layer only (prompt tuning) vs K/V of every layer (prefix tuning).",
        "C": "Both use learned (soft) vectors. Human-written words are discrete prompts.",
        "D": "Queries are unchanged in prefix tuning. Learned prefixes are added to K and V."
      },
      "takeaway": "Prompt tuning = input only. Prefix tuning = K′ = [P_k; K], V′ = [P_v; V] at every layer. Backbone frozen in both."
    },
    {
      "id": "q8",
      "type": "Scenario",
      "stem": "A team uses prompt tuning with a single 100-token soft prompt. It is costly, and analysis shows many of those tokens are not useful for most inputs. Which method from the lecture targets exactly this problem?",
      "options": {
        "A": "SMoP (Sparse Mixture of Prompts)",
        "B": "Full fine-tuning",
        "C": "IDPG",
        "D": "APT (Adaptive Prefix Tuning)"
      },
      "correct": "A",
      "why": {
        "A": "Correct. SMoP replaces one long prompt with several short prompts plus a router that selects the best one(s) per input.",
        "B": "Full fine-tuning is far more expensive and doesn't address soft-prompt length.",
        "C": "IDPG generates a different prompt for each input. Its target problem is 'one fixed prompt for all inputs', not cost from one long prompt.",
        "D": "APT adjusts prefix length per layer. Its target is 'same prefix length at every layer'."
      },
      "takeaway": "Long costly prompt → SMoP. Same length at every layer → APT. Same prompt for every input → IDPG."
    },
    {
      "id": "q9",
      "type": "Which is true",
      "stem": "Which statement about soft prompts is correct?",
      "options": {
        "A": "A soft prompt compresses a long text prompt into fewer words.",
        "B": "A soft prompt is a set of learned vectors tied to one model's embedding space, so it generally cannot be reused on a different LLM.",
        "C": "A soft prompt retrieves relevant documents at inference time, like RAG.",
        "D": "A soft prompt is a system prompt written by a developer and stored with the model."
      },
      "correct": "B",
      "why": {
        "A": "That describes prompt compression, a different technique.",
        "B": "Correct. Soft prompts live in a specific model's embedding space, so they aren't transferable across LLMs.",
        "C": "RAG adds retrieved knowledge. Soft prompts steer behaviour and retrieve nothing.",
        "D": "System prompts are human-written text. Soft prompts are trained vectors."
      },
      "takeaway": "Soft prompt = learned vectors, trained by backprop for one model. Not a system prompt, not RAG, not compression."
    },
    {
      "id": "q10",
      "type": "Scenario",
      "stem": "In Adaptive Prefix Tuning (APT), learned prefix weights for an entity-relation extraction task concentrated in the bottom layers. What explanation did the lecture give?",
      "options": {
        "A": "Top layers have no attention, so prefixes cannot be added there.",
        "B": "Bottom layers are frozen while top layers are trainable.",
        "C": "Bottom layers capture concrete, phrase-level features, which this kind of task relies on, so they benefit from a longer prefix.",
        "D": "APT always places all prefix tokens in the first layer."
      },
      "correct": "C",
      "why": {
        "A": "Every transformer layer has attention. Prefixes can go in all of them.",
        "B": "The whole backbone is frozen in APT. Only the prefixes and gates are learned.",
        "C": "Correct. Lower layers handle shallow, phrase-level information, while top layers handle abstract meaning. COPA (commonsense reasoning) leaned on higher layers instead.",
        "D": "APT learns a different prefix length per layer. That's the whole point."
      },
      "takeaway": "APT: different layers need different prefix lengths. Lower layers handle phrase-level detail, upper layers handle semantics."
    }
  ]
};
