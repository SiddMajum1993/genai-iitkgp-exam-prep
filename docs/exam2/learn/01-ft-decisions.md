# 01 · When to fine-tune: Prompting vs RAG vs Fine-tuning, and the kinds of fine-tuning

**Session:** 11 July 2026 · Prof. Jiaul Paik (IIT Kharagpur) · Module 3, *LLM Fine-tuning Fundamentals I: Decision Framework & Fine-tuning Approaches*
**Read time:** 35–45 min · **Builds on:** the 1st-exam topics (transformers, prompt engineering, RAG)

---

## In one minute

A pre-trained LLM is a brilliant generalist. When it doesn't do what you need, you have three ways to fix it:

- **Prompt engineering.** Change the *instructions*. The model stays the same. It's instant and free.
- **RAG.** Change *what the model can see*: fetch relevant documents at question time and put them in the prompt. Best when the problem is **missing, private or changing facts**.
- **Fine-tuning.** Change *the model itself* by training its weights on your examples. Best for **consistent style, format and behaviour** on a narrow task.

The course's rule of thumb is to **try prompting first, then RAG, then fine-tuning**, and to combine them when needed.

The second half of the lecture sorted fine-tuning into a **taxonomy**:

- **By what it teaches:** unsupervised, supervised or alignment fine-tuning.
- **By how many weights change:** full fine-tuning or PEFT.
- **Four kinds of full fine-tuning:** SFT, instruction FT, multi-task FT and curriculum FT.
- **Three ways to make full fine-tuning cheaper:** layer-wise, block-wise and progressive.

If you remember one sentence: **"RAG changes what the model *knows* at runtime; fine-tuning changes how the model *behaves*."**

---

## Exam radar

| Topic | Likelihood | Why we think so |
|---|---|---|
| Prompt vs RAG vs Fine-tuning: which to use for a given scenario | 🔴 High | 5 slides, a decision matrix, pre-read and post-read, ~25 Q&A questions (the most-asked topic). It also suits the 1st exam's "a team wants X" question style |
| The decision matrix: which approach is strong on what | 🔴 High | A single slide that packs the whole comparison into a table, which makes it perfect MCQ material |
| SFT vs Instruction Fine-tuning (IFT) | 🔴 High | A dedicated comparison slide, two notebooks and several Q&A rows |
| The taxonomy: unsupervised, supervised and alignment vs full and PEFT (two *independent* axes) | 🔴 High | The professor's own clarification, repeated by the TAs |
| PEFT's three families: additive, selective, re-parameterization | 🟠 Medium | A preview here, taught in depth in Chapters 03 and 04 |
| Multi-task fine-tuning: shared body plus a head per task | 🟠 Medium | Slides and a notebook |
| Curriculum fine-tuning: easy → hard, and why it helps | 🟠 Medium | 6 slides, with named strategies |
| Layer-wise vs block-wise vs progressive fine-tuning | 🟠 Medium | Diagram slides plus a notebook in the next session |
| Why full fine-tuning is expensive (memory) and why it causes forgetting | 🟠 Medium | Slides and Q&A. The exact numbers are inconsistent (see Concept 9) |
| Encoder-only vs decoder-only vs encoder-decoder (recap) | 🟢 Low | A recap from the 1st exam, but students still asked about it |
| Med-PaLM and BloombergGPT facts | 🟢 Low | Examples only |

---

## The story that frames the lecture: generalist vs specialist

The professor showed the same question going to a **base model** and to a **fine-tuned model**:

> A patient describes **skin irritation, redness and itching**.
>
> - **Base LLM:** "Probably acne, visit a dermatologist."
> - **Fine-tuned LLM** (trained on dermatology data): "A mix of non-inflammatory and inflammatory acne", which is a specific, expert-style answer.

Real-world specialists built this way:

- **Med-PaLM** (Google) is PaLM fine-tuned on medical Q&A. Med-PaLM 2 reached expert level on US medical-licensing-style questions.
- **BloombergGPT** was trained on decades of financial filings and news, and beat general models of similar size on finance tasks.

!!! example "Everyday analogy"
    A general physician (GP) knows a bit of everything. A dermatologist went through extra, focused training. Fine-tuning is that extra residency: same doctor, new specialisation.

---

## Concept 1 — Why fine-tune at all?

The slides give **four reasons**:

| Reason | Plain meaning | Example |
|---|---|---|
| **Domain expertise** | Learn the jargon and conventions of a field | Medical, legal or finance terms |
| **Consistent behaviour** | Always follow a tone, format or policy that prompting can't *guarantee* | Every reply in your company's support style |
| **Higher accuracy** | A small fine-tuned model can beat a much bigger general model on one narrow task | A 3B model trained on your ticket categories |
| **Lower cost and latency** | No long instructions in every prompt, and a smaller model | Faster, cheaper inference |

### What fine-tuning is *not* good at

Fine-tuning shapes **behaviour and format**. It is an **unreliable way to add lots of new facts**, and those facts become frozen at training time. For facts, use RAG (Concept 2).

### Common misunderstanding

- **"Fine-tuning means training from scratch."** No. Fine-tuning *starts from pre-trained weights* and does a smaller, targeted round of training ("mini training"). The forward pass, the loss and backpropagation are the same as in normal training; only the starting point is different.

---

## Concept 2 — The three pathways: Prompting, RAG, Fine-tuning

| Pathway | What you change | How it works | Best for |
|---|---|---|---|
| **Prompt engineering** | The *instructions* | Write clear instructions, a role, examples and an output format; check the output; refine. The model is frozen. | Quick prototypes, standard tasks, zero setup |
| **RAG** | What the model *sees* | Documents → vector DB. At question time, retrieve the top-k chunks → put them in the prompt → answer | Large, private, frequently changing knowledge |
| **Fine-tuning** | The model's *weights* | Train the pre-trained model further on your examples | Deep customisation, strict style or format, domain language |

!!! example "Everyday analogy"
    You're getting a new employee to answer customer questions.

    - **Prompting:** you give them a sticky note: "Be polite, answer in 3 bullets."
    - **RAG:** you give them access to the company handbook and let them look things up for each question.
    - **Fine-tuning:** you send them on a training course until the company way of answering becomes a habit.

### The decision matrix (learn this table)

| Dimension | Prompt engineering | RAG | Fine-tuning |
|---|---|---|---|
| Primary goal | Directing behaviour, framing the task | Accessing external, up-to-date facts | Adapting style, format and core behaviour |
| External knowledge | Very low (limited by the context window) | **Extremely high** | Low to medium |
| Style / format control | Medium | Low | **Extremely high** |
| Cost to set up | **Minimal** | Medium | **High** (clean data and GPU time) |
| Latency impact | Low | Medium (adds a retrieval step) | **Very low** (shorter prompts) |

How to read it: **RAG wins on knowledge, fine-tuning wins on style and format, and prompting wins on cost and speed of setup.**

### The rules of thumb from the Q&A

- **Order:** prompting → RAG → fine-tuning. Move to the next step only when the current one fails.
- **Missing or changing knowledge → RAG.** Policies, prices, product docs, anything that updates.
- **Stable behaviour, format or a narrow repeated task → fine-tuning.**
- **"RAG = temporary knowledge; fine-tuning = permanent behaviour change."**
- **Combine them.** Fine-tune for behaviour and use RAG for facts. Production systems often use all three.
- **An HR-policy chatbot** typically uses RAG plus prompting, *not* fine-tuning, because policies change often.
- **Strict JSON output?** First try structured outputs, a JSON schema or constrained decoding. Fine-tune only if the model keeps failing. Keep validation code either way, because fine-tuning can't guarantee valid output.
- **Chain-of-thought prompting vs fine-tuning:** CoT guides the model only in *that* conversation. Fine-tuning makes the behaviour stick without the instructions.
- **Is it worth fine-tuning?** Build a test set and measure the base model first. If it's already good, don't fine-tune. Otherwise compare all three approaches on accuracy, consistency, latency and cost.

### Two classification traps the TAs clarified

| Situation | What it is |
|---|---|
| Pasting a whole document straight into the prompt | **Prompt engineering** with in-context information. It's *not* RAG, because there's no retrieval step choosing what to include |
| Copilot Studio with a knowledge base, prompts and tool flows | **Prompting + RAG + agentic orchestration**. It's *not* fine-tuning, because no weights are retrained |

Also: RAG doesn't *require* a vector database. Keyword search, SQL or rules can do the retrieval too. And combining RAG with fine-tuning does **not** increase the context window.

---

## Concept 3 — The taxonomy: two independent ways to describe fine-tuning

The slides split fine-tuning approaches into two groups. The professor stressed that they **answer different questions**.

| Axis | Question it answers | Options |
|---|---|---|
| **Task / style adaptation** | *What* signal are we learning from? | Unsupervised FT · Supervised FT · Safety/Alignment FT |
| **Parameter-update strategy** | *How many* weights do we change? | Full FT · Parameter-efficient FT (PEFT) |

Because the axes are independent, you can mix them freely: unsupervised + full, supervised + PEFT (for example SFT with LoRA), alignment + PEFT, and so on.

!!! example "Everyday analogy"
    Describing a meal by *cuisine* (Indian, Italian) and by *portion size* (full plate, half plate). "Half-plate Italian" makes perfect sense. In the same way, "supervised" is the cuisine and "full" or "PEFT" is the portion size.

### The three kinds by *learning signal*

| Kind | Data | What it teaches | Example |
|---|---|---|---|
| **Unsupervised (continued pre-training)** | Raw, unlabelled domain text, with the *same objective as pre-training* (predict the next or masked token) | Domain vocabulary, jargon and style. Cheapest data: wikis, filings, papers | Feed a model thousands of legal contracts |
| **Supervised (SFT)** | Human-written (input → output) **demonstrations** | Behaviour and format: following instructions, tone, JSON, task skills | InstructGPT's SFT stage |
| **Safety / Alignment** | **Feedback** on outputs: rankings, (chosen, rejected) pairs, or self-critique against principles | Safe, preferred, human-aligned answers | RLHF with PPO, DPO, Constitutional-AI-style revision (Chapter 05) |

For a brand-new domain, a common recipe is to **first** do unsupervised continued pre-training on domain text, **then** supervised fine-tuning on labelled examples.

### The two kinds by *how many weights change*

| Kind | What changes | Pros | Cons |
|---|---|---|---|
| **Full fine-tuning** | **Every** parameter | Maximum capacity to adapt; usually the strongest results with enough data and compute | Huge GPU memory; one full model copy per task; **catastrophic forgetting** of general skills |
| **PEFT** | Only a small set of (new or chosen) parameters | Far less memory, tiny per-task files (MBs), less forgetting, easy to swap per task | Slightly less capacity |

**PEFT has three families** (taught in Chapters 03 and 04):

- **Additive:** add *new* small trainable parts and train only those. Examples: adapters, soft prompts.
- **Selective:** pick a *subset of existing* parameters to train. Example: only the bias terms.
- **Re-parameterization:** express the weight *change* with small low-rank matrices. Examples: **LoRA, QLoRA**. This is the most popular family today.

!!! warning "Three families or four?"
    The PEFT lecture (Chapter 03) splits **soft prompts** out of "additive" into their own family. That makes four: additive, soft prompt, selective, re-parameterization. Both views appear in the course. If an option lists soft prompts separately, that's the four-family view, not a mistake. The other three families mean the same thing in both views.

### Common misunderstandings

- **"Unsupervised FT and full FT are alternatives."** No. They sit on *different axes*. Unsupervised FT can be done as full FT or as PEFT.
- **"Multi-task fine-tuning is a type of additive PEFT."** No. "Multi-task" describes the *training data* (several tasks). "Additive" describes *which parameters* are trained.

---

## Concept 4 — Supervised Fine-Tuning (SFT)

### The idea

Take a pre-trained model and train it on **input → output pairs** written by humans (or a stronger model). The model learns to imitate the target answers token by token.

!!! example "Everyday analogy (from the slides)"
    You hire someone who already speaks excellent English: they read millions of books in "pre-training". Then you hand them your company manual:

    - "What are your business hours?" → "We're open from 9 AM to 6 PM."
    - "How do I reset my password?" → "Click 'Forgot Password' on the login page."

    After thousands of such examples, they answer *your* way.

### Key facts

- It's the simplest regime: **one task, one dataset, one specialist model**.
- The main danger on small datasets is **overfitting**. Use few epochs, early stopping and weight decay (Chapter 02).
- **SFT teaches behaviour and format, not new facts.** In the demo, the fine-tuned model answered fluently but still invented facts.

---

## Concept 5 — Instruction Fine-Tuning (IFT)

### The idea

IFT is a **special kind of SFT** where each input is written as a **natural-language instruction**, across **many different tasks**. The goal isn't one skill. It's the general habit of *following instructions*, including ones the model has never seen.

The data looks like this:

```json
{"instruction": "Translate the sentence to French.",
 "input": "Good morning",
 "output": "Bonjour"}
```

### Before vs after

| Prompt | Before IFT (base model just *continues text*) | After IFT (model *follows the instruction*) |
|---|---|---|
| "Translate to French: Good morning" | "French is a Romance language spoken by millions…" | "Bonjour" |
| "Write a Python function to compute factorial" | "Python is a programming language. A factorial is…" | Working `def factorial(n): …` code |

### SFT vs IFT (this table is on the slides)

| SFT | IFT |
|---|---|
| General training on labelled input–output pairs | A specialised form of SFT using **instruction–response** pairs |
| May train on **one task** (for example, sentiment) | Trains on **many tasks**, each phrased as an instruction |
| Input can be plain text | Input includes an instruction (and often extra context) |
| Goal: better **task performance** | Goal: better **instruction-following** |

**Facts to remember:**

- ChatGPT-style and Claude-style chat models are instruction-tuned.
- Success is measured on **held-out (unseen) instructions**.
- Re-doing IFT on a modern chat model is only worth it for specialised behaviour it doesn't already follow reliably.

!!! example "Everyday analogy"
    SFT is training a cashier to do one job really well. IFT is training a personal assistant who can take *any* request ("book a table", "draft a leave letter", "summarise this") and act on it.

---

## Concept 6 — Multi-task fine-tuning

### The idea

Fine-tune **one model on several related tasks at the same time**, instead of one model per task.

| Input | Task | Output |
|---|---|---|
| "I love this movie." | Sentiment | Positive |
| "Who discovered gravity?" | Question answering | Isaac Newton |
| A long article | Summarisation | A short summary |
| "Google is in California." | Named entities | Google → Organisation, California → Location |

### How it works (from the notebook)

- **One shared body** (for example, a DistilBERT encoder) learns a representation that's useful for all tasks.
- **One small "head" per task** turns that shared representation into the task's answer.
- **Each task uses its own loss:** cross-entropy for classification, mean-squared error for a regression task like similarity scores.
- Training alternates between tasks (round-robin), and each batch contains one task.

!!! example "Everyday analogy"
    A student taking physics, chemistry and maths together. The shared foundation (algebra, logic) helps all three, but each subject has its own exam.

**Watch out:** the tasks should share useful knowledge. There is no fixed rule for how to mix them.

---

## Concept 7 — Curriculum fine-tuning: easy first, hard later

### The idea

Instead of showing training examples in random order, show them in a **meaningful order, usually easy → hard**, like a school syllabus that teaches arithmetic before calculus.

It needs two ingredients:

1. **A difficulty measure** ("what counts as hard?"). This can be simple rules (length, rare words), the model's own loss or perplexity, or human labels.
2. **A schedule** ("when do harder examples come in?"). Fixed stages or adaptive ones.

### Why it helps (four benefits on the slides)

- **Faster convergence:** clean, easy examples give clear signals early on.
- **More stable training:** fewer loss spikes.
- **Better generalisation.**
- **Better use of limited data and compute.**

### Scheduling strategies (recognise the names)

| Strategy | What it does |
|---|---|
| **Baby Steps** | Train on the easiest bucket, then *add* the next one while keeping the earlier ones. Robust against forgetting |
| **One-Pass** | Each bucket once, in order, and *throw away* earlier ones. Cheap, but risks forgetting |
| **Self-Paced** | The *model* decides what's easy (low loss), and the threshold rises over time |
| **Competence-Based** | A pacing function grows over time; sample from the easiest fraction available so far |
| **Anti-curriculum (hard-first)** | Start with hard examples. Occasionally useful when easy data adds little signal |

Enterprise uses mentioned: fraud detection, medical diagnosis, industrial fault detection and atmospheric science (learn from clean conditions first, then extreme cases).

!!! example "Everyday analogy"
    Learning to drive: first an empty car park, then quiet streets, then highway traffic in the rain.

---

## Concept 8 — Encoder, decoder, or both? (quick recap)

Students still asked about this, so here is the one table you need:

| Type | Sees | Trained by | Good at | Examples |
|---|---|---|---|---|
| **Encoder-only** | Whole sentence, both directions | Masked-word prediction (fill the [MASK]) | **Understanding:** classification, NER, embeddings | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | Only earlier tokens (causal mask) | Next-token prediction | **Generation:** chat, code, agents | GPT, Claude, LLaMA, Mistral |
| **Encoder-decoder** | Encoder: all input. Decoder: earlier outputs plus cross-attention to the encoder | Denoising / paired data | **Input → output transformation:** translation, summarisation | T5, BART |

**A common confusion:** BERT's "[MASK]" means hiding a word and predicting it from *both* sides. A decoder's "causal mask" means blocking *future* tokens. Same word, different idea.

---

## Concept 9 — Why full fine-tuning is costly, and three ways to make it cheaper

### The cost

Full fine-tuning updates **every** weight. For every parameter, GPU memory must hold the weight, its gradient and the Adam optimizer's extra states, plus activations.

- **Optimizer states and gradients take more memory than the weights themselves.**
- The slides use a **10-billion-parameter model** as the example, needing **over 100 GB**, spread across about seven 24 GB GPUs.

!!! warning "Inconsistent numbers on the slides"
    One slide says **120 GB** for this 10B example (about 12 bytes per parameter). Another says Adam needs **"roughly 16 bytes per parameter"** (which would be 160 GB). Both are common ways of counting, depending on whether a 32-bit master copy of the weights is included. If a question quotes a number, it will most likely state its own assumption. What you must know is the *idea*: **memory ≫ weights alone, because of gradients and optimizer states.**

The slides' bottom line: full FT is very expensive in time and memory, **but adapts best**. It also carries the highest risk of **catastrophic forgetting** (Chapter 02).

### Three ways to fine-tune the whole model without training it all at once

Shared principle: **frozen layers need no gradients and no optimizer states**, so if fewer layers train at a time, the memory per step is smaller. Over the full schedule, the whole network still gets tuned.

| Strategy | What happens in each phase | Key feature |
|---|---|---|
| **Layer-wise** | Unfreeze **exactly one** layer (or small group), train it, freeze it again, move to the next | Only one layer has optimizer state at any time |
| **Block-wise** | Split the network into **contiguous blocks** (for example, 32 layers → 4 blocks of 8); train one block per phase | Like layer-wise, in bigger chunks |
| **Progressive (gradual unfreezing)** | Start with **only the top layer(s)**; each phase **unfreezes more, deeper layers, and the earlier ones stay trainable** | Unfreezing is *cumulative*. Introduced in ULMFiT |

!!! example "Everyday analogy"
    Renovating a 6-floor building.

    - **Layer-wise:** renovate one floor at a time, then lock it.
    - **Block-wise:** renovate two floors at a time.
    - **Progressive:** start with the top floor, then keep adding the floor below *while still working on the ones already open*.

**The difference most likely to be tested:** layer-wise **re-freezes** the previous layer, while progressive **keeps it unfrozen**.

---

## Decision table: which approach when?

| Situation | Choose | Why |
|---|---|---|
| The model can do it, but needs clearer direction | **Prompt engineering** | Zero cost, instant |
| Answers need private, current or frequently changing facts | **RAG** | Fresh knowledge, cheap updates, traceable sources |
| Need a consistent style, strict format or narrow repeated task | **Supervised fine-tuning** | Bakes the behaviour into the weights |
| Model must follow many kinds of instructions | **Instruction fine-tuning** | Generalises to unseen instructions |
| Model doesn't know the domain's language at all | **Unsupervised continued pre-training**, then SFT | Raw text is cheap and teaches vocabulary |
| Several related tasks, one model to serve | **Multi-task fine-tuning** | Shared representation, one deployment |
| Limited GPU memory | **PEFT (LoRA/QLoRA)** | Train a tiny fraction of parameters |
| Lots of data and compute, big domain shift | **Full fine-tuning** | Maximum adaptation |
| Outputs must be safe and match human preferences | **Alignment fine-tuning** (RLHF/DPO) | Learns from feedback (Chapter 05) |

---

## What the hands-on showed

Three notebooks, all using small models so they run on Colab:

- **SFT demo.** Qwen2.5-0.5B on the Dolly-15k instruction dataset (a 200-example subset).
    - **Loss masking:** the prompt tokens are given the label `-100`, so the loss counts **only the assistant's answer tokens**. The model learns to *answer*, not to repeat the question.
    - Used `apply_chat_template` to format conversations in the model's own chat format.
    - Result: training loss kept falling while validation loss **rose in epoch 3**, which is a textbook case of **overfitting**. The TA said this was a workflow demo, not a tuned model.
- **IFT demo.** The same set-up, but each user turn = instruction + optional context, with a lower learning rate. The validation loss curve was healthier. Lesson: **IFT is SFT with instruction-formatted data.**
- **Multi-task demo.** DistilBERT with a shared encoder and four heads (sentiment, news topic, grammar acceptability, similarity score). Cross-entropy for the three classification heads, MSE for the similarity regression head, and tasks interleaved round-robin.

---

## Where students got confused (from the live Q&A)

| Student question | The clarified answer |
|---|---|
| "When should I use RAG instead of fine-tuning?" | RAG when knowledge is missing, private or changing. FT for stable style, format or behaviour. Try prompting first. |
| "If I put the whole document in the prompt, is that RAG?" | No. It's prompt engineering with in-context information. RAG needs a retrieval step. |
| "Can I fine-tune GPT or Claude myself?" | Not their weights directly. Some providers offer a fine-tuning API (watch data exposure). Open-weight models (Llama, Qwen, Mistral, Gemma) can be fine-tuned locally if the licence allows. |
| "Is Ollama a fine-tuning tool?" | No. It's for running and serving models (inference). |
| "Isn't unsupervised FT the opposite of full FT?" | No. Different axes: *learning signal* vs *how many weights change*. |
| "Is multi-task FT a form of additive PEFT?" | No. Multi-task describes the data; additive describes which parameters train. |
| "Is fine-tuning the same as retraining?" | Full FT is a type of retraining that starts from pre-trained weights. Retraining in general can start from random weights. |
| "In full FT, which parts change?" | Everything: embeddings, attention, feed-forward, norms and the output head, unless you freeze them explicitly. |
| "My training loss is exactly 0. Great?" | Suspicious. Likely overfitting, data leakage or a too-easy task. |
| "Does combining RAG with fine-tuning increase the context window?" | No. |
| "To teach the model a new syntax, is it supervised or unsupervised?" | It was debated in class. Self-supervised data teaches the pattern; supervised data teaches when and how to use it. Expect any question on this to be worded carefully. |

---

## Traps to watch for in MCQs

- ❌ "Fine-tuning is the best way to keep the model updated with daily-changing prices." → That's a **RAG** job. Fine-tuned knowledge is frozen at training time.
- ❌ "RAG gives the strongest control over output style and format." → **Fine-tuning** does. RAG is weak on style control.
- ❌ "Fine-tuning adds the most latency." → It adds the **least**: shorter prompts and no retrieval step. RAG adds a retrieval step.
- ❌ "IFT and SFT are unrelated." → IFT **is** a type of SFT, using instruction-formatted, multi-task data.
- ❌ "Full fine-tuning and supervised fine-tuning are alternatives." → Different axes. SFT can be done as full FT or with PEFT.
- ❌ "Layer-wise and progressive fine-tuning are the same." → Progressive keeps earlier-unfrozen layers trainable; layer-wise re-freezes them.
- ❌ "Curriculum learning means training on the hardest examples to toughen the model." → The standard curriculum is **easy → hard**. Hard-first is the *anti*-curriculum variant.
- ❌ "Pasting a document into the prompt = RAG." → Only if a **retrieval step** chose the content.

---

## Recap

1. **Three pathways.** Prompting changes instructions, RAG changes what the model sees, fine-tuning changes the weights. Try them in that order.
2. **RAG for knowledge, fine-tuning for behaviour.** RAG = temporary, updatable knowledge; FT = permanent behaviour change. Combine them in production.
3. **Two independent axes.** Learning signal (unsupervised / supervised / alignment) × how many weights change (full / PEFT, where PEFT is additive, selective or re-parameterization).
4. **Full FT flavours.** SFT (one task) → IFT (many tasks phrased as instructions) → multi-task (shared body, one head per task) → curriculum (easy → hard).
5. **Full FT is expensive and forgets.** Layer-wise, block-wise and progressive unfreezing cut memory per step; progressive keeps layers unfrozen cumulatively.

---

## Practice questions

Pick an option to see instantly whether you're right, with an explanation for every option.

<div class="practice-quiz" data-quiz="ch01">
  <p><em>Loading practice questions…</em></p>
</div>
