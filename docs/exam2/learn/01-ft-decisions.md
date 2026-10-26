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

### 🔴 High: expect questions here

**Prompt vs RAG vs fine-tuning: which to use for a given scenario.** This got 5 slides, a decision matrix, the pre-read and the post-read, and about 25 Q&A questions, which makes it the most-asked topic. It also suits the 1st exam's "a team wants X" question style.

**The decision matrix: which approach is strong on what.** A single slide packs the whole comparison into one grid, which makes it perfect MCQ material.

**SFT vs Instruction Fine-tuning (IFT).** There's a dedicated comparison slide, two notebooks and several Q&A rows.

**The taxonomy: unsupervised, supervised and alignment vs full and PEFT.** These are two *independent* axes. The professor clarified this himself, and the TAs repeated it.

### 🟠 Medium: likely to show up in one question

**PEFT's three families: additive, selective, re-parameterization.** This is a preview here, taught in depth in Chapters 03 and 04.

**Multi-task fine-tuning: a shared body plus a head per task.** It's on the slides and in a notebook.

**Curriculum fine-tuning: easy → hard, and why it helps.** Six slides, with named strategies.

**Layer-wise vs block-wise vs progressive fine-tuning.** Diagram slides, plus a notebook in the next session.

**Why full fine-tuning is expensive (memory) and why it causes forgetting.** It's on the slides and in the Q&A. The exact numbers are inconsistent (see Concept 9).

### 🟢 Low: understand the idea, skip the details

**Encoder-only vs decoder-only vs encoder-decoder.** This is a recap from the 1st exam, but students still asked about it.

**Med-PaLM and BloombergGPT facts.** These were used as examples only.

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

**Domain expertise.** Fine-tuning helps a model learn the jargon and conventions of a field, such as medical, legal or finance terms.

**Consistent behaviour.** It makes the model always follow a tone, format or policy, which prompting can't *guarantee*. For example, every reply comes out in your company's support style.

**Higher accuracy.** A small fine-tuned model can beat a much bigger general model on one narrow task. Think of a 3B model trained on your own ticket categories.

**Lower cost and latency.** You no longer need long instructions in every prompt, and you can use a smaller model, so inference is faster and cheaper.

### What fine-tuning is *not* good at

Fine-tuning shapes **behaviour and format**. It is an **unreliable way to add lots of new facts**, and those facts become frozen at training time. For facts, use RAG (Concept 2).

### Common misunderstanding

- **"Fine-tuning means training from scratch."** No. Fine-tuning *starts from pre-trained weights* and does a smaller, targeted round of training ("mini training"). The forward pass, the loss and backpropagation are the same as in normal training; only the starting point is different.

---

## Concept 2 — The three pathways: Prompting, RAG, Fine-tuning

There are three ways to get a model to do what you want. Each one changes something different.

**Prompt engineering changes the *instructions*.** You write clear instructions, a role, examples and an output format, then check the output and refine. The model itself stays frozen. It's best for quick prototypes and standard tasks, with zero setup.

**RAG changes what the model *sees*.** Your documents go into a vector DB. At question time, the top-k chunks are retrieved and put into the prompt, and the model answers from them. It's best for large, private or frequently changing knowledge.

**Fine-tuning changes the model's *weights*.** You train the pre-trained model further on your own examples. It's best for deep customisation, strict style or format, and domain language.

!!! example "Everyday analogy"
    You're getting a new employee to answer customer questions.

    - **Prompting:** you give them a sticky note: "Be polite, answer in 3 bullets."
    - **RAG:** you give them access to the company handbook and let them look things up for each question.
    - **Fine-tuning:** you send them on a training course until the company way of answering becomes a habit.

### The decision matrix (learn this one)

The matrix compares the three approaches on five dimensions.

**Primary goal.** Prompt engineering is for directing behaviour and framing the task. RAG is for accessing external, up-to-date facts. Fine-tuning is for adapting style, format and core behaviour.

**External knowledge.** Prompt engineering is very low here, because it's limited by the context window. RAG is **extremely high**. Fine-tuning is low to medium.

**Style and format control.** Prompt engineering is medium and RAG is low. Fine-tuning is **extremely high**.

**Cost to set up.** Prompt engineering is **minimal** and RAG is medium. Fine-tuning is **high**, because it needs clean data and GPU time.

**Latency impact.** Prompt engineering is low. RAG is medium, because it adds a retrieval step. Fine-tuning is **very low**, because prompts get shorter.

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

**Pasting a whole document straight into the prompt is *prompt engineering* with in-context information.** It's *not* RAG, because there's no retrieval step choosing what to include.

**Copilot Studio with a knowledge base, prompts and tool flows is *prompting + RAG + agentic orchestration*.** It's *not* fine-tuning, because no weights are retrained.

Also: RAG doesn't *require* a vector database. Keyword search, SQL or rules can do the retrieval too. And combining RAG with fine-tuning does **not** increase the context window.

---

## Concept 3 — The taxonomy: two independent ways to describe fine-tuning

The slides split fine-tuning approaches into two groups. The professor stressed that they **answer different questions**.

**The task or style axis asks *what* signal we're learning from.** The options are unsupervised FT, supervised FT and safety/alignment FT.

**The parameter-update axis asks *how many* weights we change.** The options are full FT and parameter-efficient FT (PEFT).

Because the axes are independent, you can mix them freely: unsupervised + full, supervised + PEFT (for example SFT with LoRA), alignment + PEFT, and so on.

!!! example "Everyday analogy"
    Describing a meal by *cuisine* (Indian, Italian) and by *portion size* (full plate, half plate). "Half-plate Italian" makes perfect sense. In the same way, "supervised" is the cuisine and "full" or "PEFT" is the portion size.

### The three kinds by *learning signal*

**Unsupervised fine-tuning (continued pre-training) learns from raw, unlabelled domain text.** It uses the *same objective as pre-training*: predict the next or masked token. It teaches domain vocabulary, jargon and style, and its data is the cheapest to get: wikis, filings, papers. For example, you feed a model thousands of legal contracts.

**Supervised fine-tuning (SFT) learns from human-written (input → output) *demonstrations*.** It teaches behaviour and format: following instructions, tone, JSON and task skills. InstructGPT's SFT stage is the classic example.

**Safety or alignment fine-tuning learns from *feedback* on outputs.** The feedback can be rankings, (chosen, rejected) pairs, or self-critique against principles. It teaches safe, preferred, human-aligned answers. Examples are RLHF with PPO, DPO and Constitutional-AI-style revision (Chapter 05).

For a brand-new domain, a common recipe is to **first** do unsupervised continued pre-training on domain text, **then** supervised fine-tuning on labelled examples.

### The two kinds by *how many weights change*

**Full fine-tuning changes *every* parameter.** That gives maximum capacity to adapt, and usually the strongest results when you have enough data and compute. The costs are huge GPU memory, one full model copy per task, and **catastrophic forgetting** of general skills.

**PEFT changes only a small set of new or chosen parameters.** It needs far less memory, produces tiny per-task files (MBs), causes less forgetting and is easy to swap per task. The cost is slightly less capacity.

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

Two prompts show the difference clearly.

**"Translate to French: Good morning."** Before IFT, the base model just *continues the text*: "French is a Romance language spoken by millions…". After IFT, the model *follows the instruction* and replies "Bonjour".

**"Write a Python function to compute factorial."** Before IFT, you get "Python is a programming language. A factorial is…". After IFT, you get working `def factorial(n): …` code.

### SFT vs IFT (straight from the slides)

IFT is a kind of SFT, so the differences are about scope and goal.

**What it trains on.** SFT is general training on labelled input–output pairs. IFT is a specialised form of SFT that uses **instruction–response** pairs.

**How many tasks.** SFT may train on **one task**, for example sentiment. IFT trains on **many tasks**, each phrased as an instruction.

**What the input looks like.** In SFT the input can be plain text. In IFT the input includes an instruction, and often extra context.

**The goal.** SFT aims for better **task performance**. IFT aims for better **instruction-following**.

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

For example, one model might handle all four of these at once:

- "I love this movie." is a **sentiment** task, and the output is *Positive*.
- "Who discovered gravity?" is **question answering**, and the output is *Isaac Newton*.
- A long article is a **summarisation** task, and the output is a short summary.
- "Google is in California." is **named-entity** tagging, and the output is *Google → Organisation, California → Location*.

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

**Baby Steps.** Train on the easiest bucket, then *add* the next one while keeping the earlier ones. It's robust against forgetting.

**One-Pass.** Go through each bucket once, in order, and *throw away* the earlier ones. It's cheap, but it risks forgetting.

**Self-Paced.** The *model* decides what's easy (low loss), and the difficulty threshold rises over time.

**Competence-Based.** A pacing function grows over time, and you sample from the easiest fraction available so far.

**Anti-curriculum (hard-first).** Start with the hard examples. This is occasionally useful when easy data adds little signal.

Enterprise uses mentioned: fraud detection, medical diagnosis, industrial fault detection and atmospheric science (learn from clean conditions first, then extreme cases).

!!! example "Everyday analogy"
    Learning to drive: first an empty car park, then quiet streets, then highway traffic in the rain.

---

## Concept 8 — Encoder, decoder, or both? (quick recap)

Students still asked about this, so here is the short version you need.

**Encoder-only models see the whole sentence in both directions.** They're trained by masked-word prediction (fill in the [MASK]), and they're good at **understanding**: classification, NER and embeddings. Examples are BERT, RoBERTa and DeBERTa.

**Decoder-only models see only earlier tokens**, because of a causal mask. They're trained by next-token prediction, and they're good at **generation**: chat, code and agents. Examples are GPT, Claude, LLaMA and Mistral.

**Encoder-decoder models do both.** The encoder sees all of the input. The decoder sees its earlier outputs plus cross-attention to the encoder. They're trained on denoising or paired data, and they're good at **input → output transformation** such as translation and summarisation. Examples are T5 and BART.

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

**Layer-wise.** Unfreeze **exactly one** layer (or a small group), train it, freeze it again and move to the next. The key feature is that only one layer has optimizer state at any time.

**Block-wise.** Split the network into **contiguous blocks** (for example, 32 layers → 4 blocks of 8) and train one block per phase. It's like layer-wise, in bigger chunks.

**Progressive (gradual unfreezing).** Start with **only the top layer(s)**. Each phase **unfreezes more, deeper layers, and the earlier ones stay trainable**. So the unfreezing is *cumulative*. This was introduced in ULMFiT.

!!! example "Everyday analogy"
    Renovating a 6-floor building.

    - **Layer-wise:** renovate one floor at a time, then lock it.
    - **Block-wise:** renovate two floors at a time.
    - **Progressive:** start with the top floor, then keep adding the floor below *while still working on the ones already open*.

**The difference most likely to be tested:** layer-wise **re-freezes** the previous layer, while progressive **keeps it unfrozen**.

---

## Which approach when?

Here's how to match a situation to an approach.

**If the model can already do it but needs clearer direction, use prompt engineering.** It costs nothing and works instantly.

**If answers need private, current or frequently changing facts, use RAG.** You get fresh knowledge, cheap updates and traceable sources.

**If you need a consistent style, a strict format or a narrow repeated task, use supervised fine-tuning.** It bakes the behaviour into the weights.

**If the model must follow many kinds of instructions, use instruction fine-tuning.** It generalises to instructions it hasn't seen.

**If the model doesn't know the domain's language at all, use unsupervised continued pre-training first, then SFT.** Raw text is cheap and teaches the vocabulary.

**If several related tasks must be served by one model, use multi-task fine-tuning.** You get a shared representation and one deployment.

**If GPU memory is limited, use PEFT (LoRA/QLoRA).** You train only a tiny fraction of the parameters.

**If you have lots of data and compute and a big domain shift, use full fine-tuning.** It gives maximum adaptation.

**If outputs must be safe and match human preferences, use alignment fine-tuning (RLHF/DPO).** It learns from feedback (Chapter 05).

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

These are real questions students asked during the session, with the answers they were given.

**"When should I use RAG instead of fine-tuning?"** Use RAG when knowledge is missing, private or changing. Use FT for stable style, format or behaviour. Try prompting first.

**"If I put the whole document in the prompt, is that RAG?"** No. It's prompt engineering with in-context information. RAG needs a retrieval step.

**"Can I fine-tune GPT or Claude myself?"** Not their weights directly. Some providers offer a fine-tuning API (watch out for data exposure). Open-weight models such as Llama, Qwen, Mistral and Gemma can be fine-tuned locally if the licence allows.

**"Is Ollama a fine-tuning tool?"** No. It's for running and serving models, which is inference.

**"Isn't unsupervised FT the opposite of full FT?"** No. They sit on different axes: *learning signal* vs *how many weights change*.

**"Is multi-task FT a form of additive PEFT?"** No. Multi-task describes the data. Additive describes which parameters train.

**"Is fine-tuning the same as retraining?"** Full FT is a type of retraining that starts from pre-trained weights. Retraining in general can start from random weights.

**"In full FT, which parts change?"** Everything: embeddings, attention, feed-forward, norms and the output head, unless you freeze some of them explicitly.

**"My training loss is exactly 0. Great?"** That's suspicious. It likely means overfitting, data leakage or a task that's too easy.

**"Does combining RAG with fine-tuning increase the context window?"** No.

**"To teach the model a new syntax, is it supervised or unsupervised?"** This was debated in class. Self-supervised data teaches the pattern, and supervised data teaches when and how to use it. Expect any question on this to be worded carefully.

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
