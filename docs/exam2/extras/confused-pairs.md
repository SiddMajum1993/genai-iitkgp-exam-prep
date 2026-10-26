# Confused pairs

**How to use this page.** Many wrong options in an MCQ are the *other half* of a pair that sounds alike: prefix tuning instead of prompt tuning, Store instead of checkpointer, MCP instead of A2A. Each pair below explains both sides in a short paragraph, then gives a one-line **tell**: the word or clue in a question that tells you which side is meant. Read the tell first, then check that you can explain the difference in your own words. If a pair still feels fuzzy, open the chapter linked in its heading.

---

## Part A · Fine-tuning

### 1. RAG vs fine-tuning ([Ch 01](../learn/01-ft-decisions.md))

**RAG** adds **knowledge** at run time. It's easy to update, because you only change the documents, and the answer can cite its sources.

**Fine-tuning** changes **behaviour** by changing the weights. Whatever knowledge it learns is frozen at training time, but you get shorter prompts and lower latency.

**The tell:** "changing / private / current facts" → RAG. "Style, format, tone, narrow task" → fine-tuning.

### 2. SFT vs instruction fine-tuning ([Ch 01](../learn/01-ft-decisions.md))

**Supervised fine-tuning (SFT)** trains on labelled input → output pairs for a task. The goal is to do *this* task well.

**Instruction fine-tuning (IFT)** is a *kind* of SFT. The data covers many tasks, each phrased as an instruction, and the goal is to follow instructions in general, even ones the model has never seen.

**The tell:** "follow many kinds of instructions" → IFT. IFT is not a separate category from SFT; it sits inside it.

### 3. The two taxonomy axes ([Ch 01](../learn/01-ft-decisions.md))

**The learning-signal axis** asks *what* the model learns from: unsupervised, supervised or alignment data.

**The weights-changed axis** asks *how much* of the model changes: full fine-tuning or PEFT.

**The tell:** they're independent. You can do supervised **and** PEFT (e.g. SFT with LoRA), or alignment **and** full fine-tuning.

### 4. Overfitting vs catastrophic forgetting ([Ch 02](../learn/02-ft-data-training.md))

**Overfitting** is when the model memorises the **training set**. It's visible on the loss curves: training loss keeps going down while validation loss goes up. You fix it with early stopping, regularisation or more varied data.

**Catastrophic forgetting** is when the model loses **general skills** it had before. It's **silent**: the target-task metrics look fine. You fix it with a low learning rate, replay, L2-SP / EWC, weight averaging, freezing or PEFT.

**The tell:** "worse on *other / general* tasks" → forgetting. "Worse on *unseen examples of the same task*" → overfitting.

### 5. Loss spike vs divergence ([Ch 02](../learn/02-ft-data-training.md))

**A loss spike** is a short jump in the loss that **recovers by itself**. It's often caused by one bad batch.

**Divergence** is when the loss climbs and **doesn't come back**, and it can become NaN. It's usually caused by a learning rate that's too high or a precision problem.

**The tell:** "recovers" → spike, keep watching. "Keeps rising / NaN" → divergence: lower the LR, roll back to a checkpoint, use bf16.

### 6. Gradient clipping vs lowering the learning rate ([Ch 02](../learn/02-ft-data-training.md))

**Gradient clipping** caps the size of a single update. Think of it as a **seatbelt** for rare spikes.

**Lowering the learning rate** makes every update smaller. It's the real **fix** when clipping is firing all the time.

**The tell:** "clipping triggers on most steps" → the LR is too high.

### 7. Cosine vs WSD schedule ([Ch 02](../learn/02-ft-data-training.md))

**Cosine (with warm-up)** is a smooth decay planned over a **known** run length. It's the common default.

**WSD (warm-up, stable, decay)** keeps the learning rate on a flat plateau and starts the decay whenever you choose, **at any time**. It suits an unknown run length or continual training.

**The tell:** "we don't know how long we'll train" → WSD.

### 8. Batch size vs gradient accumulation ([Ch 02](../learn/02-ft-data-training.md))

**A bigger batch** needs more GPU memory at once. There's one forward and backward pass per update.

**Gradient accumulation** gives the same *effective* batch, built from several small micro-batches. It runs several passes, then makes one update.

**The tell:** "the batch doesn't fit in memory, but we want the same effective batch" → accumulation.

### 9. Accuracy vs recall / precision on imbalanced data ([Ch 02](../learn/02-ft-data-training.md))

**Accuracy** can look excellent on imbalanced data while meaning nothing. A fraud model can score 99% just by always predicting "not fraud".

**Recall, precision, F1 and PR-AUC** show whether the rare class is actually being caught.

**The tell:** "rare class" + "high accuracy" → the accuracy paradox.

### 10. Additive vs selective vs re-parameterization PEFT ([Ch 03](../learn/03-peft-adapters-prompts.md))

**Additive PEFT** **adds** something new and trains only that: new modules (adapters) or new vectors (soft prompts).

**Selective PEFT** trains a **subset of the existing** weights and adds nothing. BitFit, for example, trains only the biases.

**Re-parameterization PEFT** expresses the weight update in a **low-rank form**. LoRA is the main example.

**The tell:** "adds nothing new" → selective. "Low-rank B·A" → re-parameterization. Ch 03 lists soft prompts as a fourth family of their own; see the note at the bottom of this page.

### 11. Houlsby vs Lin adapters ([Ch 03](../learn/03-peft-adapters-prompts.md))

**Houlsby adapters (sequential)** put two adapters in each block, **in series**: one after attention and one after the feed-forward layer. Because data must pass through them one after another, they add inference latency.

**Lin adapters (parallel / residual)** run **beside** the sub-layer, and the two outputs are merged. They add little extra delay.

**The tell:** "sequential", "in series", "adds latency" → Houlsby.

### 12. Prefix tuning vs prompt tuning ([Ch 03](../learn/03-peft-adapters-prompts.md))

**Prefix tuning** adds learned vectors to the **keys and values of every layer**. That means more parameters, but also more control.

**Prompt tuning** adds learned "virtual tokens" at the **input layer only**. It has the fewest parameters and is the simplest.

**The tell:** "every layer" / "keys and values" → prefix. "Input embeddings only" → prompt tuning.

### 13. Soft prompt vs discrete (hard) prompt ([Ch 03](../learn/03-peft-adapters-prompts.md))

**A soft prompt** is a set of learned **vectors**, trained by gradient descent. Humans can't read it.

**A discrete (hard) prompt**, such as a system prompt, is made of ordinary **words** that you write. It's human-readable and needs no training.

**The tell:** "trained", "virtual tokens", "embeddings" → soft prompt.

### 14. SMoP vs APT vs IDPG ([Ch 03](../learn/03-peft-adapters-prompts.md))

**SMoP** uses many **short prompts plus a router** that picks between them. It fixes the problem of one long, costly prompt.

**APT** uses an **adaptive prefix length** for each layer. It fixes the problem of using the same length at every layer.

**IDPG** **generates a prompt for each input**. It fixes the problem of using one prompt for very different inputs.

**The tell:** "router" → SMoP. "Per layer" → APT. "Per input / instance" → IDPG.

### 15. Adapters vs LoRA at inference ([Ch 03](../learn/03-peft-adapters-prompts.md), [04](../learn/04-lora-qlora.md))

**Adapters** stay in the network after training, so every request passes through them and they add latency.

**LoRA** can be **merged into W₀** after training (W₀ + B·A), so there's **no extra latency** once it's merged.

**The tell:** "no additional inference latency" → merged LoRA.

### 16. Merge vs hot-swap ([Ch 04](../learn/04-lora-qlora.md))

**Merging** folds B·A into W₀, leaving you with one ordinary model. It's the fastest option for **one** task.

**Hot-swapping** keeps one base model plus many small adapter files, and plugs in the right one per request. It's best for **many** clients or tasks.

**The tell:** "200 clients, one base model" → hot-swap.

### 17. LoRA vs QLoRA ([Ch 04](../learn/04-lora-qlora.md))

**LoRA** keeps the frozen base in 16-bit and trains the A and B matrices.

**QLoRA** keeps the frozen base in **4-bit NF4**, but still trains A and B **in BF16**. On top of that it adds double quantization, the paged optimizer and gradient checkpointing.

**The tell:** "limited GPU memory, very large model" → QLoRA. Trap: the A and B matrices are **not** 4-bit.

### 18. Rank r vs alpha α ([Ch 04](../learn/04-lora-qlora.md))

**The rank r** sets how **big** the side path is, which means its capacity and its parameter count.

**The alpha α** sets how **loud** the side path is. The update is scaled by α / r.

**The tell:** "number of trainable parameters" → r. "Strength of the update" → α.

### 19. Double quantization vs paged optimizer ([Ch 04](../learn/04-lora-qlora.md))

**Double quantization** quantizes the **scaling constants** too. It gives a steady memory saving.

**The paged optimizer** moves optimizer states to **CPU RAM** during memory spikes. It prevents **out-of-memory crashes**.

**The tell:** "spikes / crashes on long batches" → paged optimizer.

### 20. RLHF vs DPO ([Ch 05](../learn/05-rlhf-dpo.md))

**RLHF (with PPO)** runs in three stages: SFT, then a reward model, then reinforcement learning. It needs **four models**: the policy, the reference, the reward model and the value model. It's online, meaning it samples new outputs as it trains.

**DPO** replaces all that with one supervised-style loss on preference pairs. It needs only **two models**: the policy and a frozen reference. It's offline, learning from fixed (prompt, chosen, rejected) data.

**The tell:** "no reward model, no RL loop" → DPO. Trap: DPO **still keeps a frozen reference**.

### 21. Reward model vs value model ([Ch 05](../learn/05-rlhf-dpo.md))

**The reward model** scores a **finished** answer, the way a human would. It's trained on human comparisons (Bradley–Terry).

**The value model (critic)** predicts the **expected future reward** from a partial answer. It's trained during PPO and used to compute the **advantage**.

**The tell:** "baseline", "advantage", "expected return" → value model.

### 22. KL penalty (RLHF) vs β (DPO) ([Ch 05](../learn/05-rlhf-dpo.md))

**The KL penalty** in RLHF pulls the PPO policy back towards the reference model.

**The β** in DPO controls how far the model may move away from the reference.

**The tell:** both are "leashes" to the reference model. Higher = stay closer.

### 23. Alignment vs guardrails vs red-teaming ([Ch 06](../learn/06-revision-slm-guardrails.md))

**Alignment** shapes the model's **preferences** during training. It is not a security boundary.

**Guardrails** **check the input and the output** at run time. They're the enforcement layer.

**Red-teaming** deliberately **attacks** the system to find gaps. It measures safety, and over-blocking too.

**The tell:** "at run time, in code" → guardrails. "Deliberately trying to break it" → red-teaming.

---

## Part B · Vision-language models & multimodal RAG

### 24. CLIP vs LLaVA ([Ch 07](../learn/07-vlm-architectures.md))

**CLIP** has two encoders that map images and text into one shared space. It **scores** how well an image and a text match, but it cannot write a caption.

**LLaVA** connects a vision encoder to an LLM through a projector. It **generates** text about an image and can chat about it.

**The tell:** "classify / search / tag" → CLIP. "Describe / answer / chat" → LLaVA.

### 25. LLaVA stage 1 vs stage 2 ([Ch 07](../learn/07-vlm-architectures.md))

**Stage 1 (alignment)** trains **only the projector**, using image–caption pairs.

**Stage 2 (instruction tuning)** trains **the projector and the LLM**, using visual instruction data.

**The tell:** the vision encoder is **frozen in both stages**.

### 26. LLaVA projector vs Qwen-VL adapter ([Ch 07](../learn/07-vlm-architectures.md))

**The LLaVA projector** turns **every** image patch into a token, so the token count grows with the image.

**The Qwen-VL adapter** uses cross-attention with **256 learned queries**, so the image always becomes a **fixed** 256 tokens, with positions kept.

**The tell:** "fixed number of visual tokens" → Qwen-VL.

### 27. Zero-shot vs linear probe ([Ch 07](../learn/07-vlm-architectures.md))

**Zero-shot classification (CLIP)** compares the image with text such as "a photo of a {label}". It needs no labelled training images.

**A linear probe** trains a small classifier on top of frozen features. It needs labelled examples.

**The tell:** "no training examples for the new labels" → zero-shot.

### 28. Qwen-VL grounding vs SAM segmentation ([Ch 07](../learn/07-vlm-architectures.md), [08](../learn/08-vlm-applications.md))

**Qwen-VL grounding** outputs a **box** as text tokens (`<ref>`, `<box>`). It knows *what* the object is.

**SAM** outputs an exact **pixel mask**. It knows *where* the edges are, but not *what* the object is.

**The tell:** "exact outline / pixels" → SAM. "Point to the object named X" → grounding.

### 29. Hallucination vs presupposition hallucination ([Ch 08](../learn/08-vlm-applications.md))

**Hallucination** is when the model invents details that aren't in the image.

**Presupposition hallucination** is when the model accepts a **false premise in the question**. For example, it answers "What colour is the cat?" when there's no cat.

**The tell:** the question assumes something that isn't there → presupposition. The fix is to allow "not present".

### 30. CIDEr vs ANLS vs CLIPScore ([Ch 08](../learn/08-vlm-applications.md))

**CIDEr** scores captions against **reference captions** written by people.

**ANLS** scores document VQA answers by text similarity to the correct answer.

**CLIPScore** scores a caption against **the image itself**, so no reference caption is needed.

**The tell:** "DocVQA" → ANLS. "No reference caption" → CLIPScore.

### 31. Keyword (BM25) vs semantic (dense) search ([Ch 09](../learn/09-multimodal-rag.md))

**BM25 (keyword search)** matches exact words. It's great for codes, IDs and names.

**Dense (semantic) search** matches meaning. It's great for paraphrases.

**The tell:** "exact error code" → BM25. Both needed → hybrid.

### 32. Bi-encoder vs reranker (cross-encoder) ([Ch 09](../learn/09-multimodal-rag.md))

**A bi-encoder** embeds the query and the documents **separately**. It's fast, so it's used for first-stage retrieval.

**A reranker (cross-encoder)** reads the query and a document **together**. It's slower but more precise, so it's used to re-order the top results.

**The tell:** "re-order the top 50" → reranker.

### 33. Parse-then-embed vs ColPali ([Ch 09](../learn/09-multimodal-rag.md))

**Parse-then-embed** runs OCR or a parser, turns the page into text chunks and embeds those. The index is small and cheap, but it breaks on charts and complex layouts.

**ColPali** embeds the **page image** directly, keeping the visual information, with about 1,000 patch vectors per page. The index is bigger and it costs GPU time.

**The tell:** "charts, scans, complex layouts" → ColPali. "Clean tables, tight budget" → parser.

### 34. Single-vector vs late interaction ([Ch 09](../learn/09-multimodal-rag.md))

**Single-vector retrieval** stores one embedding per page and scores it with one dot product.

**Late interaction (MaxSim)** stores many vectors per page. Each query token takes its **best match** among them, and those best scores are **summed**.

**The tell:** "max over patches, then sum" → late interaction.

### 35. Recall@K vs nDCG@5 ([Ch 09](../learn/09-multimodal-rag.md))

**Recall@K** asks whether the right item appeared in the top K at all.

**nDCG@5** asks whether the right items were ranked **near the top**.

**The tell:** "order matters" → nDCG.

---

## Part C · Agents

### 36. Workflow vs agent ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md))

**A workflow** is a set of steps that **you** fix in advance. It's predictable, cheap and easy to audit.

**An agent** lets **the model** decide the next step at run time. It's flexible, but costlier and harder to test.

**The tell:** "you can draw the flowchart in advance" → workflow.

### 37. Function calling vs MCP ([Ch 10](../learn/10-rag-to-agents.md))

**Plain function calling** has the tool schemas written into **your** app. It's fine for a few fixed tools.

**MCP** has tools **discovered at run time** from a server, using `list_tools()` and `call_tool()`. It's best for many tools that change and are shared across apps.

**The tell:** "the server owns the schema" → MCP. In both, the model only **emits** a request and **your code executes** it.

### 38. MCP vs A2A ([Ch 11](../learn/11-multi-agent.md))

**MCP** connects an agent to **tools and data**. Its building blocks are tools and resources.

**A2A** connects an agent to **another agent**. Its building blocks are Agent Cards and Tasks.

**The tell:** "delegate to another company's agent" → A2A.

### 39. ReAct vs Plan-and-Execute vs Reflection ([Ch 10](../learn/10-rag-to-agents.md))

**ReAct** works one step at a time: think, act, observe, repeat. It's the cheapest and easy to debug.

**Plan-and-Execute** makes the whole plan first, then carries it out, re-planning only the failed tail. Its cost is predictable, and steps can run in parallel.

**Reflection** drafts an answer, critiques it, then revises. It gives the highest quality but uses the **most tokens**.

**The tell:** "critique its own output" → Reflection. "Plan up front" → Plan-and-Execute.

### 40. Conversation state vs task state vs long-term memory ([Ch 10](../learn/10-rag-to-agents.md))

**Conversation state** is what was said in **this chat**.

**Task state** is the progress in **this run**, such as held fares or budget left.

**Long-term memory** is the facts kept **across sessions**, such as your home airport.

**The tell:** "next week, in a new chat" → long-term memory.

### 41. Supervisor vs peer-to-peer ([Ch 11](../learn/11-multi-agent.md))

**In a supervisor design**, one coordinator routes the work. That gives clear accountability and tracing, but the supervisor is a **single point of failure**.

**In a peer-to-peer design**, agents talk to each other directly. It's flexible and needs fewer hops, but it's harder to govern.

**The tell:** "approvals, spending money, audit" → supervisor. "Debate, writer–reviewer" → P2P.

### 42. Contract vs handoff ([Ch 11](../learn/11-multi-agent.md))

**A contract** is the agent's **job description**: its role, inputs, outputs, constraints and permissions. It's defined once.

**A handoff** is the **message** passed from one agent to another. It's sent every time, and should carry the minimum sufficient context.

**The tell:** "what an agent may do" → contract. "What gets passed along" → handoff.

### 43. Round limit vs deadlock check ([Ch 11](../learn/11-multi-agent.md))

**A round limit or token budget** is a hard stop after N rounds or N tokens.

**A deadlock check** spots the **same conflict repeating** and stops early.

**The tell:** "agents keep undoing each other's changes" → deadlock check, then escalate to a human.

### 44. Checkpointer vs Store ([Ch 12](../learn/12-langchain-langgraph.md))

**The checkpointer** saves the state of one **thread** (identified by `thread_id`) after every super-step. That's what enables resume, human-in-the-loop and time travel. An example is `SqliteSaver`.

**The Store** is long-term memory **across threads**, for things like user preferences and facts to remember. An example is `SqliteStore`.

**The tell:** "remember for a new conversation" → Store. "Resume this run" → checkpointer.

### 45. Retry vs hand back to the model ([Ch 12](../learn/12-langchain-langgraph.md))

**Retrying** is for **infrastructure** failures, such as a timeout or a 503 error. Retry a bounded number of times, then fall back.

**Handing back to the model** is for **meaning** failures, such as "no flights" or "invalid date". Return the error as an observation so the model can re-plan.

**The tell:** retrying "no flights found" won't help → hand back.

### 46. `recursion_limit` vs oscillation guard ([Ch 12](../learn/12-langchain-langgraph.md))

**`recursion_limit`** is a cap on the number of **super-steps** a run may take.

**An oscillation guard** detects **A → B → A** ping-pong between nodes.

**The tell:** `recursion_limit` counts super-steps, **not** tool calls or LLM calls.

### 47. Interrupt vs fallback ([Ch 12](../learn/12-langchain-langgraph.md))

**`interrupt()`** **pauses** the run for a human. You resume with `Command(resume=...)`, and the node **re-runs from the top**.

**`with_fallbacks`** switches to a **backup** model or tool when the first one fails.

**The tell:** "approval before booking" → interrupt.

### 48. LangChain vs LangGraph ([Ch 12](../learn/12-langchain-langgraph.md))

**LangChain** is the **toolbox**: models, tools, prompts and `create_agent`.

**LangGraph** is the **flowchart**: nodes, edges, state and checkpoints.

**The tell:** they aren't rivals. `create_agent` is itself built on LangGraph.

---

## Part D · Responsible AI

### 49. Responsible AI vs trustworthy AI ([Ch 13](../learn/13-responsible-ai.md))

**Responsible AI** is the **promise**: the principles and intentions.

**Trustworthy AI** is the **proof**: evidence that the system actually works as promised.

**The tell:** "show evidence" → trustworthy.

### 50. Provider vs deployer ([Ch 13](../learn/13-responsible-ai.md))

**The provider** builds the model or tool.

**The deployer** puts it in front of customers or staff, and **answers for the outcome**, as the Air Canada case showed.

**The tell:** "the vendor's model made the mistake" → the deployer is still accountable.

### 51. Instruction vs permission; alert vs cap ([Ch 13](../learn/13-responsible-ai.md))

**An instruction or an alert** asks the model or a person to behave. It can be ignored or missed.

**A permission or a cap** is **enforced** by the system. It can't be bypassed.

**The tell:** "an instruction is not a control; an alert is not a cap".

### 52. Data Principal vs Data Fiduciary (DPDP) ([Ch 13](../learn/13-responsible-ai.md))

**The Data Principal** is the **person** the data is about.

**The Data Fiduciary** is the **organisation** that decides why and how the data is processed.

**The tell:** the fiduciary carries the duty, even when staff paste data into a third-party tool (Samsung-style cases).

---

!!! note "Where the chapters disagree"
    **How many PEFT families?** Chapter 01 previews **three** (additive, selective, re-parameterization) and counts soft prompts as additive. Chapter 03 lists **four**, giving soft prompts their own family. Both framings appear in the course; if an option lists four families including "soft prompt", that matches Chapter 03.
