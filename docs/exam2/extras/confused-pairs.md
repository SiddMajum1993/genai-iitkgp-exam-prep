# Confused pairs

**How to use this page.** Many wrong options in an MCQ are the *other half* of a pair that sounds alike: prefix tuning instead of prompt tuning, Store instead of checkpointer, MCP instead of A2A. Each pair below has a side-by-side contrast and a one-line **tell**: the word or clue in a question that tells you which side is meant. Read the tell first, then check you can explain the table. If a pair still feels fuzzy, open the chapter linked in its heading.

---

## Part A · Fine-tuning

### 1. RAG vs fine-tuning ([Ch 01](../learn/01-ft-decisions.md))

| RAG | Fine-tuning |
|---|---|
| Adds **knowledge** at run time | Changes **behaviour** in the weights |
| Easy to update: just change the documents | Knowledge is frozen at training time |
| Sources can be cited | Shorter prompts, lower latency |

**The tell:** "changing / private / current facts" → RAG. "Style, format, tone, narrow task" → fine-tuning.

### 2. SFT vs instruction fine-tuning ([Ch 01](../learn/01-ft-decisions.md))

| Supervised fine-tuning (SFT) | Instruction fine-tuning (IFT) |
|---|---|
| Labelled input → output pairs for a task | A *kind* of SFT: many tasks, each phrased as an instruction |
| Goal: do this task well | Goal: follow instructions in general, even unseen ones |

**The tell:** "follow many kinds of instructions" → IFT. IFT is not a separate category from SFT; it sits inside it.

### 3. The two taxonomy axes ([Ch 01](../learn/01-ft-decisions.md))

| Learning-signal axis | Weights-changed axis |
|---|---|
| Unsupervised / supervised / alignment | Full fine-tuning / PEFT |
| *What* the model learns from | *How much* of the model changes |

**The tell:** they're independent. You can do supervised **and** PEFT (e.g. SFT with LoRA), or alignment **and** full fine-tuning.

### 4. Overfitting vs catastrophic forgetting ([Ch 02](../learn/02-ft-data-training.md))

| Overfitting | Catastrophic forgetting |
|---|---|
| Memorises the **training set** | Loses **general skills** it had before |
| Visible: train loss ↓, validation loss ↑ | **Silent**: the target-task metrics look fine |
| Fix: early stopping, regularisation, more varied data | Fix: low LR, replay, L2-SP / EWC, weight averaging, freezing, PEFT |

**The tell:** "worse on *other / general* tasks" → forgetting. "Worse on *unseen examples of the same task*" → overfitting.

### 5. Loss spike vs divergence ([Ch 02](../learn/02-ft-data-training.md))

| Spike | Divergence |
|---|---|
| A short jump that **recovers by itself** | Loss climbs and **doesn't come back** (can become NaN) |
| Often a bad batch | Usually a learning rate too high or a precision problem |

**The tell:** "recovers" → spike, keep watching. "Keeps rising / NaN" → divergence: lower the LR, roll back to a checkpoint, use bf16.

### 6. Gradient clipping vs lowering the learning rate ([Ch 02](../learn/02-ft-data-training.md))

| Gradient clipping | Lower learning rate |
|---|---|
| Caps the size of one update | Makes every update smaller |
| A **seatbelt** for rare spikes | The **fix** when clipping fires all the time |

**The tell:** "clipping triggers on most steps" → the LR is too high.

### 7. Cosine vs WSD schedule ([Ch 02](../learn/02-ft-data-training.md))

| Cosine (with warm-up) | WSD (warm-up, stable, decay) |
|---|---|
| Smooth decay planned over a **known** run length | Flat plateau, then a decay you can start **any time** |
| The common default | For unknown run length or continual training |

**The tell:** "we don't know how long we'll train" → WSD.

### 8. Batch size vs gradient accumulation ([Ch 02](../learn/02-ft-data-training.md))

| Bigger batch | Gradient accumulation |
|---|---|
| Needs more GPU memory at once | Same *effective* batch, built from small micro-batches |
| One forward / backward pass per update | Several passes, then one update |

**The tell:** "the batch doesn't fit in memory, but we want the same effective batch" → accumulation.

### 9. Accuracy vs recall / precision on imbalanced data ([Ch 02](../learn/02-ft-data-training.md))

| Accuracy | Recall / precision / F1 / PR-AUC |
|---|---|
| Can be 99% by always predicting "not fraud" | Show whether the rare class is actually caught |

**The tell:** "rare class" + "high accuracy" → the accuracy paradox.

### 10. Additive vs selective vs re-parameterization PEFT ([Ch 03](../learn/03-peft-adapters-prompts.md))

| Additive | Selective | Re-parameterization |
|---|---|---|
| **Adds** new modules (adapters) or vectors (soft prompts) | Trains a **subset of existing** weights (BitFit = biases) | Expresses the update in a **low-rank form** (LoRA) |

**The tell:** "adds nothing new" → selective. "Low-rank B·A" → re-parameterization. Ch 03 lists soft prompts as a fourth family of their own; see the note at the bottom of this page.

### 11. Houlsby vs Lin adapters ([Ch 03](../learn/03-peft-adapters-prompts.md))

| Houlsby (sequential) | Lin / parallel (residual) |
|---|---|
| Two adapters per block, **in series** (after attention and after FFN) | Adapter runs **beside** the sub-layer; outputs are merged |
| Adds sequential steps, so inference latency | Little extra delay |

**The tell:** "sequential", "in series", "adds latency" → Houlsby.

### 12. Prefix tuning vs prompt tuning ([Ch 03](../learn/03-peft-adapters-prompts.md))

| Prefix tuning | Prompt tuning |
|---|---|
| Learned vectors added to the **K and V of every layer** | Learned "virtual tokens" at the **input layer only** |
| More parameters, more control | Fewest parameters, simplest |

**The tell:** "every layer" / "keys and values" → prefix. "Input embeddings only" → prompt tuning.

### 13. Soft prompt vs discrete (hard) prompt ([Ch 03](../learn/03-peft-adapters-prompts.md))

| Soft prompt | Discrete / system prompt |
|---|---|
| Learned **vectors**, trained by gradient descent | Ordinary **words** that you write |
| Not human-readable | Human-readable, no training |

**The tell:** "trained", "virtual tokens", "embeddings" → soft prompt.

### 14. SMoP vs APT vs IDPG ([Ch 03](../learn/03-peft-adapters-prompts.md))

| SMoP | APT | IDPG |
|---|---|---|
| Many **short prompts + a router** | **Adaptive prefix length** per layer | **Generates a prompt per input** |
| Fixes: one long, costly prompt | Fixes: same length at every layer | Fixes: one prompt for very different inputs |

**The tell:** "router" → SMoP. "Per layer" → APT. "Per input / instance" → IDPG.

### 15. Adapters vs LoRA at inference ([Ch 03](../learn/03-peft-adapters-prompts.md), [04](../learn/04-lora-qlora.md))

| Adapters | LoRA |
|---|---|
| Stay in the network after training | B·A can be **merged into W₀** |
| Add latency | **No extra latency** once merged |

**The tell:** "no additional inference latency" → merged LoRA.

### 16. Merge vs hot-swap ([Ch 04](../learn/04-lora-qlora.md))

| Merge | Hot-swap |
|---|---|
| Fold B·A into W₀, giving one model | Keep one base + many small adapter files |
| Fastest for **one** task | Best for **many** clients or tasks |

**The tell:** "200 clients, one base model" → hot-swap.

### 17. LoRA vs QLoRA ([Ch 04](../learn/04-lora-qlora.md))

| LoRA | QLoRA |
|---|---|
| Frozen base in 16-bit | Frozen base in **4-bit NF4** |
| Trains A and B | Still trains A and B **in BF16** |
| — | Adds double quantization, paged optimizer, gradient checkpointing |

**The tell:** "limited GPU memory, very large model" → QLoRA. Trap: the A and B matrices are **not** 4-bit.

### 18. Rank r vs alpha α ([Ch 04](../learn/04-lora-qlora.md))

| r (rank) | α (alpha) |
|---|---|
| How **big** the side path is (capacity, parameter count) | How **loud** the side path is (the scale is α / r) |

**The tell:** "number of trainable parameters" → r. "Strength of the update" → α.

### 19. Double quantization vs paged optimizer ([Ch 04](../learn/04-lora-qlora.md))

| Double quantization | Paged optimizer |
|---|---|
| Quantizes the **scaling constants** too | Moves optimizer states to **CPU RAM** during memory spikes |
| Saves steady memory | Prevents **OOM crashes** |

**The tell:** "spikes / crashes on long batches" → paged optimizer.

### 20. RLHF vs DPO ([Ch 05](../learn/05-rlhf-dpo.md))

| RLHF (PPO) | DPO |
|---|---|
| 3 stages: SFT → reward model → RL | One supervised-style loss on preference pairs |
| **4 models**: policy, reference, reward, value | **2 models**: policy + frozen reference |
| Online: samples new outputs | Offline: fixed (prompt, chosen, rejected) data |

**The tell:** "no reward model, no RL loop" → DPO. Trap: DPO **still keeps a frozen reference**.

### 21. Reward model vs value model ([Ch 05](../learn/05-rlhf-dpo.md))

| Reward model | Value model (critic) |
|---|---|
| Scores a **finished** answer, as a human would | Predicts the **expected future reward** from a partial answer |
| Trained on human comparisons (Bradley–Terry) | Trained during PPO; used to compute the **advantage** |

**The tell:** "baseline", "advantage", "expected return" → value model.

### 22. KL penalty (RLHF) vs β (DPO) ([Ch 05](../learn/05-rlhf-dpo.md))

| KL penalty | β |
|---|---|
| Pulls the PPO policy back towards the reference | Controls how far DPO may move from the reference |

**The tell:** both are "leashes" to the reference model. Higher = stay closer.

### 23. Alignment vs guardrails vs red-teaming ([Ch 06](../learn/06-revision-slm-guardrails.md))

| Alignment | Guardrails | Red-teaming |
|---|---|---|
| Shapes the model's **preferences** in training | **Checks input and output** at run time | **Attacks** the system to find gaps |
| Not a security boundary | The enforcement layer | Measures safety (and over-blocking) |

**The tell:** "at run time, in code" → guardrails. "Deliberately trying to break it" → red-teaming.

---

## Part B · Vision-language models & multimodal RAG

### 24. CLIP vs LLaVA ([Ch 07](../learn/07-vlm-architectures.md))

| CLIP | LLaVA |
|---|---|
| Two encoders, one shared space | Vision encoder + projector + LLM |
| **Scores** image–text similarity | **Generates** text about an image |
| Cannot write a caption | Can chat about the image |

**The tell:** "classify / search / tag" → CLIP. "Describe / answer / chat" → LLaVA.

### 25. LLaVA stage 1 vs stage 2 ([Ch 07](../learn/07-vlm-architectures.md))

| Stage 1 (alignment) | Stage 2 (instruction tuning) |
|---|---|
| Trains **only the projector** | Trains **projector + LLM** |
| Image–caption pairs | Visual instruction data |

**The tell:** the vision encoder is **frozen in both stages**.

### 26. LLaVA projector vs Qwen-VL adapter ([Ch 07](../learn/07-vlm-architectures.md))

| LLaVA projector | Qwen-VL adapter |
|---|---|
| Maps **every** patch into a token | Cross-attention with **256 learned queries** |
| Token count grows with the image | **Fixed** 256 tokens, positions kept |

**The tell:** "fixed number of visual tokens" → Qwen-VL.

### 27. Zero-shot vs linear probe ([Ch 07](../learn/07-vlm-architectures.md))

| Zero-shot (CLIP) | Linear probe |
|---|---|
| Compare the image with "a photo of a {label}" | Train a small classifier on frozen features |
| No labelled training images | Needs labelled examples |

**The tell:** "no training examples for the new labels" → zero-shot.

### 28. Qwen-VL grounding vs SAM segmentation ([Ch 07](../learn/07-vlm-architectures.md), [08](../learn/08-vlm-applications.md))

| Qwen-VL grounding | SAM |
|---|---|
| A **box** as text tokens (`<ref>`, `<box>`) | An exact **pixel mask** |
| Knows *what* the object is | Knows *where* the edges are, not *what* it is |

**The tell:** "exact outline / pixels" → SAM. "Point to the object named X" → grounding.

### 29. Hallucination vs presupposition hallucination ([Ch 08](../learn/08-vlm-applications.md))

| Hallucination | Presupposition hallucination |
|---|---|
| Invents details that aren't in the image | Accepts a **false premise in the question** ("What colour is the cat?" with no cat) |

**The tell:** the question assumes something that isn't there → presupposition. The fix is to allow "not present".

### 30. CIDEr vs ANLS vs CLIPScore ([Ch 08](../learn/08-vlm-applications.md))

| CIDEr | ANLS | CLIPScore |
|---|---|---|
| Captions vs **reference captions** | Document VQA answers, scored by text similarity | Caption vs the **image itself**, no reference needed |

**The tell:** "DocVQA" → ANLS. "No reference caption" → CLIPScore.

### 31. Keyword (BM25) vs semantic (dense) search ([Ch 09](../learn/09-multimodal-rag.md))

| BM25 | Dense / semantic |
|---|---|
| Exact word matches | Meaning matches |
| Great for codes, IDs, names | Great for paraphrases |

**The tell:** "exact error code" → BM25. Both needed → hybrid.

### 32. Bi-encoder vs reranker (cross-encoder) ([Ch 09](../learn/09-multimodal-rag.md))

| Bi-encoder | Reranker |
|---|---|
| Embeds query and documents **separately** | Reads query + document **together** |
| Fast; used for first-stage retrieval | Slower but more precise; re-orders the top results |

**The tell:** "re-order the top 50" → reranker.

### 33. Parse-then-embed vs ColPali ([Ch 09](../learn/09-multimodal-rag.md))

| Parse-then-embed | ColPali |
|---|---|
| OCR / parse → text chunks → embed | Embed the **page image** directly |
| Breaks on charts and complex layouts | Keeps visual information; ~1,000 patch vectors per page |
| Small index, cheap | Bigger index, GPU cost |

**The tell:** "charts, scans, complex layouts" → ColPali. "Clean tables, tight budget" → parser.

### 34. Single-vector vs late interaction ([Ch 09](../learn/09-multimodal-rag.md))

| Single vector | Late interaction (MaxSim) |
|---|---|
| One embedding per page; one dot product | Many vectors per page; each query token takes its **best match**, then the scores are **summed** |

**The tell:** "max over patches, then sum" → late interaction.

### 35. Recall@K vs nDCG@5 ([Ch 09](../learn/09-multimodal-rag.md))

| Recall@K | nDCG@5 |
|---|---|
| Did the right item appear in the top K at all? | Were the right items ranked **near the top**? |

**The tell:** "order matters" → nDCG.

---

## Part C · Agents

### 36. Workflow vs agent ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md))

| Workflow | Agent |
|---|---|
| **You** fix the steps in advance | The **model** decides the next step at run time |
| Predictable, cheap, easy to audit | Flexible, costlier, harder to test |

**The tell:** "you can draw the flowchart in advance" → workflow.

### 37. Function calling vs MCP ([Ch 10](../learn/10-rag-to-agents.md))

| Plain function calling | MCP |
|---|---|
| Tool schemas written into **your** app | Tools **discovered at run time** from a server (`list_tools()`, `call_tool()`) |
| Fine for a few fixed tools | Best for many tools that change, shared across apps |

**The tell:** "the server owns the schema" → MCP. In both, the model only **emits** a request and **your code executes** it.

### 38. MCP vs A2A ([Ch 11](../learn/11-multi-agent.md))

| MCP | A2A |
|---|---|
| Agent ↔ **tools and data** | Agent ↔ **agent** |
| Tools, resources | Agent Cards, Tasks |

**The tell:** "delegate to another company's agent" → A2A.

### 39. ReAct vs Plan-and-Execute vs Reflection ([Ch 10](../learn/10-rag-to-agents.md))

| ReAct | Plan-and-Execute | Reflection |
|---|---|---|
| Think → act → observe, one step at a time | Whole plan first, then execute; re-plan the failed tail | Draft → critique → revise |
| Cheapest, easy to debug | Predictable cost, parallel steps | Highest quality, **most tokens** |

**The tell:** "critique its own output" → Reflection. "Plan up front" → Plan-and-Execute.

### 40. Conversation state vs task state vs long-term memory ([Ch 10](../learn/10-rag-to-agents.md))

| Conversation | Task state | Long-term |
|---|---|---|
| What was said in **this chat** | Progress in **this run** (held fares, budget left) | Facts kept **across sessions** (home airport) |

**The tell:** "next week, in a new chat" → long-term memory.

### 41. Supervisor vs peer-to-peer ([Ch 11](../learn/11-multi-agent.md))

| Supervisor | Peer-to-peer |
|---|---|
| One coordinator routes the work | Agents talk directly |
| Clear accountability and tracing; **single point of failure** | Flexible, fewer hops; harder to govern |

**The tell:** "approvals, spending money, audit" → supervisor. "Debate, writer–reviewer" → P2P.

### 42. Contract vs handoff ([Ch 11](../learn/11-multi-agent.md))

| Contract | Handoff |
|---|---|
| The agent's **job description**: role, inputs, outputs, constraints, permissions | The **message** passed between agents |
| Defined once | Sent every time; minimum sufficient context |

**The tell:** "what an agent may do" → contract. "What gets passed along" → handoff.

### 43. Round limit vs deadlock check ([Ch 11](../learn/11-multi-agent.md))

| Round limit / token budget | Deadlock check |
|---|---|
| Hard stop after N rounds or N tokens | Spots the **same conflict repeating** and stops early |

**The tell:** "agents keep undoing each other's changes" → deadlock check, then escalate to a human.

### 44. Checkpointer vs Store ([Ch 12](../learn/12-langchain-langgraph.md))

| Checkpointer | Store |
|---|---|
| Saves state per **thread** (`thread_id`) after every super-step | Long-term memory **across threads** |
| Enables resume, human-in-the-loop, time travel | User preferences, facts to remember |
| e.g. `SqliteSaver` | e.g. `SqliteStore` |

**The tell:** "remember for a new conversation" → Store. "Resume this run" → checkpointer.

### 45. Retry vs hand back to the model ([Ch 12](../learn/12-langchain-langgraph.md))

| Retry | Hand back |
|---|---|
| **Infrastructure** failures: timeout, 503 | **Meaning** failures: "no flights", "invalid date" |
| Bounded retries, then fallback | Return the error as an observation so the model can re-plan |

**The tell:** retrying "no flights found" won't help → hand back.

### 46. `recursion_limit` vs oscillation guard ([Ch 12](../learn/12-langchain-langgraph.md))

| `recursion_limit` | Oscillation guard |
|---|---|
| A cap on the number of **super-steps** | Detects **A → B → A** ping-pong |

**The tell:** `recursion_limit` counts super-steps, **not** tool calls or LLM calls.

### 47. Interrupt vs fallback ([Ch 12](../learn/12-langchain-langgraph.md))

| `interrupt()` | `with_fallbacks` |
|---|---|
| **Pauses** for a human; resume with `Command(resume=...)`; the node **re-runs from the top** | Switches to a **backup** model or tool when the first fails |

**The tell:** "approval before booking" → interrupt.

### 48. LangChain vs LangGraph ([Ch 12](../learn/12-langchain-langgraph.md))

| LangChain | LangGraph |
|---|---|
| The **toolbox**: models, tools, prompts, `create_agent` | The **flowchart**: nodes, edges, state, checkpoints |

**The tell:** they aren't rivals. `create_agent` is itself built on LangGraph.

---

## Part D · Responsible AI

### 49. Responsible AI vs trustworthy AI ([Ch 13](../learn/13-responsible-ai.md))

| Responsible AI | Trustworthy AI |
|---|---|
| The **promise**: principles, intentions | The **proof**: evidence that it works |

**The tell:** "show evidence" → trustworthy.

### 50. Provider vs deployer ([Ch 13](../learn/13-responsible-ai.md))

| Provider | Deployer |
|---|---|
| Builds the model or tool | Puts it in front of customers or staff |
| — | **Answers for the outcome** (Air Canada) |

**The tell:** "the vendor's model made the mistake" → the deployer is still accountable.

### 51. Instruction vs permission; alert vs cap ([Ch 13](../learn/13-responsible-ai.md))

| Instruction / alert | Permission / cap |
|---|---|
| Asks the model or a person to behave | **Enforced** by the system |
| Can be ignored or missed | Can't be bypassed |

**The tell:** "an instruction is not a control; an alert is not a cap".

### 52. Data Principal vs Data Fiduciary (DPDP) ([Ch 13](../learn/13-responsible-ai.md))

| Data Principal | Data Fiduciary |
|---|---|
| The **person** the data is about | The **organisation** deciding why and how data is processed |

**The tell:** the fiduciary carries the duty, even when staff paste data into a third-party tool (Samsung-style cases).

---

!!! note "Where the chapters disagree"
    **How many PEFT families?** Chapter 01 previews **three** (additive, selective, re-parameterization) and counts soft prompts as additive. Chapter 03 lists **four**, giving soft prompts their own family. Both framings appear in the course; if an option lists four families including "soft prompt", that matches Chapter 03.
