# Cheat sheet: the last-hour read

**How to use this page.** This is a one-page squeeze of all 13 chapters, for the last hour before the exam. Each chapter has a few bullets with the **key terms in bold**, then its top traps (the wrong statements that keep appearing as MCQ options). If a bullet doesn't make sense, it's a sign to re-read that chapter, not to memorise the bullet. Use the [focus map](focus-map.md) to decide what to read first.

---

## The formulas in one place

Full details and worked examples are on [Numbers & formulas](formulas.md).

- **Full fine-tuning memory:** about **12 bytes per parameter** (2 weight + 2 gradient + 8 Adam), or 16 with an FP32 master copy. 10B → 120 GB (or 160 GB). 70B → 840 GB.
- **Storing a model:** FP16 = 2 bytes per parameter (7B → 14 GB), NF4 = 0.5 bytes (7B → 3.5 GB).
- **LoRA and QLoRA on 70B:** LoRA ≈ 17.6 bits per parameter → ≈ 154 GB. QLoRA ≈ 5.2 bits → ≈ 46 GB, one GPU.
- **LoRA parameters:** r × (d + k) per matrix, and L × M × r × (d + k) for the model. LLaMA-13B with r = 16 on Q, K, V, O → 26.2M → 52.4 MB (0.2%).
- **LoRA scaling:** h = W₀x + (α/r)·BAx. α = 16, r = 8 → ×2.
- **Adapter parameters:** 2md + m + d. d = 768, m = 64 → 99,136 ≈ 0.1M ≈ 1% of a 9.5M layer.
- **Prompt tuning:** t × d. **Prefix tuning:** about 2 × L × t × d.
- **Absmax (int8):** c = 127 ÷ max|X|, q = round(c·x), x̂ = q ÷ c. [0.5, −2.0, 1.2, 3.0] → [21, −85, 51, 127].
- **Block overhead:** 32 ÷ block size bits per parameter (64 → 0.5, 128 → 0.25).
- **Training steps:** steps per epoch = examples ÷ effective batch. Warm-up = warmup_ratio × total steps (1,000 × 0.03 = 30).
- **Effective batch:** micro × accumulation × GPUs (2 × 16 × 4 = 128). **Linear scaling:** LR × B_new ÷ B_old (32 at 2e-5 → 64 at 4e-5).
- **Clipping:** g × min(1, c ÷ ‖g‖), with c = 1.0. Same direction, smaller size.
- **Reward-model loss:** −log σ(r_chosen − r_rejected). An equal score gives 0.69.
- **RLHF reward:** r_φ − β·KL. **Advantage:** A = Q − V. **PPO models:** 4; **DPO models:** 2.
- **ViT patches:** (H ÷ P)². 224/16 → 196, 224/14 → 256, 336/14 → 576.
- **CLIP:** an N × N score grid, with N − 1 negatives per image. Lower τ gives sharper choices.
- **MaxSim:** for each query token, take its best patch score, then add them up.

---

## Part A · Fine-tuning

### 01 · When to fine-tune ([chapter](../learn/01-ft-decisions.md))

- **Three pathways, in this order:** **prompting** (change the instructions) → **RAG** (change what the model sees) → **fine-tuning** (change the weights).
- **RAG for knowledge, fine-tuning for behaviour.** RAG gives fresh, updatable, citable facts. Fine-tuning gives style, format and narrow-task skill, with **the lowest latency**. Production often combines them.
- **Two independent axes:** learning signal (**unsupervised / supervised / alignment**) × how many weights change (**full / PEFT**).
- **IFT is a kind of SFT**: many tasks phrased as instructions. **Multi-task** = shared body, one head per task. **Curriculum** = easy → hard.
- **Progressive unfreezing** keeps earlier-unfrozen layers trainable; **layer-wise** re-freezes them.

**Top traps:** fine-tuning for daily-changing prices (→ RAG) · "RAG controls style best" (→ fine-tuning) · "pasting a document into the prompt is RAG" (only if a retrieval step chose it).

### 02 · Data & training mechanics ([chapter](../learn/02-ft-data-training.md))

- **Quality beats quantity** (**LIMA**: ~1,000 curated examples). Pipeline: source → clean → **dedupe** → filter → review → format; match the **chat template** exactly.
- **Data traps:** imbalance (the **accuracy paradox**: use recall, precision, F1, PR-AUC), **leakage** (split first; time and group splits), **domain shift**.
- **Learning rate is the key knob:** small for fine-tuning; **warm-up**, then decay. **Cosine** is the default; **WSD** for unknown run length.
- **Loss curves:** both fall = healthy; validation turns up = **overfitting**; both high = underfitting; both flat = no learning.
- **Stability:** a **spike** recovers, **divergence** doesn't. **Gradient clipping** is a seatbelt; prefer **bf16**. **Gradient accumulation** gives a big effective batch in small memory.
- **Catastrophic forgetting is silent.** Defend with a low LR, few epochs, **replay**, **L2-SP / EWC**, **weight averaging (WiSE-FT)**, freezing, PEFT.

**Top traps:** "frequent clipping fixes a high LR" (→ lower the LR) · "frozen layers block gradients" (gradients still flow through) · "fp16 for stability" (→ bf16).

### 03 · PEFT: adapters & soft prompts ([chapter](../learn/03-peft-adapters-prompts.md))

- **PEFT** freezes the backbone and trains a tiny part (often under 1%). **Four families** in this chapter: **additive** (adapters), **soft prompt**, **re-parameterization** (LoRA), **selective** (BitFit).
- **Adapter** = down-project → non-linearity → up-project + **residual**, starting **near identity**.
- **Houlsby** adapters are **sequential** (in series, adding latency); **Lin** adapters are **parallel / residual**.
- **Prompt tuning** = learned virtual tokens at the **input only**. **Prefix tuning** = learned vectors in the **K and V of every layer** (W_k and W_v stay frozen).
- **SMoP** (short prompts + router), **APT** (adaptive prefix length per layer; bottom layers need more), **IDPG** (a prompt per input).

**Top traps:** "adapters modify the original weights" (they're new modules) · "prompt tuning works at every layer" (→ prefix tuning) · "BitFit adds a module" (it's selective).

### 04 · LoRA & QLoRA ([chapter](../learn/04-lora-qlora.md))

- **Intrinsic dimension:** adapting to a task needs only a small change, so a **low-rank** update is enough.
- **LoRA:** freeze **W₀**, train **ΔW = B·A**, with **B = 0** at the start, scaled by **α / r**.
- **Main memory saving:** frozen weights need **no gradients and no Adam states**.
- **Merge** B·A into W₀ for zero extra latency; keep adapters separate to **hot-swap** many tenants on one base.
- **QLoRA = LoRA + NF4 4-bit base + double quantization + paged optimizer** (+ gradient checkpointing). **Only the frozen base is 4-bit**; A and B stay **BF16**.

**Top traps:** "QLoRA trains in 4-bit" (it *stores* the base in 4-bit) · "double quantization quantizes weights twice" (it quantizes the scaling constants) · "the paged optimizer is faster" (it prevents OOM crashes) · "higher r is always better".

### 05 · Alignment: RLHF & DPO ([chapter](../learn/05-rlhf-dpo.md))

- **HHH** (Helpful, Honest, Harmless): the three often conflict.
- **RLHF = SFT → reward model** (trained on human **comparisons**, Bradley–Terry) **→ RL with PPO**, plus a **KL penalty** against a frozen reference.
- **Reward hacking:** the policy exploits flaws in the reward model; KL keeps it close to the reference.
- **PPO juggles 4 models:** policy, reference, reward, value (critic). The value model gives the **advantage** ("better than average?").
- **DPO** learns directly from (prompt, chosen, rejected): **no reward model, no RL**, just policy + **frozen reference**; **β** sets the pull.
- **DPO** for a fixed offline dataset and simplicity; **PPO** for online exploration or a complex, changing reward.

**Top traps:** "DPO drops the reference model" (it keeps it) · "the reward model writes answers" (it only scores) · "larger β lets it drift further" (it stays closer).

### 06 · Revision, SLMs & guardrails ([chapter](../learn/06-revision-slm-guardrails.md))

- **Revision is a signal:** LoRA, QLoRA, prompt vs RAG vs fine-tune and RLHF / DPO were all re-taught.
- **SLMs** (hundreds of millions to a few billion parameters) win for **narrow, high-volume, private or on-device** tasks with **good data**.
- **5 steps to fine-tune an SLM:** choose → prepare data → configure (QLoRA + LoRA) → run and monitor → evaluate against the base.
- **Guardrails** check **input and output** (content filters, **PII redaction**), because **alignment isn't a security boundary**.
- **Red-teaming loop:** attack → find gap → fix → re-test; also check for **over-blocking**.

**Top traps:** "aligned models don't need guardrails" · "regex catches all PII" (use NER tools like Presidio) · "passing red-team tests proves safety".

---

## Part B · Vision-language models & multimodal RAG

### 07 · VLM architectures ([chapter](../learn/07-vlm-architectures.md))

- **One template:** ViT "eyes" → connector → output. The models differ in the connector and the output.
- **CLIP:** two encoders, one shared space, **contrastive** training (negatives come free from the batch). **Zero-shot** via "a photo of a {label}". **Lower temperature = sharper.** It **can't generate text**.
- **LLaVA:** frozen CLIP encoder + **projector** + LLM. **Stage 1 trains only the projector; Stage 2 trains projector + LLM.** The encoder is **always frozen**.
- **Qwen-VL:** cross-attention adapter with **256 learned queries** (fixed count, 2D positions kept), `<ref>` / `<box>` grounding as **text tokens**, 3 training stages.
- **SAM:** heavy encoder runs **once**, light decoder **per prompt**, **3 candidate masks** (scored by IoU), built with a **data engine** (SA-1B). Knows *where*, not *what*.

**Top traps:** "CLIP captions images" · "LLaVA uses cross-attention" (→ projector) · "Qwen-VL's token count grows with resolution" (fixed at 256) · "SAM re-runs its encoder per click".

### 08 · Applications of VLMs ([chapter](../learn/08-vlm-applications.md))

- **Two decisions:** which model fits the task, and which prompting technique fits the task.
- **Captioning:** a sentence → generative VLM; a tag → **CLIP**. Use scoped, length-limited, **fixed templates**.
- **VQA:** regions or text → Qwen-VL; **chain-of-thought** for counting and comparison; fix **presupposition hallucination** by allowing "not present".
- **Documents:** show the **exact JSON schema**, allow null, keep a **human in the loop**; **resolution** is the biggest lever.
- **Reasoning comes from the LLM backbone.** For charts, **extract the values first, then reason**. **VLM finds + SAM cuts.**
- Metrics to recognise: **CIDEr** (captions), **ANLS** (DocVQA), **CLIPScore** (no reference needed).

**Top traps:** "a generative VLM to tag 10M photos" (→ CLIP) · "vary the prompt for creativity" (consistency wins) · "the model leaves missing fields blank" (it may invent them).

### 09 · Multimodal RAG ([chapter](../learn/09-multimodal-rag.md))

- **Same pattern everywhere:** embed → nearest-neighbour search (**ANN**) → generate. Only *what's embedded* changes.
- **Search basics:** **BM25** (exact words) vs **dense** (meaning) vs **hybrid**; **bi-encoder** for fast retrieval, **reranker** for precision; **hard negatives** improve precision.
- **Parse-then-embed** (OCR → text) is brittle on charts and tables. **ColPali** embeds the **page image** as ~1,000 patch vectors with **late interaction / MaxSim**.
- **Trade-off:** ColPali is more accurate on visual pages but needs a **bigger index and more GPU memory**. Pick by document type and budget; **hybrid** in production. **Graph RAG** for multi-hop, **table RAG** for exact numbers.
- **Retrieval and generation fail independently.** Grounding and citations **reduce** hallucination, not eliminate it. Metrics: **Recall@K**, **nDCG@5**, **ViDoRe**.

**Top traps:** "ANN is more accurate" (it's faster) · "ColPali uses OCR" (no OCR) · "ColPali has a smaller index" (much larger).

---

## Part C · Agents

### 10 · From RAG to agents ([chapter](../learn/10-rag-to-agents.md))

- **The ladder:** LLM → RAG (knowledge) → assistant (context) → tool-using assistant (actions) → **agent** (decisions) → multi-agent (delegation). "RAG answers. Agents finish."
- **Workflow vs agent** = who picks the next step. If you can draw the flowchart in advance, build a **workflow**.
- **Tool calling:** the model **emits** a request; **your code executes** it. **MCP** standardises discovery (`list_tools`) and use (`call_tool`); MCP = agent ↔ tools.
- **Agent loop:** Think → Act → Observe → Remember; **bounded, observable, recoverable**; **verify success in code**.
- **Loop shapes:** **ReAct** (cheapest), **Plan-and-Execute** (predictable), **Reflection** (most tokens). **Memory:** conversation, **task state**, long-term.

**Top traps:** "an agent = a more powerful LLM" · "MCP is for agent-to-agent" (→ A2A) · "a failed tool call should stop the agent" (it's an observation).

### 11 · Multi-agent systems ([chapter](../learn/11-multi-agent.md))

- **Split only for:** specialisation, parallelism, context isolation, verification.
- **Contract** (role, inputs, outputs, constraints, permissions); **handoff** = a small structured message with **minimum sufficient context**.
- **Coordination cycle:** decompose → execute → attribute → reconcile → re-plan. **Bound it:** round limit, token budget, **deadlock check**, **escalate** to a human.
- **Supervisor** for accountability (but a **single point of failure**); **peer-to-peer** for debate. **Parallel** when independent, **sequential** when dependent.
- **Safety:** **least privilege**, human approval for writes, inter-agent messages are **untrusted**, **fail closed**, log every handoff. **MCP = agent ↔ tools; A2A = agent ↔ agent** (Agent Cards, Tasks).

**Top traps:** "more agents = more reliable" · "send the full history to every agent" · "merge invalid results now, fix later".

### 12 · LangChain & LangGraph ([chapter](../learn/12-langchain-langgraph.md))

- **LangChain = toolbox; LangGraph = flowchart.** `create_agent` **is** a LangGraph graph.
- **Nodes return partial updates; reducers merge them.** Overwrite is the default; `Annotated[..., add]` appends. Parallel writes need a reducer.
- **Checkpointer** (per **`thread_id`**) = resume, human-in-the-loop, **time travel**. **Store** = long-term memory **across threads**.
- **`interrupt()`** pauses; **`Command(resume=...)`** continues, but **the node re-runs from the top**.
- **`recursion_limit`** counts **super-steps**, not tool calls.
- **Reliability:** retry infrastructure failures; hand meaning failures back to the model; **`with_fallbacks`** and degrade *visibly*; **idempotency key** derived from thread + intent.

**Top traps:** "the checkpointer remembers preferences across chats" (→ Store) · "a new UUID per attempt" (the key must be derived) · "retry everything".

---

## Part D · Responsible AI

### 13 · Responsible & trusted AI ([chapter](../learn/13-responsible-ai.md))

- **Responsible = promise; trustworthy = proof.** Answer the **five leader questions** with evidence.
- **The deployer is accountable** for its AI's output, even when the vendor's model erred (**Air Canada**).
- **Five controls:** **guardrails** (permission, not instruction; re-test after changes; DPD), **privacy** (Samsung), **hallucination management** (draft + proportional checks; fake citations), **cost control** (cap, not alert; named owner), **system documentation** (one-page record; NYC MyCity).
- **Agentic shift:** AI that *does*, not just *says*. Know which systems can write, modify, approve or spend.
- **Frameworks in one line each:** **NIST AI RMF** (Govern, Map, Measure, Manage), **EU AI Act** (4 risk tiers), **ISO/IEC 42001** (AI management system), **DPDP** (Data Principal vs Data Fiduciary), **India AI Governance Guidelines** (seven sutras).

**Top traps:** "a 'never do X' line in the prompt is a guardrail" · "a spending alert controls cost" · "a disclaimer makes a wrong bot acceptable".
