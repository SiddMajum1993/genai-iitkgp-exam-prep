# Focus map: where the questions will likely come from

**How to use this page.** This page pulls together the "Exam radar" tables from all 13 chapters into one ranked list, so you can decide where to spend your time.

- Start with the 🔴 table. Every row is something you should be able to explain in one sentence and recognise inside a scenario.
- Then skim the 🟠 list. Leave the 🟢 items for the end, if you have time.
- The rankings come from how much class time a topic got, how often it appeared in the pre-read, slides and post-read, how many student questions it drew, and whether it was re-taught in the revision session.
- They are **our estimate, not an official blueprint**.

---

## 🔴 High: know these cold

Ranked roughly by how strong the evidence is. Topics that were also **re-taught in the revision session** (Chapter 06) are marked ♻️ and sit near the top.

| # | Topic | Ch | What to know (one line) | Evidence |
|---|---|---|---|---|
| 1 | Prompt vs RAG vs fine-tuning ♻️ | [01](../learn/01-ft-decisions.md) | RAG for missing or changing **knowledge**; fine-tuning for **style, format and behaviour**; prompting first, because it's cheapest | ~25 Q&A rows (most-asked topic), decision matrix, re-taught |
| 2 | How LoRA works ♻️ | [04](../learn/04-lora-qlora.md) | Freeze W₀, train a side path B·A (B starts at 0), scaled by α/r | Pre-read, slides, ~10 Q&A rows, re-taught |
| 3 | Why LoRA saves memory ♻️ | [04](../learn/04-lora-qlora.md) | Frozen weights need **no gradients and no Adam states**; that's the main saving | Slides and Q&A, re-taught |
| 4 | QLoRA's ingredients ♻️ | [04](../learn/04-lora-qlora.md) | NF4 base + double quantization + paged optimizer (+ gradient checkpointing). **Only the frozen base is 4-bit**; A and B stay BF16 | Half a lecture; asked 4 ways in the Q&A |
| 5 | Merge → no extra latency ♻️ | [04](../learn/04-lora-qlora.md) | B·A folds into W₀ after training; adapters (Houlsby) stay in the network and add latency | Pre-read: "the fundamental advantage" |
| 6 | DPO ♻️ | [05](../learn/05-rlhf-dpo.md) | Learns from (prompt, chosen, rejected); **no reward model, no RL**, but **keeps a frozen reference**; β controls the pull | Pre-read, slides, full notebook, re-taught |
| 7 | RLHF's three stages ♻️ | [05](../learn/05-rlhf-dpo.md) | SFT → reward model (from comparisons) → RL with PPO + KL penalty | InstructGPT slide, re-taught |
| 8 | The 4 models in PPO ♻️ | [05](../learn/05-rlhf-dpo.md) | Policy, reference, reward, value. DPO needs just 2 (policy + frozen reference) | Natural "which is NOT" question |
| 9 | Reward hacking + KL penalty ♻️ | [05](../learn/05-rlhf-dpo.md) | The policy exploits flaws in the reward model; KL keeps it close to the reference | Memorable slide example |
| 10 | Workflow vs agent | [10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md) | Who decides the next step? You (workflow) or the model at run time (agent). "If you can draw the flowchart in advance, you don't need an agent" | Two chapters, pre-reads, post-reads, many Q&A threads |
| 11 | Tool calling | [10](../learn/10-rag-to-agents.md) | The model **emits** a request; **your code executes** it. Safety checks live at that boundary | "The model never touches the API" |
| 12 | MCP | [10](../learn/10-rag-to-agents.md) | `list_tools()` / `call_tool()`, discovery at run time, the server owns the schema; MCP = agent ↔ tools | 10+ Q&A questions (most in that session) |
| 13 | CLIP | [07](../learn/07-vlm-architectures.md) | Two encoders, one shared space; contrastive grid; zero-shot via "a photo of a {label}"; **cannot generate text** | Heaviest slide time, 6-question quick check |
| 14 | LLaVA's two stages | [07](../learn/07-vlm-architectures.md) | Stage 1 trains **only the projector**; Stage 2 trains projector + LLM; the vision encoder is **always frozen** | 6-question quick check |
| 15 | Qwen-VL | [07](../learn/07-vlm-architectures.md) | Cross-attention adapter with **256 learned queries** (fixed count, positions kept); boxes are text tokens | Stated learning objective |
| 16 | Parse-then-embed vs ColPali | [09](../learn/09-multimodal-rag.md) | OCR pipelines break on charts and tables; ColPali embeds the **page image** as ~1,000 patch vectors (no OCR) | Lecture quiz, pre-read, post-read, Q&A |
| 17 | Late interaction | [09](../learn/09-multimodal-rag.md) | Each query token finds its best-matching patch, then the scores are added; many vectors per page, not one | Lecture quiz |
| 18 | Checkpointer vs Store | [12](../learn/12-langchain-langgraph.md) | Checkpointer = one thread (`thread_id`): resume, human-in-the-loop, time travel. Store = long-term, across threads | Largest Q&A cluster in that session |
| 19 | Idempotency | [12](../learn/12-langchain-langgraph.md) | A derived key (thread + intent) so a retried booking never happens twice | Slide, Mentimeter quiz, notebook demo |
| 20 | Contract vs handoff | [11](../learn/11-multi-agent.md) | Contract = role, inputs, outputs, constraints, permissions. Handoff = a small structured message with **minimum sufficient context** | Two slides, ~8 Q&A rows |
| 21 | Five essential RAI controls | [13](../learn/13-responsible-ai.md) | Guardrails (permission, not instruction), privacy, hallucination management, cost control (cap, not alert), system documentation | Centrepiece of agenda and pre-read |
| 22 | Catastrophic forgetting | [02](../learn/02-ft-data-training.md), [03](../learn/03-peft-adapters-prompts.md) | It's **silent**: the target task improves while general skills drop. Fixes: low LR, replay, L2-SP/EWC, weight averaging, freezing, PEFT | Biggest Q&A cluster in Ch 02 |
| 23 | Reading loss curves | [02](../learn/02-ft-data-training.md) | Train ↓ + validation ↑ = overfit; both high = underfit; both flat = no learning | "Ultimate guide" section |
| 24 | Learning rate, warm-up, schedules | [02](../learn/02-ft-data-training.md) | Most important knob; small for fine-tuning; warm up, then decay (cosine is the default; WSD for unknown run length) | 14 slides |
| 25 | Training instability | [02](../learn/02-ft-data-training.md) | A spike self-repairs; divergence doesn't; gradient clipping is a seatbelt; prefer bf16 | Stability checklist |
| 26 | Data quality and curation | [02](../learn/02-ft-data-training.md) | Quality beats quantity (LIMA ~1,000 examples); source → clean → dedupe → filter → review → format | Agenda, pre-read, post-read |
| 27 | SFT vs instruction fine-tuning | [01](../learn/01-ft-decisions.md) | IFT is a kind of SFT on many tasks phrased as instructions; the goal is instruction-following | Comparison slide, two notebooks |
| 28 | Two independent taxonomy axes | [01](../learn/01-ft-decisions.md) | Learning signal (unsupervised / supervised / alignment) × how many weights change (full / PEFT) | Professor's own clarification |
| 29 | The PEFT families | [03](../learn/03-peft-adapters-prompts.md) | Additive (adapters), soft prompt, re-parameterization (LoRA), selective (BitFit) | Concept map, revision |
| 30 | Adapter structure | [03](../learn/03-peft-adapters-prompts.md) | Down-project → non-linearity → up-project + residual, starting near identity | "Two requirements" slide |
| 31 | Houlsby vs Lin adapters | [03](../learn/03-peft-adapters-prompts.md) | Sequential (adds latency) vs parallel / residual | Limitations slide |
| 32 | Prefix vs prompt tuning | [03](../learn/03-peft-adapters-prompts.md) | Prompt tuning = input layer only; prefix tuning = K and V of **every** layer | Highest-volume confusion in that Q&A |
| 33 | HHH and its trade-offs | [05](../learn/05-rlhf-dpo.md) | Helpful, Honest, Harmless often pull against each other | Opening slides |
| 34 | SLM vs large model | [06](../learn/06-revision-slm-guardrails.md) | Narrow + high volume + private / on-device + good data → a fine-tuned SLM wins | Pre-read table, post-read rules |
| 35 | Guardrails | [06](../learn/06-revision-slm-guardrails.md) | Check **input and output** (content filters, PII redaction), because alignment isn't a security boundary | Half the hands-on |
| 36 | Choosing the VLM paradigm | [07](../learn/07-vlm-architectures.md), [08](../learn/08-vlm-applications.md) | Tag → CLIP; free text → LLaVA; read, point or documents → Qwen-VL; pixel outline → SAM | The anchor table of both lectures |
| 37 | Prompting per VLM task | [08](../learn/08-vlm-applications.md) | Scoped fixed templates for captions; CoT for counting; explicit JSON schema for extraction; allow "not present" | Every quick check |
| 38 | VLM failure modes | [08](../learn/08-vlm-applications.md) | Hallucination, presupposition hallucination, compounding errors, hallucinated fields → human review | Pitfall slide in every section |
| 39 | Which retrieval strategy when | [09](../learn/09-multimodal-rag.md) | Plain text → text RAG; clean tables on a budget → PDF parser; charts and scans → ColPali; production → hybrid | Dedicated slide |
| 40 | RAG pitfalls | [09](../learn/09-multimodal-rag.md) | Retrieval and generation fail independently; grounding **reduces** hallucination but doesn't eliminate it | Lecture quiz |
| 41 | Text and image search basics | [09](../learn/09-multimodal-rag.md) | Keyword vs semantic vs hybrid; bi-encoder; ANN for speed; reranking; CLIP shared space for images | Two sets of five quick-check questions |
| 42 | The six-rung ladder | [10](../learn/10-rag-to-agents.md) | LLM → RAG → assistant → tool-using assistant → agent → multi-agent; each adds one thing | Opening and closing slides |
| 43 | Agent loop + guards | [10](../learn/10-rag-to-agents.md) | Think → Act → Observe → Remember; bounded, observable, recoverable; verify success in code | Slide plus notebook guards |
| 44 | ReAct vs Plan-and-Execute vs Reflection | [10](../learn/10-rag-to-agents.md) | ReAct is cheapest; Plan-and-Execute is predictable; Reflection costs the most tokens | Side-by-side slide |
| 45 | When to go multi-agent | [11](../learn/11-multi-agent.md) | Specialisation, parallelism, context isolation, verification. Otherwise stay with one agent | Slides, pre-read, post-read |
| 46 | Coordination cycle + loop bounds | [11](../learn/11-multi-agent.md) | Decompose → execute → attribute → reconcile → re-plan; round limit, token budget, deadlock check, escalate | Dedicated slides with code |
| 47 | Supervisor vs peer-to-peer; parallel vs sequential | [11](../learn/11-multi-agent.md) | Supervisor for accountability (but a single point of failure); P2P for debate; parallel when independent | Slides plus Q&A |
| 48 | Multi-agent safety | [11](../learn/11-multi-agent.md) | Inter-agent messages are untrusted; least privilege; human approval for writes; fail closed | Three slides |
| 49 | State and reducers | [12](../learn/12-langchain-langgraph.md) | Nodes return partial updates; reducers merge them (overwrite by default; `add` appends) | Its own slide, many Q&A rows |
| 50 | `interrupt()` / `Command(resume=...)` | [12](../learn/12-langchain-langgraph.md) | Pause for a human; on resume **the node re-runs from the top** | Slide plus Mentimeter |
| 51 | Retry vs hand back to the model | [12](../learn/12-langchain-langgraph.md) | Retry infrastructure failures; return meaning-based failures to the model as observations | Two slides, post-read |
| 52 | LangChain vs LangGraph | [12](../learn/12-langchain-langgraph.md) | Toolbox vs flowchart; `create_agent` *is* a LangGraph graph | Pre-read, takeaway slide |
| 53 | Responsible vs trustworthy AI | [13](../learn/13-responsible-ai.md) | Promise vs proof | Agenda and pre-read |
| 54 | Five leader questions | [13](../learn/13-responsible-ai.md) | Works as intended? Who could it harm? Whose data? Explainable? Who is accountable? Answer with evidence | Pre-read table |
| 55 | Provider vs deployer | [13](../learn/13-responsible-ai.md) | The deploying organisation answers for the outcome, even if the vendor's model erred (Air Canada) | Repeated several times |
| 56 | Case → control matching | [13](../learn/13-responsible-ai.md) | DPD → guardrails; Samsung → privacy; fake citations → hallucination management; huge bill → cost control; MyCity → documentation | Every case is paired with a control |

---

## 🟠 Medium: understand the idea

- **Ch 01:** multi-task fine-tuning (shared body, one head per task); curriculum (easy → hard); layer-wise vs block-wise vs progressive unfreezing; why full fine-tuning is costly; the PEFT families preview.
- **Ch 02:** batch size and gradient accumulation; imbalanced data and the accuracy paradox; splitting without leakage (split first; time and group splits); domain shift.
- **Ch 03:** soft prompt vs discrete prompt vs system prompt vs RAG; SMoP / APT / IDPG (what each fixes); adapter routing and AdapterFusion; choosing a PEFT method.
- **Ch 04:** intrinsic dimension; the r and α dials; multi-tenant hot-swapping; block-wise quantization and outliers.
- **Ch 05:** the RL vocabulary for LLMs (state, action, policy); why policy gradients are noisy (reward-to-go, baseline, advantage); PPO's "proximal" idea.
- **Ch 06:** red-teaming loop (attack → find gap → fix → re-test, and check for over-blocking); "alignment is not a security boundary"; the 5-step SLM workflow; loss curves again.
- **Ch 07:** SAM (Medium–High: encoder once, 3 masks, data engine); ViT basics; LLaVA data and upgrades; limits of each model.
- **Ch 08:** SAM loss (focal + Dice, Medium–High); composing models (VLM finds, SAM cuts); datasets and metrics (COCO, VQAv2, DocVQA/ANLS, CIDEr, CLIPScore); resolution as the main lever for documents.
- **Ch 09:** Recall@K, nDCG@5, ViDoRe (what they mean); ColPali's bigger index and GPU cost; three storage designs; chunking; "fine-tune the LLM for a domain? Usually no".
- **Ch 10:** three kinds of memory (conversation, task state, long-term); Agentic RAG; why the loop gets expensive; write tools need approval.
- **Ch 11:** MCP vs A2A (+ ANP, AGNTCY); shared state in slices; what to log at each handoff; cost of extra agents.
- **Ch 12:** super-steps and `recursion_limit`; nested timeouts and fallbacks; structured output, `@tool`, chains; time travel and streaming; `Command` handoffs and `Send` fan-out.
- **Ch 13:** the agentic shift (AI that *does* vs *says*); three 90-day priorities; governance as one system; framework one-liners (NIST AI RMF, EU AI Act, ISO/IEC 42001, DPDP, India AI Governance Guidelines).

---

## 🟢 Low: skim only if time allows

- Encoder vs decoder recap; Med-PaLM and BloombergGPT facts (Ch 01).
- Label-error detection with cleanlab; the one-training-step walkthrough (Ch 02).
- Exact GPU tables; paper names like Compacter and P-Tuning v2 (Ch 03).
- Gradient checkpointing details; exact NF4 level values (Ch 04).
- Preference data sourcing; the value-head loss (Ch 05).
- Library names, regex patterns, the Luhn check (Ch 06).
- Variant tables, parameter counts, history dates (Ch 07); benchmark scores and case studies (Ch 08).
- Survey details and EM training (Ch 09); framework landscape (Ch 10); RL tangents (Ch 11).
- Planning architectures beyond ReAct (ReWOO, LLMCompiler) and release dates (Ch 12).
- Deeper framework details, case figures and dates (Ch 13).

---

## Suggested weighting for 24 questions

!!! warning "This is an estimate, not a known blueprint"
    We don't know how the paper is split. The estimate below assumes questions are spread roughly in proportion to the number of sessions per part, nudged by how much emphasis each part got. Use it to plan your revision time, not to predict the paper.

| Part | Sessions | Chapters | Estimated questions (out of 24) |
|---|---|---|---|
| A · Fine-tuning (incl. PEFT, alignment, revision) | 6 | 01–06 | **9–11** |
| B · Vision-language models & multimodal RAG | 3 | 07–09 | **5–6** |
| C · Agents (single, multi-agent, LangGraph) | 3 | 10–12 | **5–7** |
| D · Responsible AI | 1 | 13 | **2–3** |

Part A was also the only block with a dedicated revision session, which is another reason to expect it to carry the most weight.

---

## Themes that span chapters

Questions often test the *same idea* in different clothing. If you understand these themes, many questions become easier.

1. **Freeze vs train.** Almost every method is defined by what's frozen and what's trained.
    - LoRA: freeze W₀, train A and B.
    - Adapters and soft prompts: freeze the backbone, train the new parts.
    - DPO: train the policy against a frozen reference.
    - LLaVA: the encoder is always frozen; the projector is trained, then the projector and LLM together.
    - Gradients still *flow through* frozen weights; they just don't update them.
2. **Start as "do nothing".** New parts begin as no-ops so training starts from the known-good model: LoRA's B = 0 and adapters' near-identity initialisation.
3. **Memory and cost trade-offs.**
    - Optimizer states dominate training memory (Ch 03 and 04).
    - Gradient checkpointing trades compute for memory.
    - ColPali trades index size for accuracy on visual documents.
    - Reflection trades tokens for quality.
    - Every extra agent adds messages, tokens and latency.
4. **Latency at inference.**
    - Merged LoRA adds none; sequential adapters add some.
    - RAG adds a retrieval step; fine-tuning shortens prompts.
    - CLIP is fast because it only scores.
5. **Knowledge vs behaviour.** RAG supplies facts at run time; fine-tuning shapes behaviour; alignment shapes preferences. In Ch 09 the same rule applies: usually don't fine-tune the LLM for domain knowledge.
6. **Don't trust: enforce and verify in code.**
    - Guardrails, not prompts (Ch 06, 13).
    - The model emits, your code executes (Ch 10, 12).
    - Verify "done" in code (Ch 10).
    - Inter-agent messages are untrusted (Ch 11).
    - "An instruction is not a permission; an alert is not a cap" (Ch 13).
7. **A human before anything irreversible.** Write tools need approval (Ch 10, 11), `interrupt()` puts a human in the loop (Ch 12), and there's a 90-day priority on write/approve/spend authority (Ch 13).
8. **Bound every loop.**
    - KL penalty: don't drift too far (Ch 05).
    - Bounded agent loops (Ch 10).
    - Round limit, token budget and deadlock check (Ch 11).
    - `recursion_limit` and oscillation guards (Ch 12).
    - Hard spending caps (Ch 13).
9. **Quality beats quantity.** Curated data (LIMA, Ch 02), a few hundred clean examples for an SLM (Ch 06), and minimum sufficient context in handoffs (Ch 11).
10. **"Reduced, not eliminated."** Alignment, guardrails, red-teaming, grounding and citations all lower risk; none of them *guarantees* safety or truth.
