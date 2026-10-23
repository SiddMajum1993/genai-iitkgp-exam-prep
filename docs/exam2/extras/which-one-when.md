# Which one when? Scenario → choice

**How to use this page.** Many exam questions describe a situation ("a team wants…", "the system keeps…") and ask you to pick the right method. This page collects every "which one when" rule from the chapters into one place. Each row reads: **if the scenario says… → choose… → because…**. Cover the middle column and test yourself. When a row surprises you, go back to the chapter linked in the section heading.

---

## Part A · Fine-tuning

### Prompting vs RAG vs fine-tuning ([Ch 01](../learn/01-ft-decisions.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| The model *can* do it, but needs clearer direction | **Prompt engineering** | Zero cost, instant; always try first |
| Answers need **private, current or frequently changing** facts (prices, policies, product docs) | **RAG** | Fresh knowledge, cheap updates, traceable sources |
| HR-policy chatbot; policies change often | **RAG + prompting**, not fine-tuning | Fine-tuned knowledge is frozen at training time |
| Need a **consistent style, strict format** or a narrow repeated task | **Supervised fine-tuning** | Bakes the behaviour into the weights |
| Need strict JSON output | First **structured outputs / JSON schema / constrained decoding**; fine-tune only if that keeps failing | Cheaper; and fine-tuning can't guarantee valid output, so keep validation code either way |
| Model must follow **many kinds of instructions** | **Instruction fine-tuning** | Generalises to unseen instructions |
| Model doesn't know the domain's language at all | **Unsupervised continued pre-training**, then SFT | Raw text is cheap and teaches vocabulary |
| Several related tasks, one model to serve | **Multi-task fine-tuning** | Shared body, one head per task, one deployment |
| Outputs must be safe and match human preferences | **Alignment fine-tuning** (RLHF / DPO) | Learns from feedback |
| Lowest latency matters most | **Fine-tuning** (over RAG) | Shorter prompts, no retrieval step |
| Facts *and* consistent behaviour | **Combine**: fine-tune for behaviour, RAG for facts | Production systems often use all three |
| Not sure whether fine-tuning is worth it | **Measure the base model on a test set first** | If it's already good, don't fine-tune |

### Which fine-tuning method? ([Ch 01](../learn/01-ft-decisions.md), [03](../learn/03-peft-adapters-prompts.md), [04](../learn/04-lora-qlora.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Lots of data and compute, big domain shift | **Full fine-tuning** | Maximum adaptation (but costly, and it can forget) |
| General instruction tuning or domain adaptation, normal budget | **LoRA** | The default today ("nowadays LoRA is mostly used") |
| Very large model, **limited GPU memory** | **QLoRA** | The frozen base is stored in 4-bit NF4 |
| Several separate tasks or domains on one backbone | **Task adapters** (routed) or **LoRA adapters, hot-swapped** | One backbone, a small module per task, no forgetting |
| **Latency-sensitive** deployment | **Merged LoRA**; avoid **sequential (Houlsby) adapters** | Merged LoRA has the original structure; sequential adapters add steps |
| Generation tasks (summarisation, dialogue) with the fewest trainable parameters | **Prefix tuning** | Learned K/V prefixes at every layer |
| Don't want to touch the architecture at all; just learn input vectors | **Prompt tuning** | Virtual tokens at the input layer only |
| One long, expensive soft prompt | **SMoP** | Several short prompts + a router |
| One prefix length for every layer feels wrong | **APT** | Adaptive prefix length per layer (bottom layers usually need more) |
| Inputs vary a lot within one task | **IDPG** (or SMoP) | Generates a prompt per input |
| Train only existing bias terms, add nothing new | **BitFit** (selective PEFT) | Updates a chosen subset of existing weights |

### Merge vs hot-swap ([Ch 04](../learn/04-lora-qlora.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| One task, lowest possible latency | **Merge** B·A into W₀ | Same shape and speed as the original model |
| 200 clients, one base model, each wants their own behaviour | **Keep adapters separate and hot-swap** per request (router uses the client ID) | One resident base + small adapter files instead of 200 full copies |

### QLoRA's memory problems ([Ch 04](../learn/04-lora-qlora.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| The base model itself is too big to store | **NF4 4-bit quantization** | Shrinks the biggest item in memory about 4× |
| A few outliers ruin quantization precision | **Block-wise quantization** | Each small block gets its own scale |
| The block scaling constants add up | **Double quantization** | Quantizes the constants themselves |
| Activations eat memory on long sequences | **Gradient checkpointing** | Keep some activations, recompute the rest |
| Occasional **OOM crashes on spikes** (long batches) | **Paged optimizer** | Spills optimizer states to CPU RAM instead of crashing |

### Training problems ([Ch 02](../learn/02-ft-data-training.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Train loss ↓, validation loss ↑ | **Overfitting** → early stopping, fewer epochs, regularisation, more varied data | The model is memorising |
| Great on the new domain, worse on general tasks | **Catastrophic forgetting** → lower LR, replay / data mixing, L2-SP / EWC, weight averaging, freezing, PEFT | General skills are being overwritten |
| Already fine-tuned and degraded; want one quick fix | **Weight averaging (WiSE-FT)** | Blends fine-tuned and original weights in one pass |
| Both losses flat | **No learning** → check for a bug, frozen parameters, or an LR far too low | Nothing is updating |
| Unknown run length / continual training | **WSD schedule** | You can start the decay whenever you decide to stop |
| Big batch doesn't fit in memory | **Gradient accumulation** | Adds up micro-batch gradients before one update |
| Frequent gradient clipping | **Lower the learning rate** | Clipping is a seatbelt, not a fix |
| 99% accuracy on fraud detection | **Check recall, precision, F1, PR-AUC** | The accuracy paradox on imbalanced data |
| Need to scale features | **Split first**, fit transforms on train only | Otherwise test information leaks |

### RLHF vs DPO ([Ch 05](../learn/05-rlhf-dpo.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| A **fixed, offline** set of (prompt, chosen, rejected) pairs; want simple and stable | **DPO** | No reward model, no RL loop, 2 models |
| Need **online exploration**; complex, sequential or changing reward (e.g. coding or web agent with a measurable outcome) | **RLHF with PPO** | Samples fresh outputs and scores them |
| The model found a loophole and gets high reward for bad answers | **KL penalty** (stronger pull towards the reference) | Stops reward hacking and drift |

### SLM vs large model; alignment vs guardrails ([Ch 06](../learn/06-revision-slm-guardrails.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Narrow task, high volume, must run on-device or on-prem, good task data | **Fine-tuned SLM** | Cheaper, faster, you own the weights |
| Very broad, open-ended tasks | **Large general model** | Breadth of knowledge |
| "The model is aligned, so it's safe" | Still add **guardrails** | Training is not a security boundary |
| Users paste emails, phone numbers or card numbers | **PII detection + redaction** on input and output | Protects privacy; supports legal obligations |
| Need to know how safe the system really is | **Red-teaming** (attack → find gap → fix → re-test) | Measures safety, and checks for over-blocking |

---

## Part B · Vision-language models & multimodal RAG

### Which VLM? ([Ch 07](../learn/07-vlm-architectures.md), [08](../learn/08-vlm-applications.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Tag 10 million product photos into 20 categories; image search; moderation | **CLIP** (zero-shot) | Fast similarity scoring; no generation needed |
| A conversational description or general visual Q&A | **LLaVA-style** | Frozen encoder + projector + LLM = chat |
| Read text in images, point at regions, documents, multiple images | **Qwen-VL** | Grounding tokens, OCR training, fixed 256-token budget |
| The exact pixel outline of an object | **SAM** (with a VLM or detector for the "what") | Promptable, ambiguity-aware masks; SAM knows *where*, not *what* |
| Find an object, then cut it out | **VLM to find it + SAM to segment it** | Each does what it's good at |
| Add grounding to a LLaVA-like model with minimal change | Emit coordinates **as text tokens** (Qwen-VL style) | No new output head needed |
| Pure bulk text extraction from clean scans | **Dedicated OCR pipeline** | Faster and cheaper than a VLM |
| Check whether requirements are covered by test documents | Extraction: **document VLM + JSON schema**; matching: **embed, retrieve, rerank, LLM judge** (no VLM) | Matching is a retrieval problem |

### Which prompt? ([Ch 08](../learn/08-vlm-applications.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Captions too generic | **Add scope and detail constraints** first | Cheaper than switching models |
| Cataloguing many items | **One fixed template** | Consistency beats creativity |
| Counting, comparison, multi-step visual reasoning | **Chain-of-thought** | Reasoning comes from the LLM backbone |
| Model answers "black" to "What colour is the cat?" with no cat | **Ground the prompt and allow "not present"** | Fixes presupposition hallucination |
| Extract invoice fields | **Show the exact JSON schema**, allow null, add **human review** | Otherwise fields get hallucinated |
| Chart questions | **Extract the values first, then reason** | Separates reading from reasoning |
| Documents are read poorly | **Higher resolution** | The biggest lever for documents |

### Which retrieval strategy? ([Ch 09](../learn/09-multimodal-rag.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Plain text corpus (policies, articles) | **Text RAG** (dense or hybrid) | Multimodal adds cost for nothing |
| Product photos with simple descriptive queries | **CLIP-style image search** | Shared text–image space; no captioning step |
| Reports with charts and tables; scanned pages | **Multimodal RAG with vision-space retrieval (ColPali)** | OCR pipelines break on visuals |
| Yearly PDFs with clean tables, fixed fields, tight budget | **PDF parser + text RAG** (parse-then-embed) | Cheaper; ColPali only pays off for complex visuals |
| Exact values and calculations from tables | **Table / structured RAG** | Needs exact numbers, not similarity |
| Exact codes or IDs plus meaning | **Hybrid (BM25 + dense)** | Keyword precision + semantic recall |
| Multi-hop questions across linked entities | **Graph RAG** | Follows relationships |
| Real production mix | **Hybrid**: parse text and tables + vision retrieval for complex pages | No single best approach |
| Retrieval quality is weak | **Fine-tune the embedding or reranker** (or add reranking), not the LLM | Domain knowledge comes from retrieval |
| Near-miss wrong passages keep ranking high | **Hard negative mining / reranking** | Improves precision |
| Search is too slow at scale | **ANN index** | Much faster, with a small accuracy loss |

---

## Part C · Agents

### Which rung? Workflow or agent? ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| General-knowledge question | **Plain LLM** | The answer is in the weights |
| Question about private or recent documents | **RAG** | Needs grounded, citable evidence |
| Follow-up questions in a chat ("And the hotel?") | **Assistant** (RAG + conversation context) | Needs the earlier turn |
| Same fixed sequence of API calls every time | **Workflow / tool-using assistant** | Predictable, auditable, cheaper |
| Fully rule-based process, clean inputs | **Ordinary automation (no LLM)** | Cheaper, faster, more predictable |
| Open-ended goal; later steps depend on earlier results | **Agent** | Needs run-time decisions and re-planning |
| Many tools that change across several systems | **MCP** | Discovery at run time; the server owns the schema |
| A few tools only | **Plain function calling** | MCP isn't mandatory |
| Start building an agent quickly | **LangChain `create_agent`**; drop to **LangGraph** for more control | `create_agent` is already a LangGraph graph |

### Which loop shape? ([Ch 10](../learn/10-rag-to-agents.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Step-by-step task; cheapest and easiest to debug | **ReAct** | One thought + one action at a time |
| Deep research or batch jobs; predictable cost; parallel steps | **Plan-and-Execute** | Plan first, re-plan only the failed tail |
| Written output that must be polished (itinerary write-up, code review) | **Reflection** | Draft → critique → revise (costs the most tokens) |
| Agent claims "done" when it isn't | **Verify success in code** (e.g. require `check_policy` passed) | A better prompt isn't enough |

### Which memory? ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| What the user said earlier in this chat | **Conversation state** | Lives for one thread |
| Held fares, budget remaining, steps done | **Task state** (scratchpad) | Lives for one run; "without task state there is no loop" |
| Home airport, employee grade, "avoid Air India" | **Long-term memory** (LangGraph **Store**) | Lives across sessions |
| Resume after a crash; pause for approval; rewind | **Checkpointer** with a `thread_id` | Saves the full state after every super-step |

### Single or multi-agent? Which architecture? ([Ch 11](../learn/11-multi-agent.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Simple or deterministic task | **One agent or a workflow** | Extra agents add cost and failure points |
| One agent with too many tools or a crowded context | First try **fewer tools, clearer descriptions or isolated context**; then split | Go multi-agent only when the problem is structural |
| Needs specialisation, parallelism, context isolation or independent verification | **Multi-agent** | Those are the four good reasons to split |
| Sub-tasks don't depend on each other | **Parallel** | Lower latency |
| One step needs another's output | **Sequential** | Correct dependencies |
| Spending money, approvals, clear accountability | **Supervisor** | Simple governance and tracing (accept the single point of failure) |
| Debate, critique, writer–reviewer loops, simulations | **Peer-to-peer** | Flexible, fewer hops |
| Delegating to another company's agent | **A2A** | Agent ↔ agent (Agent Cards, Tasks) |
| Giving an agent access to tools or data | **MCP** | Agent ↔ tools |
| Agents keep undoing each other's fixes | **Deadlock check + escalate to a human** | Same conflict seen twice → stop early |
| Booking, paying, sending | **Human approval + least privilege** | Write actions can't be undone |
| A result fails validation | **Fail closed**: don't merge it into shared state | Invalid results must never be promoted |

### Reliability in LangGraph ([Ch 12](../learn/12-langchain-langgraph.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Dependency is slow | **Timeout, then bounded retry** | Infrastructure failures are retryable |
| Dependency is down | **Fallback (`with_fallbacks`), then degrade *visibly*** | A made-up answer is worse than an outage |
| No valid option exists ("no flights") | **Return it to the model as an observation** | It's about meaning, not infrastructure |
| Bad arguments | **Validate before running** (typed tool signature) | Fail early |
| Loop won't end | **`recursion_limit` + oscillation guard** | Bounds super-steps; catches A → B → A |
| Lost reply → retry might book twice | **Idempotency key** derived from thread + intent | Retries reuse the same key |
| Irreversible action | **`interrupt()` for human approval**; no side effects before it | The node re-runs from the top on resume |
| Two parallel nodes write the same field | **A reducer** (e.g. `add`) | Otherwise updates collide |

---

## Part D · Responsible AI

### Which control? ([Ch 13](../learn/13-responsible-ai.md))

| If the scenario says… | Choose… | Because… |
|---|---|---|
| Bot misbehaves after an update or model change (DPD) | **Guardrails**, re-tested after every change | Guardrails aren't set once |
| "We told it in the prompt never to…", and it did | **Guardrails**: a permission, not an instruction | "An instruction is not a control" |
| Staff paste confidential data into a public AI tool (Samsung) | **Privacy** | Rules on what data goes where; the organisation stays responsible under DPDP |
| Invented citations, clauses or facts (Supreme Court of India, Deloitte Australia) | **Hallucination management** | Treat output as a draft; check in proportion to the stakes |
| Surprise huge bill, runaway loop, leaked API key | **Cost control**: hard cap, loop limits, named owner | "An alert is not a cap" |
| Nobody owns the bot; a disclaimer replaces a fix (NYC MyCity) | **System documentation**: a one-page record (owner, boundaries, failure detection) | A disclaimer isn't a control |
| An agent deletes, approves or spends without asking (PocketOS) | **Permissions + human approval** for write / modify / approve / spend | The agentic shift from *says* to *does* |
| "The bot said it, not us" (Air Canada) | **Accountability**: the organisation is liable | Deployers answer for the outcome |
| Overall accuracy looks fine, but one group is harmed (Amazon recruiting) | **Check error rates per group** (leader question 2) | Averages hide group harm |
