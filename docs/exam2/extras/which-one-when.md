# Which one when? Scenario → choice

**How to use this page.** Many exam questions describe a situation ("a team wants…", "the system keeps…") and ask you to pick the right method. This page collects every "which one when" rule from the chapters into one place. Each rule reads: **if the scenario says…, choose…**, followed by the reason. Read only the scenario, guess the choice, then check yourself. When a rule surprises you, go back to the chapter linked in the section heading.

---

## Part A · Fine-tuning

### Prompting vs RAG vs fine-tuning ([Ch 01](../learn/01-ft-decisions.md))

**If the scenario says the model *can* do it but needs clearer direction, choose prompt engineering.** Because it costs nothing and works instantly, so always try it first.

**If the scenario says answers need private, current or frequently changing facts (prices, policies, product docs), choose RAG.** Because it gives fresh knowledge, cheap updates and traceable sources.

**If the scenario says an HR-policy chatbot where policies change often, choose RAG plus prompting, not fine-tuning.** Because fine-tuned knowledge is frozen at training time.

**If the scenario says you need a consistent style, a strict format or a narrow repeated task, choose supervised fine-tuning.** Because it bakes the behaviour into the weights.

**If the scenario says you need strict JSON output, choose structured outputs, a JSON schema or constrained decoding first, and fine-tune only if that keeps failing.** Because it's cheaper, and fine-tuning can't guarantee valid output, so keep validation code either way.

**If the scenario says the model must follow many kinds of instructions, choose instruction fine-tuning.** Because it generalises to unseen instructions.

**If the scenario says the model doesn't know the domain's language at all, choose unsupervised continued pre-training, then SFT.** Because raw text is cheap and teaches the vocabulary.

**If the scenario says several related tasks served by one model, choose multi-task fine-tuning.** Because it gives a shared body, one head per task and one deployment.

**If the scenario says outputs must be safe and match human preferences, choose alignment fine-tuning (RLHF or DPO).** Because it learns from feedback.

**If the scenario says lowest latency matters most, choose fine-tuning over RAG.** Because prompts are shorter and there's no retrieval step.

**If the scenario says you need facts *and* consistent behaviour, combine them: fine-tune for behaviour, RAG for facts.** Because production systems often use all three.

**If the scenario says you're not sure whether fine-tuning is worth it, measure the base model on a test set first.** Because if it's already good, you don't need to fine-tune.

### Which fine-tuning method? ([Ch 01](../learn/01-ft-decisions.md), [03](../learn/03-peft-adapters-prompts.md), [04](../learn/04-lora-qlora.md))

**If the scenario says lots of data and compute and a big domain shift, choose full fine-tuning.** Because it gives maximum adaptation, though it's costly and can forget.

**If the scenario says general instruction tuning or domain adaptation on a normal budget, choose LoRA.** Because it's the default today ("nowadays LoRA is mostly used").

**If the scenario says a very large model with limited GPU memory, choose QLoRA.** Because the frozen base is stored in 4-bit NF4.

**If the scenario says several separate tasks or domains on one backbone, choose routed task adapters or hot-swapped LoRA adapters.** Because you keep one backbone and a small module per task, with no forgetting.

**If the scenario says a latency-sensitive deployment, choose merged LoRA and avoid sequential (Houlsby) adapters.** Because merged LoRA keeps the original structure, while sequential adapters add steps.

**If the scenario says generation tasks (summarisation, dialogue) with the fewest trainable parameters, choose prefix tuning.** Because it learns K/V prefixes at every layer.

**If the scenario says you don't want to touch the architecture at all and just want to learn input vectors, choose prompt tuning.** Because it adds virtual tokens at the input layer only.

**If the scenario says one long, expensive soft prompt, choose SMoP.** Because it uses several short prompts plus a router.

**If the scenario says one prefix length for every layer feels wrong, choose APT.** Because it adapts the prefix length per layer (bottom layers usually need more).

**If the scenario says inputs vary a lot within one task, choose IDPG (or SMoP).** Because it generates a prompt per input.

**If the scenario says train only existing bias terms and add nothing new, choose BitFit (selective PEFT).** Because it updates a chosen subset of existing weights.

### Merge vs hot-swap ([Ch 04](../learn/04-lora-qlora.md))

**If the scenario says one task and the lowest possible latency, merge B·A into W₀.** Because the result has the same shape and speed as the original model.

**If the scenario says 200 clients on one base model, each wanting their own behaviour, keep the adapters separate and hot-swap them per request, with a router using the client ID.** Because you hold one resident base plus small adapter files instead of 200 full copies.

### QLoRA's memory problems ([Ch 04](../learn/04-lora-qlora.md))

**If the scenario says the base model itself is too big to store, choose NF4 4-bit quantization.** Because it shrinks the biggest item in memory about 4×.

**If the scenario says a few outliers ruin quantization precision, choose block-wise quantization.** Because each small block gets its own scale.

**If the scenario says the block scaling constants add up, choose double quantization.** Because it quantizes the constants themselves.

**If the scenario says activations eat memory on long sequences, choose gradient checkpointing.** Because it keeps some activations and recomputes the rest.

**If the scenario says occasional OOM crashes on spikes (long batches), choose the paged optimizer.** Because it spills optimizer states to CPU RAM instead of crashing.

### Training problems ([Ch 02](../learn/02-ft-data-training.md))

**If the scenario says training loss is falling while validation loss rises, it's overfitting, so choose early stopping, fewer epochs, regularisation or more varied data.** Because the model is memorising.

**If the scenario says great on the new domain but worse on general tasks, it's catastrophic forgetting, so choose a lower LR, replay or data mixing, L2-SP or EWC, weight averaging, freezing, or PEFT.** Because general skills are being overwritten.

**If the scenario says the model is already fine-tuned and degraded and you want one quick fix, choose weight averaging (WiSE-FT).** Because it blends the fine-tuned and original weights in one pass.

**If the scenario says both losses are flat, there's no learning, so check for a bug, frozen parameters, or an LR far too low.** Because nothing is updating.

**If the scenario says an unknown run length or continual training, choose the WSD schedule.** Because you can start the decay whenever you decide to stop.

**If the scenario says a big batch doesn't fit in memory, choose gradient accumulation.** Because it adds up micro-batch gradients before one update.

**If the scenario says frequent gradient clipping, lower the learning rate.** Because clipping is a seatbelt, not a fix.

**If the scenario says 99% accuracy on fraud detection, check recall, precision, F1 and PR-AUC.** Because of the accuracy paradox on imbalanced data.

**If the scenario says you need to scale features, split first and fit the transforms on the training set only.** Because otherwise test information leaks.

### RLHF vs DPO ([Ch 05](../learn/05-rlhf-dpo.md))

**If the scenario says a fixed, offline set of (prompt, chosen, rejected) pairs and you want something simple and stable, choose DPO.** Because it needs no reward model, no RL loop and only 2 models.

**If the scenario says you need online exploration, or a complex, sequential or changing reward (for example a coding or web agent with a measurable outcome), choose RLHF with PPO.** Because it samples fresh outputs and scores them.

**If the scenario says the model found a loophole and gets high reward for bad answers, choose the KL penalty (a stronger pull towards the reference).** Because it stops reward hacking and drift.

### SLM vs large model; alignment vs guardrails ([Ch 06](../learn/06-revision-slm-guardrails.md))

**If the scenario says a narrow task, high volume, on-device or on-prem, and good task data, choose a fine-tuned SLM.** Because it's cheaper and faster, and you own the weights.

**If the scenario says very broad, open-ended tasks, choose a large general model.** Because you need its breadth of knowledge.

**If the scenario says "the model is aligned, so it's safe", still add guardrails.** Because training is not a security boundary.

**If the scenario says users paste emails, phone numbers or card numbers, choose PII detection and redaction on both input and output.** Because it protects privacy and supports legal obligations.

**If the scenario says you need to know how safe the system really is, choose red-teaming (attack → find the gap → fix → re-test).** Because it measures safety and checks for over-blocking.

---

## Part B · Vision-language models & multimodal RAG

### Which VLM? ([Ch 07](../learn/07-vlm-architectures.md), [08](../learn/08-vlm-applications.md))

**If the scenario says tag 10 million product photos into 20 categories, image search, or moderation, choose CLIP (zero-shot).** Because it does fast similarity scoring and no generation is needed.

**If the scenario says a conversational description or general visual Q&A, choose a LLaVA-style model.** Because a frozen encoder plus a projector plus an LLM gives you chat.

**If the scenario says reading text in images, pointing at regions, documents, or multiple images, choose Qwen-VL.** Because it has grounding tokens, OCR training and a fixed 256-token budget.

**If the scenario says the exact pixel outline of an object, choose SAM (with a VLM or detector for the "what").** Because it gives promptable, ambiguity-aware masks; SAM knows *where*, not *what*.

**If the scenario says find an object, then cut it out, use a VLM to find it and SAM to segment it.** Because each does what it's good at.

**If the scenario says add grounding to a LLaVA-like model with minimal change, emit coordinates as text tokens (Qwen-VL style).** Because no new output head is needed.

**If the scenario says pure bulk text extraction from clean scans, choose a dedicated OCR pipeline.** Because it's faster and cheaper than a VLM.

**If the scenario says check whether requirements are covered by test documents, use a document VLM with a JSON schema for the extraction, and embed, retrieve, rerank and an LLM judge for the matching (no VLM).** Because matching is a retrieval problem.

### Which prompt? ([Ch 08](../learn/08-vlm-applications.md))

**If the scenario says captions are too generic, add scope and detail constraints first.** Because it's cheaper than switching models.

**If the scenario says cataloguing many items, choose one fixed template.** Because consistency beats creativity.

**If the scenario says counting, comparison or multi-step visual reasoning, choose chain-of-thought.** Because the reasoning comes from the LLM backbone.

**If the scenario says the model answers "black" to "What colour is the cat?" when there's no cat, ground the prompt and allow "not present".** Because that fixes presupposition hallucination.

**If the scenario says extract invoice fields, show the exact JSON schema, allow null, and add human review.** Because otherwise fields get hallucinated.

**If the scenario says chart questions, extract the values first, then reason.** Because it separates reading from reasoning.

**If the scenario says documents are read poorly, choose a higher resolution.** Because it's the biggest lever for documents.

### Which retrieval strategy? ([Ch 09](../learn/09-multimodal-rag.md))

**If the scenario says a plain text corpus (policies, articles), choose text RAG (dense or hybrid).** Because multimodal adds cost for nothing.

**If the scenario says product photos with simple descriptive queries, choose CLIP-style image search.** Because text and images share one space, so there's no captioning step.

**If the scenario says reports with charts and tables, or scanned pages, choose multimodal RAG with vision-space retrieval (ColPali).** Because OCR pipelines break on visuals.

**If the scenario says yearly PDFs with clean tables, fixed fields and a tight budget, choose a PDF parser plus text RAG (parse-then-embed).** Because it's cheaper; ColPali only pays off for complex visuals.

**If the scenario says exact values and calculations from tables, choose table or structured RAG.** Because it needs exact numbers, not similarity.

**If the scenario says exact codes or IDs plus meaning, choose hybrid search (BM25 plus dense).** Because it combines keyword precision with semantic recall.

**If the scenario says multi-hop questions across linked entities, choose Graph RAG.** Because it follows relationships.

**If the scenario says a real production mix, choose a hybrid: parse text and tables, plus vision retrieval for complex pages.** Because there's no single best approach.

**If the scenario says retrieval quality is weak, fine-tune the embedding model or reranker (or add reranking), not the LLM.** Because domain knowledge comes from retrieval.

**If the scenario says near-miss wrong passages keep ranking high, choose hard negative mining or reranking.** Because it improves precision.

**If the scenario says search is too slow at scale, choose an ANN index.** Because it's much faster, with a small accuracy loss.

---

## Part C · Agents

### Which rung? Workflow or agent? ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md))

**If the scenario says a general-knowledge question, choose a plain LLM.** Because the answer is in the weights.

**If the scenario says a question about private or recent documents, choose RAG.** Because it needs grounded, citable evidence.

**If the scenario says follow-up questions in a chat ("And the hotel?"), choose an assistant (RAG plus conversation context).** Because it needs the earlier turn.

**If the scenario says the same fixed sequence of API calls every time, choose a workflow or tool-using assistant.** Because it's predictable, auditable and cheaper.

**If the scenario says a fully rule-based process with clean inputs, choose ordinary automation (no LLM).** Because it's cheaper, faster and more predictable.

**If the scenario says an open-ended goal where later steps depend on earlier results, choose an agent.** Because it needs run-time decisions and re-planning.

**If the scenario says many tools that change across several systems, choose MCP.** Because tools are discovered at run time and the server owns the schema.

**If the scenario says only a few tools, choose plain function calling.** Because MCP isn't mandatory.

**If the scenario says start building an agent quickly, choose LangChain `create_agent`, and drop to LangGraph for more control.** Because `create_agent` is already a LangGraph graph.

### Which loop shape? ([Ch 10](../learn/10-rag-to-agents.md))

**If the scenario says a step-by-step task that should be cheapest and easiest to debug, choose ReAct.** Because it takes one thought and one action at a time.

**If the scenario says deep research or batch jobs, predictable cost, or parallel steps, choose Plan-and-Execute.** Because it plans first and re-plans only the failed tail.

**If the scenario says written output that must be polished (an itinerary write-up, a code review), choose Reflection.** Because it drafts, critiques and revises, though it costs the most tokens.

**If the scenario says the agent claims "done" when it isn't, verify success in code (for example, require that `check_policy` passed).** Because a better prompt isn't enough.

### Which memory? ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md))

**If the scenario says what the user said earlier in this chat, choose conversation state.** Because it lives for one thread.

**If the scenario says held fares, budget remaining or steps done, choose task state (the scratchpad).** Because it lives for one run; "without task state there is no loop".

**If the scenario says home airport, employee grade or "avoid Air India", choose long-term memory (the LangGraph Store).** Because it lives across sessions.

**If the scenario says resume after a crash, pause for approval, or rewind, choose a checkpointer with a `thread_id`.** Because it saves the full state after every super-step.

### Single or multi-agent? Which architecture? ([Ch 11](../learn/11-multi-agent.md))

**If the scenario says a simple or deterministic task, choose one agent or a workflow.** Because extra agents add cost and failure points.

**If the scenario says one agent with too many tools or a crowded context, first try fewer tools, clearer descriptions or isolated context, and only then split.** Because you should go multi-agent only when the problem is structural.

**If the scenario says it needs specialisation, parallelism, context isolation or independent verification, choose multi-agent.** Because those are the four good reasons to split.

**If the scenario says sub-tasks don't depend on each other, run them in parallel.** Because it lowers latency.

**If the scenario says one step needs another's output, run them sequentially.** Because it keeps the dependencies correct.

**If the scenario says spending money, approvals or clear accountability, choose a supervisor.** Because it gives simple governance and tracing (accept the single point of failure).

**If the scenario says debate, critique, writer–reviewer loops or simulations, choose peer-to-peer.** Because it's flexible and needs fewer hops.

**If the scenario says delegating to another company's agent, choose A2A.** Because it connects agent ↔ agent (Agent Cards, Tasks).

**If the scenario says giving an agent access to tools or data, choose MCP.** Because it connects agent ↔ tools.

**If the scenario says agents keep undoing each other's fixes, add a deadlock check and escalate to a human.** Because once the same conflict is seen twice, you should stop early.

**If the scenario says booking, paying or sending, require human approval and least privilege.** Because write actions can't be undone.

**If the scenario says a result fails validation, fail closed: don't merge it into shared state.** Because invalid results must never be promoted.

### Reliability in LangGraph ([Ch 12](../learn/12-langchain-langgraph.md))

**If the scenario says a dependency is slow, use a timeout, then a bounded retry.** Because infrastructure failures are retryable.

**If the scenario says a dependency is down, use a fallback (`with_fallbacks`), then degrade *visibly*.** Because a made-up answer is worse than an outage.

**If the scenario says no valid option exists ("no flights"), return it to the model as an observation.** Because it's about meaning, not infrastructure.

**If the scenario says bad arguments, validate before running (a typed tool signature).** Because it fails early.

**If the scenario says the loop won't end, use `recursion_limit` plus an oscillation guard.** Because the limit bounds super-steps and the guard catches A → B → A.

**If the scenario says a lost reply means a retry might book twice, use an idempotency key derived from the thread and the intent.** Because retries reuse the same key.

**If the scenario says an irreversible action, use `interrupt()` for human approval, with no side effects before it.** Because the node re-runs from the top on resume.

**If the scenario says two parallel nodes write the same field, use a reducer (for example `add`).** Because otherwise the updates collide.

---

## Part D · Responsible AI

### Which control? ([Ch 13](../learn/13-responsible-ai.md))

**If the scenario says the bot misbehaves after an update or model change (DPD), choose guardrails, re-tested after every change.** Because guardrails aren't set once.

**If the scenario says "we told it in the prompt never to…", and it did, choose guardrails: a permission, not an instruction.** Because "an instruction is not a control".

**If the scenario says staff paste confidential data into a public AI tool (Samsung), choose privacy.** Because you need rules on what data goes where, and the organisation stays responsible under DPDP.

**If the scenario says invented citations, clauses or facts (Supreme Court of India, Deloitte Australia), choose hallucination management.** Because you should treat the output as a draft and check it in proportion to the stakes.

**If the scenario says a surprise huge bill, a runaway loop or a leaked API key, choose cost control: a hard cap, loop limits and a named owner.** Because "an alert is not a cap".

**If the scenario says nobody owns the bot and a disclaimer replaces a fix (NYC MyCity), choose system documentation: a one-page record of the owner, boundaries and failure detection.** Because a disclaimer isn't a control.

**If the scenario says an agent deletes, approves or spends without asking (PocketOS), choose permissions plus human approval for write, modify, approve and spend actions.** Because of the agentic shift from *says* to *does*.

**If the scenario says "the bot said it, not us" (Air Canada), choose accountability: the organisation is liable.** Because deployers answer for the outcome.

**If the scenario says overall accuracy looks fine but one group is harmed (Amazon recruiting), check error rates per group (leader question 2).** Because averages hide group harm.
