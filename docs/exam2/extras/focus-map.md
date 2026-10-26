# Focus map: where the questions will likely come from

**How to use this page.** This page pulls together the "Exam radar" sections from all 13 chapters into one ranked list, so you can decide where to spend your time.

- Start with the 🔴 list. You should be able to explain every item in one sentence and recognise it inside a scenario.
- Then skim the 🟠 list. Leave the 🟢 items for the end, if you have time.
- The rankings come from how much class time a topic got, how often it appeared in the pre-read, slides and post-read, how many student questions it drew, and whether it was re-taught in the revision session.
- They are **our estimate, not an official blueprint**.

---

## 🔴 High: know these cold

Ranked roughly by how strong the evidence is. Topics that were also **re-taught in the revision session** (Chapter 06) are marked ♻️ and sit near the top.

### Ranks 1–9: also re-taught in revision ♻️

**1. Prompt vs RAG vs fine-tuning ♻️** ([Ch 01](../learn/01-ft-decisions.md)). Use RAG when knowledge is missing or keeps changing, and fine-tuning when you need a particular style, format or behaviour. Try prompting first, because it's cheapest. *Why it ranks here:* it was the most-asked topic, with about 25 Q&A rows, it has its own decision matrix, and it was re-taught.

**2. How LoRA works ♻️** ([Ch 04](../learn/04-lora-qlora.md)). Freeze W₀ and train a side path B·A, where B starts at zero, scaled by α/r. *Why it ranks here:* it's in the pre-read and the slides, drew about 10 Q&A rows, and was re-taught.

**3. Why LoRA saves memory ♻️** ([Ch 04](../learn/04-lora-qlora.md)). Frozen weights need **no gradients and no Adam states**, and that's the main saving. *Why it ranks here:* it's repeated in the slides and the Q&A, and was re-taught.

**4. QLoRA's ingredients ♻️** ([Ch 04](../learn/04-lora-qlora.md)). An NF4 base, double quantization and the paged optimizer, plus gradient checkpointing. **Only the frozen base is stored in 4 bits**; A and B stay in BF16. *Why it ranks here:* it took half a lecture, and students asked about it four different ways in the Q&A.

**5. Merging means no extra latency ♻️** ([Ch 04](../learn/04-lora-qlora.md)). After training, B·A folds into W₀. Houlsby adapters stay in the network and add latency. *Why it ranks here:* the pre-read calls it "the fundamental advantage".

**6. DPO ♻️** ([Ch 05](../learn/05-rlhf-dpo.md)). It learns from (prompt, chosen, rejected) examples with **no reward model and no RL**, but it **keeps a frozen reference**, and β controls how strongly it pulls. *Why it ranks here:* it's in the pre-read and the slides, has a full notebook, and was re-taught.

**7. RLHF's three stages ♻️** ([Ch 05](../learn/05-rlhf-dpo.md)). SFT first, then a reward model trained from comparisons, then RL with PPO and a KL penalty. *Why it ranks here:* the InstructGPT slide covers it, and it was re-taught.

**8. The 4 models in PPO ♻️** ([Ch 05](../learn/05-rlhf-dpo.md)). Policy, reference, reward and value. DPO needs only two: the policy and a frozen reference. *Why it ranks here:* it makes a natural "which is NOT" question.

**9. Reward hacking and the KL penalty ♻️** ([Ch 05](../learn/05-rlhf-dpo.md)). The policy exploits flaws in the reward model, and the KL penalty keeps it close to the reference. *Why it ranks here:* the slide example is memorable.

### Ranks 10–56

**10. Workflow vs agent** ([Ch 10](../learn/10-rag-to-agents.md), [12](../learn/12-langchain-langgraph.md)). Ask who decides the next step: you (a workflow) or the model at run time (an agent). "If you can draw the flowchart in advance, you don't need an agent." *Why it ranks here:* it spans two chapters, their pre-reads and post-reads, and many Q&A threads.

**11. Tool calling** ([Ch 10](../learn/10-rag-to-agents.md)). The model **emits** a request and **your code executes** it, so the safety checks live at that boundary. *Why it ranks here:* the course stresses that "the model never touches the API".

**12. MCP** ([Ch 10](../learn/10-rag-to-agents.md)). `list_tools()` and `call_tool()`, with tools discovered at run time and the server owning the schema. MCP connects an agent to its tools. *Why it ranks here:* it drew 10+ Q&A questions, the most in that session.

**13. CLIP** ([Ch 07](../learn/07-vlm-architectures.md)). Two encoders share one space and are trained with a contrastive grid. It does zero-shot classification with prompts like "a photo of a {label}", but it **cannot generate text**. *Why it ranks here:* it got the most slide time and a 6-question quick check.

**14. LLaVA's two stages** ([Ch 07](../learn/07-vlm-architectures.md)). Stage 1 trains **only the projector**; Stage 2 trains the projector and the LLM. The vision encoder is **always frozen**. *Why it ranks here:* it had its own 6-question quick check.

**15. Qwen-VL** ([Ch 07](../learn/07-vlm-architectures.md)). It uses a cross-attention adapter with **256 learned queries**, a fixed count that keeps position information, and writes bounding boxes as text tokens. *Why it ranks here:* it's a stated learning objective.

**16. Parse-then-embed vs ColPali** ([Ch 09](../learn/09-multimodal-rag.md)). OCR pipelines break on charts and tables. ColPali skips OCR and embeds the **page image** as about 1,000 patch vectors. *Why it ranks here:* it appears in a lecture quiz, the pre-read, the post-read and the Q&A.

**17. Late interaction** ([Ch 09](../learn/09-multimodal-rag.md)). Each query token finds its best-matching patch, and those scores are added up. A page is many vectors, not one. *Why it ranks here:* it was in the lecture quiz.

**18. Checkpointer vs Store** ([Ch 12](../learn/12-langchain-langgraph.md)). The checkpointer covers one thread (`thread_id`) and enables resuming, human-in-the-loop and time travel. The Store is long-term memory shared across threads. *Why it ranks here:* it was the largest Q&A cluster in that session.

**19. Idempotency** ([Ch 12](../learn/12-langchain-langgraph.md)). Use a key derived from the thread and the intent, so a retried booking never happens twice. *Why it ranks here:* it has a slide, a Mentimeter quiz and a notebook demo.

**20. Contract vs handoff** ([Ch 11](../learn/11-multi-agent.md)). A contract defines an agent's role, inputs, outputs, constraints and permissions. A handoff is a small structured message carrying the **minimum sufficient context**. *Why it ranks here:* it has two slides and about 8 Q&A rows.

**21. The five essential Responsible-AI controls** ([Ch 13](../learn/13-responsible-ai.md)). Guardrails (a permission, not an instruction), privacy, hallucination management, cost control (a cap, not an alert) and system documentation. *Why it ranks here:* it's the centrepiece of the agenda and the pre-read.

**22. Catastrophic forgetting** ([Ch 02](../learn/02-ft-data-training.md), [03](../learn/03-peft-adapters-prompts.md)). It's **silent**: the target task improves while general skills drop. The fixes are a low learning rate, replay, L2-SP or EWC, weight averaging, freezing layers, and PEFT. *Why it ranks here:* it was the biggest Q&A cluster in Chapter 02.

**23. Reading loss curves** ([Ch 02](../learn/02-ft-data-training.md)). Training loss falling while validation loss rises means overfitting. Both staying high means underfitting. Both flat means nothing is being learned. *Why it ranks here:* it has its own "ultimate guide" section.

**24. Learning rate, warm-up and schedules** ([Ch 02](../learn/02-ft-data-training.md)). The learning rate is the most important knob, and it should be small for fine-tuning. Warm up, then decay. Cosine is the default; WSD suits runs of unknown length. *Why it ranks here:* it took 14 slides.

**25. Training instability** ([Ch 02](../learn/02-ft-data-training.md)). A spike repairs itself; divergence doesn't. Gradient clipping is a seatbelt, and bf16 is the safer number format. *Why it ranks here:* it has a stability checklist.

**26. Data quality and curation** ([Ch 02](../learn/02-ft-data-training.md)). Quality beats quantity: LIMA used only about 1,000 examples. The pipeline is source → clean → dedupe → filter → review → format. *Why it ranks here:* it's in the agenda, the pre-read and the post-read.

**27. SFT vs instruction fine-tuning** ([Ch 01](../learn/01-ft-decisions.md)). Instruction fine-tuning is a kind of SFT that uses many tasks phrased as instructions, with the goal of following instructions in general. *Why it ranks here:* it has a comparison slide and two notebooks.

**28. Two independent taxonomy axes** ([Ch 01](../learn/01-ft-decisions.md)). One axis is the learning signal (unsupervised, supervised or alignment). The other is how many weights change (full or PEFT). *Why it ranks here:* the professor clarified it personally.

**29. The PEFT families** ([Ch 03](../learn/03-peft-adapters-prompts.md)). Additive (adapters), soft prompt, re-parameterization (LoRA) and selective (BitFit). *Why it ranks here:* it's on the concept map and came back in revision.

**30. Adapter structure** ([Ch 03](../learn/03-peft-adapters-prompts.md)). Down-project, apply a non-linearity, up-project, and add a residual connection, starting close to the identity. *Why it ranks here:* the "two requirements" slide.

**31. Houlsby vs Lin adapters** ([Ch 03](../learn/03-peft-adapters-prompts.md)). Houlsby adapters are sequential and add latency; Lin adapters are parallel, on a residual path. *Why it ranks here:* the limitations slide.

**32. Prefix vs prompt tuning** ([Ch 03](../learn/03-peft-adapters-prompts.md)). Prompt tuning works at the input layer only. Prefix tuning adds learned keys and values at **every** layer. *Why it ranks here:* it was the biggest source of confusion in that session's Q&A.

**33. HHH and its trade-offs** ([Ch 05](../learn/05-rlhf-dpo.md)). Helpful, honest and harmless often pull against each other. *Why it ranks here:* it's in the opening slides.

**34. SLM vs large model** ([Ch 06](../learn/06-revision-slm-guardrails.md)). A fine-tuned small model wins when the task is narrow and high-volume, has to stay private or run on-device, and good data exists. *Why it ranks here:* it has a pre-read table and post-read rules.

**35. Guardrails** ([Ch 06](../learn/06-revision-slm-guardrails.md)). Check **both input and output**, with content filters and PII redaction, because alignment isn't a security boundary. *Why it ranks here:* it was half the hands-on.

**36. Choosing the VLM paradigm** ([Ch 07](../learn/07-vlm-architectures.md), [08](../learn/08-vlm-applications.md)). For tags, use CLIP. For free text, LLaVA. For reading, pointing or documents, Qwen-VL. For a pixel outline, SAM. *Why it ranks here:* it's the anchor table of both lectures.

**37. Prompting per VLM task** ([Ch 08](../learn/08-vlm-applications.md)). Use scoped, fixed templates for captions, chain-of-thought for counting, and an explicit JSON schema for extraction, and allow the answer "not present". *Why it ranks here:* it's in every quick check.

**38. VLM failure modes** ([Ch 08](../learn/08-vlm-applications.md)). Hallucination, presupposition hallucination, compounding errors and hallucinated fields, all of which call for human review. *Why it ranks here:* every section has a pitfall slide.

**39. Which retrieval strategy when** ([Ch 09](../learn/09-multimodal-rag.md)). Plain text calls for text RAG. Clean tables on a budget call for a PDF parser. Charts and scans call for ColPali. Production systems use a hybrid. *Why it ranks here:* it has a dedicated slide.

**40. RAG pitfalls** ([Ch 09](../learn/09-multimodal-rag.md)). Retrieval and generation fail independently, and grounding **reduces** hallucination but doesn't eliminate it. *Why it ranks here:* it was in the lecture quiz.

**41. Text and image search basics** ([Ch 09](../learn/09-multimodal-rag.md)). Keyword vs semantic vs hybrid search, bi-encoders, ANN for speed, reranking, and CLIP's shared space for images. *Why it ranks here:* it had two sets of five quick-check questions.

**42. The six-rung ladder** ([Ch 10](../learn/10-rag-to-agents.md)). LLM → RAG → assistant → tool-using assistant → agent → multi-agent, with each rung adding one thing. *Why it ranks here:* it opens and closes the lecture.

**43. The agent loop and its guards** ([Ch 10](../learn/10-rag-to-agents.md)). Think → act → observe → remember. The loop must be bounded, observable and recoverable, and success must be verified in code. *Why it ranks here:* it has a slide plus guards in the notebook.

**44. ReAct vs Plan-and-Execute vs Reflection** ([Ch 10](../learn/10-rag-to-agents.md)). ReAct is the cheapest, Plan-and-Execute the most predictable, and Reflection costs the most tokens. *Why it ranks here:* it has a side-by-side slide.

**45. When to go multi-agent** ([Ch 11](../learn/11-multi-agent.md)). For specialisation, parallelism, context isolation or verification. Otherwise, stay with one agent. *Why it ranks here:* it's in the slides, the pre-read and the post-read.

**46. The coordination cycle and loop bounds** ([Ch 11](../learn/11-multi-agent.md)). Decompose → execute → attribute → reconcile → re-plan, bounded by a round limit, a token budget and a deadlock check, with escalation when stuck. *Why it ranks here:* it has dedicated slides with code.

**47. Supervisor vs peer-to-peer, and parallel vs sequential** ([Ch 11](../learn/11-multi-agent.md)). A supervisor gives accountability but is a single point of failure. Peer-to-peer suits debate. Run steps in parallel when they're independent. *Why it ranks here:* it's in the slides and the Q&A.

**48. Multi-agent safety** ([Ch 11](../learn/11-multi-agent.md)). Messages between agents are untrusted. Use least privilege, require human approval for writes, and fail closed. *Why it ranks here:* it has three slides.

**49. State and reducers** ([Ch 12](../learn/12-langchain-langgraph.md)). Nodes return partial updates, and reducers merge them: overwrite by default, or append with `add`. *Why it ranks here:* it has its own slide and many Q&A rows.

**50. `interrupt()` and `Command(resume=...)`** ([Ch 12](../learn/12-langchain-langgraph.md)). Pause for a human; on resume, **the node re-runs from the top**. *Why it ranks here:* it has a slide and a Mentimeter question.

**51. Retry vs hand back to the model** ([Ch 12](../learn/12-langchain-langgraph.md)). Retry infrastructure failures. Return failures about meaning to the model as observations. *Why it ranks here:* it has two slides and post-read coverage.

**52. LangChain vs LangGraph** ([Ch 12](../learn/12-langchain-langgraph.md)). A toolbox vs a flowchart. `create_agent` *is* a LangGraph graph. *Why it ranks here:* it's in the pre-read and the takeaway slide.

**53. Responsible vs trustworthy AI** ([Ch 13](../learn/13-responsible-ai.md)). Promise vs proof. *Why it ranks here:* it's in the agenda and the pre-read.

**54. The five leader questions** ([Ch 13](../learn/13-responsible-ai.md)). Does it work as intended? Who could it harm? Whose data is it? Can it be explained? Who is accountable? Answer each with evidence. *Why it ranks here:* the pre-read has a table of them.

**55. Provider vs deployer** ([Ch 13](../learn/13-responsible-ai.md)). The organisation that deploys the system answers for the outcome, even when the vendor's model got it wrong, as in the Air Canada case. *Why it ranks here:* it's repeated several times.

**56. Matching a case to its control** ([Ch 13](../learn/13-responsible-ai.md)). DPD maps to guardrails, Samsung to privacy, fake citations to hallucination management, a huge bill to cost control, and MyCity to documentation. *Why it ranks here:* every case in the session is paired with a control.

---

## 🟠 Medium: understand the idea

**Chapter 01.** Multi-task fine-tuning (a shared body with one head per task), curriculum learning (easy to hard), layer-wise vs block-wise vs progressive unfreezing, why full fine-tuning is costly, and the preview of the PEFT families.

**Chapter 02.** Batch size and gradient accumulation, imbalanced data and the accuracy paradox, splitting data without leakage (split first; use time and group splits), and domain shift.

**Chapter 03.** Soft prompts vs discrete prompts vs system prompts vs RAG; what SMoP, APT and IDPG each fix; adapter routing and AdapterFusion; and choosing a PEFT method.

**Chapter 04.** Intrinsic dimension, the r and α dials, multi-tenant hot-swapping, and block-wise quantization with outliers.

**Chapter 05.** The RL vocabulary for LLMs (state, action, policy), why policy gradients are noisy (reward-to-go, baseline, advantage), and PPO's "proximal" idea.

**Chapter 06.** The red-teaming loop (attack → find the gap → fix → re-test, and check for over-blocking), "alignment is not a security boundary", the 5-step SLM workflow, and loss curves again.

**Chapter 07.** SAM, which is Medium–High (it runs the encoder once, returns 3 masks, and was built with a data engine); ViT basics; LLaVA's data and upgrades; and the limits of each model.

**Chapter 08.** SAM's loss (focal plus Dice, Medium–High); composing models (a VLM finds the object, SAM cuts it out); datasets and metrics (COCO, VQAv2, DocVQA/ANLS, CIDEr, CLIPScore); and resolution as the main lever for documents.

**Chapter 09.** What Recall@K, nDCG@5 and ViDoRe mean; ColPali's bigger index and GPU cost; the three storage designs; chunking; and "should you fine-tune the LLM for a domain? Usually no".

**Chapter 10.** The three kinds of memory (conversation, task state, long-term), Agentic RAG, why the loop gets expensive, and why write tools need approval.

**Chapter 11.** MCP vs A2A (plus ANP and AGNTCY), sharing state in slices, what to log at each handoff, and the cost of extra agents.

**Chapter 12.** Super-steps and `recursion_limit`; nested timeouts and fallbacks; structured output, `@tool` and chains; time travel and streaming; `Command` handoffs and `Send` fan-out.

**Chapter 13.** The agentic shift (AI that *does* rather than *says*), the three 90-day priorities, governance as one system, and a one-line summary of each framework (NIST AI RMF, the EU AI Act, ISO/IEC 42001, DPDP, and India's AI Governance Guidelines).

---

## 🟢 Low: skim only if time allows

**Chapter 01:** the encoder vs decoder recap, and the facts about Med-PaLM and BloombergGPT.

**Chapter 02:** label-error detection with cleanlab, and the one-training-step walkthrough.

**Chapter 03:** the exact GPU tables, and paper names like Compacter and P-Tuning v2.

**Chapter 04:** gradient checkpointing details and the exact NF4 level values.

**Chapter 05:** where preference data comes from, and the value-head loss.

**Chapter 06:** library names, regex patterns and the Luhn check.

**Chapters 07 and 08:** variant tables, parameter counts and history dates (Ch 07); benchmark scores and case studies (Ch 08).

**Chapters 09–11:** survey details and EM training (Ch 09), the framework landscape (Ch 10), and the RL tangents (Ch 11).

**Chapter 12:** planning architectures beyond ReAct (ReWOO, LLMCompiler), and release dates.

**Chapter 13:** deeper framework details, case figures and dates.

---

## Suggested weighting for 24 questions

!!! warning "This is an estimate, not a known blueprint"
    We don't know how the paper is split. The estimate below assumes questions are spread roughly in proportion to the number of sessions per part, nudged by how much emphasis each part got. Use it to plan your revision time, not to predict the paper.

**Part A · Fine-tuning (Chapters 01–06): about 9–11 questions.** This part covers PEFT, alignment and the revision session, across 6 sessions, so it's likely to carry the most weight. It was also the only block with a dedicated revision session.

**Part B · Vision-language models and multimodal RAG (Chapters 07–09): about 5–6 questions.** Three sessions.

**Part C · Agents (Chapters 10–12): about 5–7 questions.** Three sessions, covering single agents, multi-agent systems and LangGraph.

**Part D · Responsible AI (Chapter 13): about 2–3 questions.** One session.

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
