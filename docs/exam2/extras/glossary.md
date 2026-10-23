# Glossary

**How to use this page.** Every term below is explained in one or two plain sentences, with the chapter where it's taught in brackets. Use it when a chapter mentions a word you've forgotten, or skim it the night before to check that every term "rings a bell". The definitions are short on purpose. For examples and the full story, go to the chapter.

---

## A

A2A (Agent-to-Agent protocol)
:   A protocol that lets one agent delegate work to another agent, even one run by another company. Agents describe themselves with **Agent Cards** and exchange **Tasks**. Compare MCP, which connects agents to tools. (Ch 11)

Accountability (deployer)
:   The organisation that puts an AI system in front of people answers for what it says and does, even if a vendor's model made the mistake (Air Canada). (Ch 13)

Accuracy paradox
:   On imbalanced data, a model can score very high accuracy just by always predicting the common class (e.g. "not fraud"). Use recall, precision, F1 or PR-AUC instead. (Ch 02)

Adapter
:   A small new module (down-project → non-linearity → up-project + residual) inserted into a frozen model. Only the adapter is trained. It starts close to "do nothing" (near identity). (Ch 03)

AdapterFusion
:   A way to combine several already-trained task adapters by learning how to mix their outputs. (Ch 03)

Additive PEFT
:   The PEFT family that **adds** new trainable parts (adapters, and in Ch 01's framing, soft prompts) while the original weights stay frozen. (Ch 01, Ch 03)

Advantage
:   In RL, how much better an action turned out than the average expected at that point. Positive = better than usual. It's the signal PPO actually uses. (Ch 05)

Agent
:   A system where the **model decides the next step at run time**, looping until the goal is reached. Compare workflow. (Ch 10)

Agent Card
:   An A2A agent's public description of what it can do and how to reach it. (Ch 11)

Agent loop
:   Think → Act → Observe → Remember, repeated until the goal is reached and verified. It must be bounded, observable and recoverable. (Ch 10)

Agentic RAG
:   RAG inside an agent loop: the agent may rewrite the query, retrieve again or skip retrieval, instead of retrieving exactly once. (Ch 10)

Agentic shift
:   The move from AI that *says* things to AI that *does* things (writes, modifies, approves, spends). It raises the stakes for controls. (Ch 13)

Alignment fine-tuning
:   Training a model to match human preferences and values, using feedback such as comparisons. RLHF and DPO are the main methods. (Ch 01, Ch 05)

α (alpha, LoRA)
:   The LoRA dial that sets how strongly the update counts. The side path is scaled by α / r. (Ch 04)

ANLS
:   The similarity-based metric used to score answers on DocVQA (document question answering). (Ch 08)

ANN (approximate nearest neighbour)
:   An index that finds *almost* the closest vectors much faster than exact search, with a small accuracy loss. Used for speed at scale, not accuracy. (Ch 09)

APT (Adaptive Prefix Tuning)
:   Prefix tuning where each layer gets its own prefix length. The course's finding: bottom layers usually need more. (Ch 03)

Assistant
:   The ladder rung that adds conversation context to RAG, so follow-up questions work. (Ch 10)

## B

B = 0 initialisation
:   LoRA starts the B matrix at zero, so B·A = 0 and the model initially behaves exactly like the original. (Ch 04)

Baseline (policy gradients)
:   A reference value subtracted from rewards to make the training signal less noisy. It leads to the idea of advantage. (Ch 05)

bf16 (bfloat16)
:   A 16-bit number format with a wide range. The stability checklist prefers it over fp16 because it doesn't overflow. (Ch 02, Ch 04)

β (beta, DPO)
:   The DPO dial that controls the pull towards the frozen reference. A larger β means staying closer. (Ch 05)

Bi-encoder
:   A retriever that embeds the query and each document **separately**, so documents can be pre-indexed. Fast; used for first-stage retrieval. (Ch 09)

BitFit
:   A selective PEFT method that trains only the existing bias terms and adds nothing new. (Ch 03)

Block-wise quantization
:   Quantizing weights in small blocks, each with its own scale, so a few outliers don't ruin precision for everything. (Ch 04)

BM25
:   The classic keyword-search scoring method. It's good for exact words, codes and IDs. (Ch 09)

Bradley–Terry model
:   The idea used to train a reward model from comparisons: the better answer should get the higher score. (Ch 05)

Bounded / observable / recoverable
:   The three properties every agent loop needs: a step and cost limit, a log of every step, and failures treated as observations rather than crashes. (Ch 10)

## C

Catastrophic forgetting
:   When fine-tuning on a new task overwrites general skills the model had before. It's **silent**: the target task looks fine. (Ch 02, Ch 03)

Chain-of-thought (CoT)
:   Prompting the model to reason step by step. In VLMs it helps with counting, comparison and multi-step visual questions. (Ch 08)

Chat template
:   The exact format (roles, special tokens) a chat model was trained on. Fine-tuning data must match it exactly. (Ch 02)

Checkpointer
:   LangGraph's saver that stores the full state after every super-step for one **`thread_id`**. It enables resume, human-in-the-loop and time travel. Example: `SqliteSaver`. (Ch 12)

CIDEr
:   A common headline metric for image captions, comparing them with reference captions. (Ch 08)

CLIP
:   A model with an image encoder and a text encoder trained to put matching pairs close together in one shared space. It scores similarity; it can't generate text. (Ch 07)

CLIPScore
:   A caption metric that uses CLIP similarity between the caption and the image, so no reference caption is needed. (Ch 08)

ColPali
:   A document retriever that embeds the **page image** directly as about 1,000 patch vectors (no OCR) and matches them with late interaction. (Ch 09)

`Command`
:   A LangGraph object used to resume after an interrupt (`Command(resume=...)`) or to hand off control to another node. (Ch 12)

Constrained decoding / structured outputs
:   Forcing the model's output to follow a schema (such as JSON). Try it before fine-tuning for format. (Ch 01)

Content filter
:   A guardrail that blocks harmful or off-topic input or output. (Ch 06)

Context isolation
:   Giving each agent only the context it needs. It's one of the four good reasons to go multi-agent. (Ch 11)

Continued pre-training (unsupervised)
:   More pre-training on raw domain text, so the model learns the domain's vocabulary before SFT. (Ch 01)

Contract (agent)
:   An agent's job description: role, inputs, outputs, constraints and permissions. (Ch 11)

Contrastive learning
:   Training by pulling matching pairs together and pushing non-matching pairs apart. In CLIP, the other pairs in the batch are the free negatives. (Ch 07)

Coordination cycle
:   The core multi-agent loop: decompose → execute → attribute → reconcile → re-plan. (Ch 11)

Cosine schedule
:   A learning-rate schedule that decays smoothly along a cosine curve after warm-up. It's the common default. (Ch 02)

Cost control
:   One of the five RAI controls: hard spending caps, loop limits and a named owner. "An alert is not a cap." (Ch 13)

`create_agent`
:   LangChain's quick way to build an agent. It's built on, and is, a LangGraph graph. (Ch 12)

Cross-attention adapter
:   Qwen-VL's connector: 256 learned queries attend over all image patches and produce exactly 256 visual tokens. (Ch 07)

Cross-encoder reranker
:   A model that reads the query and a document **together**. It's more accurate but slower, so it's only used to re-order a short list. (Ch 09)

Curriculum learning
:   Training on examples ordered from easy to hard. (Ch 01)

## D

Data engine (SAM)
:   SAM's self-improving labelling loop: the model helps annotators, its masks improve, and the process repeats. It produced the SA-1B dataset. (Ch 07)

Data Fiduciary / Data Principal (DPDP)
:   Under India's DPDP Act, the Data Principal is the person the data is about; the Data Fiduciary is the organisation deciding why and how it's processed, and it stays accountable. (Ch 13)

Data mixing / replay
:   Mixing some general or old-task data into fine-tuning to reduce catastrophic forgetting. (Ch 02)

Deadlock check
:   A multi-agent guard that stops early when the **same conflict is seen twice** and escalates. (Ch 11)

Dedupe
:   Removing duplicate examples from training data. (Ch 02)

Deployer
:   The organisation that uses an AI system in its product or process. Most organisations are deployers, and they answer for the outcome. Compare provider. (Ch 13)

Divergence
:   Loss climbing and not recovering (it may become NaN). Usually a too-high learning rate or precision problem. Compare spike. (Ch 02)

DocVQA
:   A benchmark of questions about document images, scored with ANLS. (Ch 08)

Domain shift
:   When real-world data differs from the training data, so performance drops. (Ch 02)

Double quantization
:   In QLoRA, quantizing the block **scaling constants** too, to save more memory. (Ch 04)

DPDP Act
:   India's Digital Personal Data Protection Act, 2023. The organisation stays responsible even when a vendor processes the data. (Ch 13)

DPO (Direct Preference Optimization)
:   Alignment that learns directly from (prompt, chosen, rejected) pairs: no reward model, no RL loop, just the policy and a frozen reference. (Ch 05)

## E

Early stopping
:   Stopping training when validation loss stops improving, to avoid overfitting. (Ch 02)

Escalate (to a human)
:   Handing a stuck or risky situation to a person, e.g. after a deadlock or budget limit. (Ch 11)

EU AI Act
:   The EU's risk-based AI law with separate duties for providers and deployers. Beyond the course: four risk tiers (unacceptable, high, limited, minimal). (Ch 13)

EWC (Elastic Weight Consolidation)
:   A penalty for moving away from the original weights, holding the weights most important for old skills more tightly. (Ch 02)

## F

Fail closed
:   If a result fails validation, don't merge it into shared state. Invalid results are never promoted. (Ch 11)

Fallback (`with_fallbacks`)
:   Switching to a backup model or tool when the first one fails. If nothing works, degrade **visibly**. (Ch 12)

Focal + Dice loss
:   The loss combination used to train SAM's masks. (Ch 08)

Full fine-tuning
:   Updating all of a model's weights. Maximum adaptation, but expensive in memory and prone to forgetting. (Ch 01, Ch 03)

Function calling (tool calling)
:   The model **emits** a structured request to use a tool; **your code executes** it and returns the result. (Ch 10)

## G

Gradient accumulation
:   Adding up gradients over several small micro-batches before one update, to get a large effective batch in limited memory. (Ch 02)

Gradient checkpointing
:   Storing only some activations and recomputing the rest during the backward pass. It trades compute for memory. (Ch 04)

Gradient clipping
:   Capping the size of the gradient on each step. A seatbelt for rare spikes; frequent clipping means the LR is too high. (Ch 02)

Graph RAG
:   Retrieval over a graph of linked entities, for multi-hop questions. (Ch 09)

Grounding (VLM)
:   Linking words to image regions, e.g. Qwen-VL outputting a box for "the red car". (Ch 07, Ch 08)

Grounding (RAG)
:   Telling the model to answer only from retrieved evidence. It reduces hallucination but doesn't eliminate it. (Ch 09)

Guardrails
:   Checks in code on input and output (content filters, PII redaction, permissions). In RAI terms, a permission, not an instruction, and re-tested after every change. (Ch 06, Ch 13)

## H

Hallucination
:   Confident output that isn't true or isn't in the source (or image). (Ch 08, Ch 09, Ch 13)

Hallucination management
:   One of the five RAI controls: treat output as a draft and check it in proportion to the stakes. (Ch 13)

Handoff
:   The small structured message passed from one agent to another, carrying **minimum sufficient context**. (Ch 11)

Hard negative mining
:   Training a retriever on near-miss wrong answers, to improve precision. (Ch 09)

HHH
:   Helpful, Honest, Harmless: the three alignment goals, which often conflict. (Ch 05)

Hot-swapping (LoRA)
:   Keeping one base model in memory and loading a different small LoRA adapter per request or client. (Ch 04)

Houlsby adapter
:   The original **sequential** adapter, placed in series after attention and after the feed-forward layer. It adds inference latency. (Ch 03)

Human-in-the-loop (HITL)
:   A person approves or edits before the system continues. Essential before irreversible actions. (Ch 10, Ch 11, Ch 12)

Hybrid search
:   Combining keyword (BM25) and dense (semantic) search. (Ch 09)

## I

Idempotency key
:   A key derived from thread + intent, so a retried booking or payment is recognised and doesn't happen twice. (Ch 12)

IDPG (Instance-Dependent Prompt Generation)
:   Generates a soft prompt **per input**, for tasks where inputs vary a lot. (Ch 03)

Instruction fine-tuning (IFT)
:   A kind of SFT on many tasks phrased as instructions, so the model learns to follow instructions in general. (Ch 01)

Intrinsic dimension
:   The idea that adapting a big model to a task needs only a small change, which is why a low-rank update works. (Ch 04)

`interrupt()`
:   LangGraph's way to pause a graph for a human. On resume, the **node re-runs from the top**. (Ch 12)

ISO/IEC 42001
:   An international, certifiable standard for an **AI management system**. (Ch 13)

IoU head (SAM)
:   The part of SAM that scores each of its three candidate masks. (Ch 07)

## K

KL penalty
:   In RLHF, a penalty for drifting far from the frozen reference model. It prevents reward hacking and drift. (Ch 05)

## L

LangChain
:   The **toolbox**: one interface for models, tools, messages, typed output, chains, middleware and `create_agent`. (Ch 12)

LangGraph
:   The **flowchart**: state, nodes, edges, checkpointer, Store, interrupts and streaming. (Ch 12)

Late interaction (MaxSim)
:   Matching with many vectors: each query token finds its best-matching page patch, and those best scores are added up. (Ch 09)

Layer-wise / progressive unfreezing
:   Training only some layers at a time. Layer-wise re-freezes earlier layers; progressive keeps them trainable. (Ch 01)

Leakage
:   Test information sneaking into training (e.g. scaling before splitting), giving falsely good results. (Ch 02)

Learning rate (LR)
:   The size of each weight update; the single most important hyperparameter. Fine-tuning uses a small one. (Ch 02)

Least privilege
:   Giving each agent only the tools and permissions it needs. (Ch 11)

LIMA
:   The finding that ~1,000 carefully curated examples can align a model well: quality beats quantity. (Ch 02)

Lin adapter
:   A **parallel / residual** adapter running beside a sub-layer, with a bank of task adapters. Less latency than Houlsby. (Ch 03)

Linear probe
:   Training a small classifier on top of frozen features. Compare zero-shot. (Ch 07)

LLaVA
:   A VLM made of a frozen CLIP encoder + a **projector** + an LLM, trained in two stages. (Ch 07)

Long-term memory
:   Facts kept across sessions (home airport, preferences). In LangGraph, the Store. (Ch 10, Ch 12)

LoRA (Low-Rank Adaptation)
:   Freeze W₀ and train a small side path ΔW = B·A, scaled by α / r. It can be merged back in afterwards. (Ch 04)

Loss curves
:   Plots of training and validation loss, used to diagnose overfitting, underfitting and no learning. (Ch 02, Ch 06)

## M

MCP (Model Context Protocol)
:   A standard for connecting agents to **tools and data**. Tools are discovered at run time (`list_tools()`) and used with `call_tool()`; the server owns the schema. (Ch 10, Ch 11)

Merge (LoRA)
:   Adding B·A into W₀ after training, so the model has its original shape and no extra latency. (Ch 04)

Middleware (LangChain)
:   Hooks around the agent loop (before / after the model, around tool calls) for things like PII redaction, approval gates and logging. (Ch 12)

Minimum sufficient context
:   Sending an agent only what it needs, not the full history. (Ch 11)

Multi-agent system
:   Several agents with separate roles working together. Use it only for specialisation, parallelism, context isolation or verification. (Ch 11)

Multi-task fine-tuning
:   Training one model on several related tasks with a shared body and one head per task. (Ch 01)

## N

nDCG@5
:   A ranking metric for the top 5 results that rewards putting the right result **near the very top** (1 = ideal). (Ch 09)

NF4 (4-bit NormalFloat)
:   QLoRA's 4-bit format for storing the frozen base weights. (Ch 04)

NIST AI RMF
:   The US voluntary AI risk framework with four functions: **Govern, Map, Measure, Manage**. (Ch 13)

## O

OCR
:   Optical character recognition: turning images of text into text. For clean bulk scans, a dedicated OCR pipeline beats a VLM. (Ch 08, Ch 09)

Oscillation guard
:   A check that catches an agent bouncing A → B → A. (Ch 12)

Over-blocking
:   Guardrails refusing harmless requests. Red-teaming should check for it too. (Ch 06)

Overfitting
:   Memorising the training data: training loss falls while validation loss rises. (Ch 02)

## P

Paged optimizer
:   In QLoRA, moving optimizer states to CPU RAM during memory spikes so training doesn't crash with OOM. (Ch 04)

Parallel vs sequential (agents)
:   Run agents in parallel when their sub-tasks are independent, and in sequence when one needs another's output. (Ch 11)

Parse-then-embed
:   Document RAG that first OCRs / parses pages into text, then embeds the text. Cheap, but brittle on charts and tables. (Ch 09)

PEFT (Parameter-Efficient Fine-Tuning)
:   Freezing most of the model and training only a tiny set of new or chosen parameters. (Ch 01, Ch 03, Ch 04)

Peer-to-peer (P2P)
:   A multi-agent design where agents talk directly. Good for debate and critique; harder to govern. (Ch 11)

PII (personally identifiable information)
:   Data that identifies a person (emails, phone numbers, card numbers). Detect and redact it in guardrails; regex + Luhn is basic, NER tools like Presidio are stronger. (Ch 06)

Plan-and-Execute
:   An agent loop shape that writes the whole plan first, then executes it, re-planning only the failed tail. Predictable cost. (Ch 10)

Policy (RL)
:   The model being trained to choose actions (tokens). (Ch 05)

PPO (Proximal Policy Optimization)
:   The RL algorithm in RLHF. It improves the model without changing it too much per update, and juggles four models. (Ch 05)

Prefix tuning
:   Learned vectors added to the keys and values of **every layer**, with the model frozen. (Ch 03)

Presupposition hallucination
:   A model accepting a false premise in the question ("What colour is the cat?" when there's no cat). Fix: allow "not present". (Ch 08)

Privacy (control)
:   One of the five RAI controls: rules on what data may go into which tool (Samsung case). (Ch 13)

Projector (LLaVA)
:   A linear or MLP layer that maps image patch features into the LLM's token space. (Ch 07)

Prompt engineering
:   Improving results by changing the instructions only. Always try it first. (Ch 01)

Prompt tuning
:   Learned "virtual tokens" added at the **input layer only**; the model stays frozen. (Ch 03)

Provider
:   The organisation that builds the AI model or tool. Compare deployer. (Ch 13)

## Q

QLoRA
:   LoRA on a base model stored in 4-bit NF4, plus double quantization, a paged optimizer and gradient checkpointing. A and B stay in BF16. (Ch 04)

Qwen-VL
:   A VLM with a cross-attention adapter (256 queries), grounding as text tokens and strong OCR / multi-image ability. (Ch 07)

## R

r (rank, LoRA)
:   The size of LoRA's side path: more capacity and more parameters as r grows. Use the smallest r that validates well. (Ch 04)

RAG (Retrieval-Augmented Generation)
:   Retrieving relevant documents at run time and giving them to the model, so it answers from fresh, private knowledge. (Ch 01, Ch 09, Ch 10)

Re-parameterization PEFT
:   The PEFT family that writes the weight change in a small low-rank form. LoRA and QLoRA. (Ch 01, Ch 03, Ch 04)

ReAct
:   An agent loop shape: think → act → observe, one step at a time. Cheapest and easiest to debug. (Ch 10)

Recall@K
:   Did the correct item appear anywhere in the top K results? (Ch 09)

`recursion_limit`
:   LangGraph's cap on **super-steps** (not tool calls). Hitting it raises `GraphRecursionError`. (Ch 12)

Red-teaming
:   Attacking your own system to find safety gaps: attack → find gap → fix → re-test. (Ch 06)

Reducer
:   A LangGraph rule for merging node updates into state. Overwrite is the default; `add` appends. Needed when parallel nodes write the same field. (Ch 12)

Reference model
:   A frozen copy of the starting model. RLHF's KL penalty and DPO's β both measure drift from it. (Ch 05)

Reflection
:   An agent loop shape: draft → critique → revise. Best quality, most tokens. (Ch 10)

Reranking
:   Re-scoring a short list of retrieved results with a slower, more accurate model, to improve precision. (Ch 09)

Responsible AI vs trustworthy AI
:   Responsible = the promise (principles); trustworthy = the proof (evidence). (Ch 13)

Retry vs hand back
:   Retry infrastructure failures (timeouts); hand meaning failures ("no flights") back to the model as observations. (Ch 12)

Reward hacking
:   The policy exploiting flaws in the reward model to get high scores for bad answers. (Ch 05)

Reward model
:   A model trained on human comparisons to score finished answers. It only scores; it doesn't generate. (Ch 05)

Reward-to-go
:   Crediting each action only with the rewards that came after it, to reduce noise in policy gradients. (Ch 05)

RLHF
:   Reinforcement Learning from Human Feedback: SFT → reward model → RL with PPO, plus a KL penalty. (Ch 05)

Round limit / token budget
:   Hard stops on a multi-agent loop after a set number of rounds or tokens. (Ch 11)

## S

SA-1B
:   The large segmentation dataset produced by SAM's data engine. (Ch 07)

SAM (Segment Anything Model)
:   A promptable segmentation model: heavy image encoder once, light decoder per prompt, three candidate masks. It knows *where*, not *what*. (Ch 07, Ch 08)

Selective PEFT
:   The PEFT family that trains a chosen **subset of existing** weights, e.g. BitFit's biases. (Ch 01, Ch 03)

`Send`
:   A LangGraph object for fan-out: sending work to several parallel node runs. (Ch 12)

SFT (Supervised Fine-Tuning)
:   Fine-tuning on labelled input → output pairs. (Ch 01)

SLM (Small Language Model)
:   A model of hundreds of millions to a few billion parameters. Fine-tuned, it often wins for narrow, high-volume, private or on-device tasks. (Ch 06)

SMoP (Sparse Mixture of Prompts)
:   Several short soft prompts plus a router that picks among them, instead of one long, costly prompt. (Ch 03)

Soft prompt
:   Learned vectors (not words) prepended to the input, trained by gradient descent. Ch 03 treats them as their own PEFT family. (Ch 03)

Spike (loss)
:   A short jump in loss that recovers by itself. Compare divergence. (Ch 02)

Store (LangGraph)
:   Long-term memory shared **across threads**, e.g. user preferences. Example: `SqliteStore`. Compare checkpointer. (Ch 12)

Super-step
:   One round of a LangGraph run in which a set of nodes executes. One tool round trip usually takes two. (Ch 12)

Supervisor
:   A multi-agent design with one coordinator routing the work. Clear accountability, but a single point of failure. (Ch 11)

System documentation (control)
:   One of the five RAI controls: a one-page record of owner, boundaries and how failures are detected (NYC MyCity case). (Ch 13)

## T

Table / structured RAG
:   Retrieval over tables or databases when exact values and calculations are needed. (Ch 09)

Task (A2A)
:   The unit of work one agent delegates to another in A2A. (Ch 11)

Task state (scratchpad)
:   Progress within one run: held fares, budget remaining, steps done. "Without task state there is no loop." (Ch 10)

Temperature (CLIP)
:   The scale on CLIP's similarity scores. Lower temperature = sharper predictions. (Ch 07)

`thread_id`
:   The identifier the checkpointer uses to keep one conversation's state separate from others. (Ch 12)

Time travel
:   Rewinding a LangGraph run to an earlier checkpoint and continuing from there. (Ch 12)

## V

Value model (critic)
:   In PPO, the model that predicts expected future reward from a partial answer; used to compute the advantage. (Ch 05)

ViDoRe
:   A benchmark for retrieving visually rich document pages, scored with nDCG@5. ColPali beat OCR pipelines on it. (Ch 09)

ViT (Vision Transformer)
:   The image encoder in most VLMs: it splits the image into patches and treats them like tokens. (Ch 07)

VLM (vision-language model)
:   A model that takes images and text as input. The template is: vision encoder → connector → output. (Ch 07)

VQA (visual question answering)
:   Answering questions about an image. (Ch 08)

## W

Warm-up
:   Starting with a tiny learning rate and ramping up over the first few percent of training, for stability. (Ch 02)

Weight averaging (WiSE-FT)
:   Blending fine-tuned and original weights in one pass to recover general skills. (Ch 02)

Workflow
:   A system where **you** fix the sequence of steps in advance. Cheaper and more predictable than an agent. (Ch 10, Ch 12)

WSD (Warmup–Stable–Decay)
:   A schedule that warms up, holds the peak, then decays sharply at the end. You can start the decay whenever you decide to stop. (Ch 02)

## Z

Zero-shot classification (CLIP)
:   Classifying images into labels never trained on, by comparing the image with text like "a photo of a {label}". (Ch 07)
