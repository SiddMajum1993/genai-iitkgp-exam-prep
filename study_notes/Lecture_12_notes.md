# Lecture 12 — Agentic RAG and Multi-Agent Orchestration
**Instructor:** Prof. Pawan Goyal

## TL;DR (5 bullets)
- Agentic RAG = retrieval as an LLM-chosen action inside Observe-Think-Act loop (NOT fixed pipeline); routing/shaping become dynamic tool calls
- ReAct = interleave Thoughts (reasoning) and Actions (tool use); Observations from tools ground reasoning against reality, prevent hallucination compounding
- Multi-agent split at ~20 tools: Triage (meta-planner) delegates to specialists (Infra/Codebase/HR), each runs isolated micro-ReAct, returns synthesized result to global state
- Planner (DAG of steps, expensive model) vs Executor (runs one step safely, cheap model); re-plan on failure; LangGraph = state machine for deterministic pathways over non-deterministic LLM
- Memory: Episodic (thread-local messages, checkpointed) vs Semantic/Long-term (vector DB, agents write learnings, auto RAG); Citations via Structured JSON or Agentic State logs

## Exam-relevant concepts

### Linear RAG vs Agentic RAG
- **Linear RAG:** Fixed pipeline `Embed Query → Retrieve Top-K → Generate Answer`; assumes perfect retrieval in one shot, complete context.
- **Agentic RAG:** Retrieval is a tool the LLM chooses; controller loop decides whether to retrieve, what to retrieve, when to re-retrieve, when done.
- **Key framing:** Routing & query shaping (Lecture 11's pre-processing steps) become dynamic decisions INSIDE the loop, conditioned on observations retrieved so far.
- **When it matters (exam trap):** "Multi-hop query, needs sequential logic" rules OUT linear RAG; requires Agentic loop.
- **Common confusion:** Agentic RAG ≠ just adding more tools; it's architectural — LLM-driven control flow vs hardcoded pipeline.

### Agent = LLM + Tools + Memory + Planning
- **LLM alone:** Next-token predictor, stateless.
- **Agent:** Architectural wrapper giving LLM agency — ability to act on environment via tools, remember via memory, plan via reasoning.
- **When it matters (exam trap):** "Agent with 30 tools and correct memory still inefficiently resolves incidents" → structural problem (tool hallucination), memory ≠ action-selection.
- **Common confusion:** More tools or more memory doesn't fix tool hallucination; need hierarchical split (multi-agent) or Planner/Executor.

### ReAct (Reasoning + Acting)
- **Definition:** Synergize reasoning and acting by interleaving Thoughts (internal state updates) and Actions (tool calls); Observations from environment ground reasoning.
- **Loop:** (x, t₁, a₁, o₁, t₂, a₂, o₂, ...) where x=user input, t=thought, a=action, o=observation.
- **Why it works:** If model makes factual error in pure CoT (Chain-of-Thought), error compounds through Step 10; ReAct pauses, takes action, gets ground truth observation, corrects before next thought.
- **When it matters (exam trap):** Distinguish ReAct (tool use grounded by observations) from pure CoT (closed-book reasoning).
- **Common confusion:** ReAct does use Observations (one exam option misstated this as false).

### Tool Calling mechanism
1. **Tool Definition:** Pass JSON schema describing function name, parameters, description to LLM.
2. **LLM Response:** LLM generates function call request (not raw text), specifying tool name + parameters.
3. **Execution:** Application logic executes function, returns result.
4. **Back to LLM:** Observation injected into next prompt.
- **Why descriptions matter:** LLM chooses tools based on descriptions, not just names; poor description → hallucinated calls.
- **When it matters (exam trap):** "LLM doesn't run code directly, it generates parameters for app to process and return."

### Single-Agent bottleneck (~20 tools)
- **Problem:** Past ~20 tools, single ReAct agent suffers tool hallucination — confused schemas, guessed parameters, wrong tool selection.
- **Not fixed by:** Bigger context window (needle-in-haystack), more memory (memory ≠ action-selection), faster model (same overload, just faster).
- **Fixed by:** Hierarchical planning — multi-agent (Triage + specialists) or Planner/Executor split.
- **When it matters (exam trap):** "30 tools, correctly working memory, still inefficient" → needs structural split (exam Q from transcript).
- **Common confusion:** Tool count limit is architectural (cognitive load on action-selection), independent of memory quality.

### Multi-Agent architecture: Triage + Specialists
- **Triage Agent:** Meta-planner, front door; analyzes query, delegates tasks; action space = `DelegateTo(SpecialistAgent, subtask)`.
- **Specialist Agents:** Domain experts (Infra, Codebase, HR/Policy); each has 3-5 highly relevant tools; runs isolated micro-ReAct; returns synthesized result.
- **Hierarchical ReAct:** Triage runs meta-ReAct loop at strategic level; specialists run micro-ReAct at execution level; global state cglobal tracks provenance.
- **When it matters (exam trap):** "Checkout API slowing, no deploy, feature flag changed 3 days ago" → Infra (latency) + config/flag owner, then synthesize; flat agent with many tools more likely to mis-select/hallucinate.
- **Common confusion:** Multi-agent ≠ just splitting work; it's isolating tool sets to prevent hallucination and context exhaustion.

### Planner vs Executor
- **Planner:** "Thinking" phase; breaks complex request into step-by-step DAG; uses expensive reasoning model (GPT-4); outputs recipe, does NOT run tools.
- **Executor:** "Doing" phase; takes one step from plan, formats tool call, executes safely; uses cheaper/faster model; focuses on delivery.
- **Why separate:**
  - Accuracy: reduces cognitive load on LLM
  - Cost: expensive model for planning, cheap for routine execution
  - Fault tolerance: if Step 2 fails, re-plan without redoing Step 1
- **Re-planning:** Executor feeds result back to Planner; Planner updates plan dynamically based on new info/failures.
- **When it matters (exam trap):** "Payment gateway failing after last night's update" → Triage (planner) generates strategy: Code Analysis → Log Retrieval → Synthesis; specialists (executors) run steps.
- **Common confusion:** Planner = Project Manager, Executor = Engineer (analogy from transcript table).

### State Machines & LangGraph
- **State Machine:** System in exactly one of finite states at any time; deterministic transitions between states based on inputs.
- **Why for AI:** LLM outputs non-deterministic; state machine forces deterministic pathways on top, prevents derailing.
- **LangGraph:** Library built on LangChain; models app as cyclic graph (NOT DAG); allows loops for "Think → Act → Observe → Think".
- **LangChain vs LangGraph:** LangChain = linear chains (DAG, one-way flow); LangGraph = cyclic graphs (loop-back decisions, multi-step reasoning).
- **When it matters (exam trap):** "How to build reliable agents without while loops" → State machines (LangGraph).
- **Common confusion:** DAGs only flow forward (LangChain); cycles essential for agentic loops (LangGraph).

### LangGraph core concepts
1. **State:** Shared TypedDict passed between nodes; nodes update (not overwrite); acts as "shared table" for multi-agent team.
2. **Nodes:** Python functions receiving State, performing work (LLM compute or tool execution), returning state update; e.g., TriageNode, InfraNode, CodebaseNode.
3. **Edges:** Determine which node runs next.
   - **Conditional Edges:** Routing function inspecting State to dynamically decide next step (e.g., read `next_agent` field, route to Infra/Codebase/FINISH).
4. **Compilation:** `app = graph.compile()` transforms declarative definitions into executable with thread checkpointing, streaming, pause/resume.

### State schema example (DevOps agent)
```python
state = {
  "messages": [HumanMessage("Check logs")],
  "next_agent": "infrastructure",  # triage sets this to route
  "sender": "triage"                # audit trail
}
```
- Triage reads query, updates `next_agent = "infrastructure"`.
- Conditional edge reads `next_agent`, routes to InfraNode.
- InfraNode calls Splunk tool, updates `messages` with findings, sets `next_agent = "triage"` (hands control back).
- Triage synthesizes final answer or plans next step.

### Human-in-the-Loop (HITL)
- **What:** Pause state machine at specific node, wait for human approval before transition.
- **Use case:** Infra agent proposes destructive action (restart Kubernetes pod); graph pauses at `approve_action` node, sends Slack to admin, waits for approval, then resumes.
- **Why mandatory:** Autonomous LLMs must not silently alter infrastructure.
- **Enabled by:** State serialization — graph can pause for hours/days without losing context.
- **When it matters (exam trap):** "Mandatory human approval before infra changes + auto self-heal for approved recurring fixes" → HITL gates only unmatched actions; approved fixes saved with scoped condition + timestamp, future matches skip HITL; recursion limit still bounds attempts (exam Q from transcript).
- **Common confusion:** HITL ≠ memory briefing human; it's enforcement gate where graph suspends until approval.

### Production caveats
1. **Infinite loops:** Triage asks Codebase, Codebase asks Triage, loop forever. Fix: strict `recursion_limit` (e.g., max 15 steps) in LangGraph.
2. **State bloat:** Every agent dialogue appends to `messages`, fills context window. Fix: Summarize node triggers when `len(messages) > 10`, compresses older dialogue.
3. **Memory contradiction:** Agent saves outdated fact ("Server X is primary DB"), then infra upgraded, agent hallucinates from stale memory. Fix: timestamp + `Update_Memory` tool to overwrite.
4. **Needle-in-haystack:** Even with 1M token context, injecting too much historical memory dilutes attention. Fix: retrieve only top-3 most relevant memories. Principle: "More context is not always better context."

### Memory types
- **Episodic (Short-Term):** Current conversation thread within State; list of Message objects; LangGraph thread checkpointing saves to DB (Postgres/SQLite) at every node execution.
  - **Use:** Resolves pronouns ("it" → DB server), maintains multi-turn debugging flow.
  - **Limit:** Grows indefinitely → context window saturation, exponential cost.
- **Semantic/Long-Term:** Facts persisted across sessions; stored in Vector DB; agents write learnings via `Save_To_Knowledge_Base(fact)` tool.
  - **Use:** Before planning, Triage silently searches "Have we seen similar error?" If yes, inject past solution, bypass expensive agent deployment.
  - **Self-healing:** Day 1 = 15 API calls to solve novel issue + save knowledge; Day 30 = 2 API calls (86% reduction) to retrieve + apply solution.
  - **Autonomous RAG:** Agents write documents themselves (institutional knowledge compounds).

### Provenance & Citations
- **Provenance:** Map every claim back to origin — specific document, log line, tool output, sub-agent action.
- **Why:** Verification, hallucination mitigation, accountability.
- **Enterprise principle:** "Uncited answer is unusable answer."

### Citation architectures (4 methods)
1. **Inline Prompting (Baseline):** System prompt instructs LLM to append `[Doc_1]` inline. Pros: low latency, streams naturally. Cons: highly prone to hallucination, relies on instruction-following.
2. **Structured Tool Forcing (Enterprise Standard):** Force LLM to output JSON `{"answer": "...", "citations": [{"doc_id": "...", "exact_quote": "..."}]}`. Pros: programmatic verification of exact quote in source before display. Cons: cannot stream, higher latency (wait for full JSON).
3. **Post-Hoc Attribution (Fact-Checker):** Fast LLM generates answer, then secondary NLI model injects citations retroactively by comparing against sources. Pros: highest accuracy, generator not distracted. Cons: doubles compute cost, heavy latency.
4. **Agentic State Provenance (LangGraph):** Citations derived from execution graph, not LLM text; State records `Message(content="Commit_88X...", name="Codebase_Agent")`. Pros: 100% deterministic, literal log of API call. Cons: only works for discrete tool calls, not synthesizing large paragraphs.

**When it matters (exam trap):** "Inline Prompting only, what two risks, minimal fix?" → Citation hallucination + stale memory cited as current; fix: verify cited IDs vs message history + timestamp/prune memory (exam Q from transcript).

**Common confusion:** Inline = fast but unreliable; Structured = verify exact quote; Post-Hoc = most accurate but expensive; Agentic State = deterministic but tool-call-only.

### UI/UX of citations
- Don't just show `[1]`; hyperlink directly to live Splunk dashboards, GitHub PRs for immediate verification.
- Verification step: lightweight code checks every cited ID exists in State's message history (never trust LLM blindly).

### Example: Multi-agent citation output
"The payment container failed due to Out of Memory error [Infra Agent: Splunk-pod-1, Line 452]. This aligns with yesterday's commit increasing cache size [Codebase Agent: Git Commit 8f4b2a]."
→ Human engineer has immediate confidence + direct links to verify both specialists.

## Agentic RAG anatomy

### Observe-Think-Act loop
- **Observe:** Receive user input or environment state (tool outputs).
- **Think:** LLM decides next step based on prompts, tools, observations.
- **Act:** Execute tool (query DB, call API, delegate to specialist).
- Loop continues until task complete or recursion limit hit.

### Key differences from Linear RAG
| Aspect | Linear RAG | Agentic RAG |
|--------|-----------|-------------|
| Retrieval | Fixed, one-shot | LLM-chosen action, multi-step |
| Routing | Static pre-processing (rule/embedding/LLM) | Dynamic tool call inside loop |
| Query shaping | Explicit rewriting before retrieval | Implicit planning (plan = shaped directive) |
| Reflection | Cannot loop back | Can re-retrieve, re-plan, iterate |
| Control flow | DAG (one-way) | Cyclic graph (loop-back) |
| Memory | None or simple context | Episodic (thread) + Semantic (vector DB) |
| Citations | Post-hoc or inline | State-based provenance |

### Tool use
- Planner selects tools; Executor formats and invokes.
- Observations inject ground truth, prevent hallucination compounding.
- Tool descriptions critical (LLM chooses by description, not name).

### Planning
- **Single-Agent ReAct:** One agent, ~20 tools max, one loop.
- **Hierarchical ReAct:** Triage meta-plans at strategic level; specialists micro-plan at execution level; global state tracks all.
- **Planner/Executor split:** Planner outputs DAG, Executor runs steps, re-plan on failure.

### Memory integration
- **Episodic:** Resolves pronouns, maintains multi-turn flow, checkpointed.
- **Semantic:** Autonomous learning, self-healing, compound institutional knowledge.
- **Write:** `Save_To_Knowledge_Base(fact)` after solving novel issue.
- **Read:** Silent search before planning; if similar error found, inject past solution.

## Likely MCQ angles

1. **Linear vs Agentic RAG:** "Multi-hop query, sequential logic" → Agentic (NOT linear which assumes one-shot retrieval).

2. **Tool hallucination threshold:** "30 tools, correct memory, still inefficient" → Structural issue, needs multi-agent or Planner/Executor split (NOT more memory, NOT bigger context).

3. **ReAct vs CoT:** ReAct interleaves reasoning and actions, Observations ground reasoning (prevents hallucination compounding in closed-book CoT).

4. **Multi-agent routing:** "Checkout API slowing, no deploy, feature flag changed" → Infra (latency) + config/flag owner; flat agent with many tools more likely to mis-select/hallucinate params.

5. **HITL + self-healing:** "Mandatory approval + auto self-heal for recurring fixes" → HITL gates only unmatched actions; approved fixes saved with scoped condition + timestamp, future matches skip HITL; recursion limit bounds attempts.

6. **Planner vs Executor:** Planner = thinking/DAG/expensive model/no tools; Executor = doing/runs one step/cheap model/safety.

7. **LangGraph vs LangChain:** LangChain = DAG (one-way); LangGraph = cyclic graph (loop-back for agentic loops).

8. **State bloat + memory contradiction:** "Resolution time rising, API costs spiking, incident volume flat" → Memory contradiction (stale facts) + growing messages (token inflation); diagnostic: check failures vs memory age, message-length growth over time.

9. **Citation architectures:** Inline = fast/unreliable; Structured = verify exact quote; Post-Hoc = most accurate/expensive; Agentic State = deterministic/tool-only. "Inline only, what risks, minimal fix?" → Citation hallucination + stale memory; fix: verify IDs + timestamp/prune.

10. **Episodic vs Semantic memory:** Episodic = thread-local, checkpointed, resolves pronouns; Semantic = vector DB, cross-session, autonomous learning, self-healing.

11. **State Machine rationale:** Force deterministic pathways on non-deterministic LLM outputs; prevent derailing/infinite loops.

12. **Recursion limit:** Kill runaway loops (agents arguing in loop burn thousands of tokens/minute); typical max 15 steps.

13. **Specialist isolation:** Each specialist has 3-5 highly relevant tools, runs micro-ReAct, returns synthesized result to Triage; prevents tool hallucination and context exhaustion.

14. **Needle-in-haystack:** "Even 1M token context, too much historical memory dilutes attention" → retrieve only top-3 relevant memories; more context ≠ better context.

15. **Self-healing pattern:** Day 1 = 15 calls (solve + save); Day 30 = 2 calls (retrieve + apply) = 86% reduction.

## One-line flashcards

- Q: Linear RAG assumption? → A: Perfect retrieval in one shot, complete context; fixed pipeline
- Q: Agentic RAG retrieval? → A: LLM-chosen action inside Observe-Think-Act loop, multi-step, can re-retrieve
- Q: Agent = ? → A: LLM + Tools + Memory + Planning (architectural wrapper for agency)
- Q: ReAct formula? → A: (x, t₁, a₁, o₁, t₂, a₂, o₂, ...) where t=thought, a=action, o=observation
- Q: ReAct vs CoT difference? → A: ReAct grounds reasoning with tool observations; CoT closed-book, errors compound
- Q: Tool calling flow? → A: JSON schema → LLM generates call → App executes → Observation back to LLM
- Q: Single-agent tool hallucination threshold? → A: ~20 tools; past that, confused schemas/guessed params
- Q: Tool hallucination fixed by memory? → A: NO, memory ≠ action-selection; need structural split (multi-agent/Planner-Executor)
- Q: Multi-agent Triage action space? → A: DelegateTo(SpecialistAgent, subtask), NOT raw API tools
- Q: Specialist agents purpose? → A: 3-5 relevant tools each, isolated micro-ReAct, prevent hallucination/context exhaustion
- Q: Planner role? → A: Thinking phase, breaks request into DAG, expensive model, outputs recipe, no tool execution
- Q: Executor role? → A: Doing phase, runs one step safely, cheap model, focuses on delivery
- Q: Why separate Planner/Executor? → A: Accuracy (reduce load), cost (expensive for plan, cheap for exec), fault tolerance (re-plan Step 2 without redoing Step 1)
- Q: Re-planning trigger? → A: Executor failure or new info; Planner updates plan dynamically
- Q: State Machine for AI why? → A: Force deterministic pathways on non-deterministic LLM outputs, prevent derailing
- Q: LangChain vs LangGraph? → A: LangChain = DAG (one-way); LangGraph = cyclic graph (loop-back for agentic loops)
- Q: LangGraph State? → A: Shared TypedDict, nodes update (not overwrite), "shared table" for agents
- Q: LangGraph Nodes? → A: Python functions receiving State, doing work (LLM/tool), returning update
- Q: LangGraph Conditional Edge? → A: Routing function inspecting State to decide next node (e.g., read next_agent field)
- Q: LangGraph compile()? → A: Transform declarative graph into executable with checkpointing, streaming, pause/resume
- Q: HITL definition? → A: Pause state machine at node, wait human approval before transition
- Q: HITL use case? → A: Destructive action (restart pod); graph pauses, Slack admin, waits approval, resumes
- Q: HITL + self-healing? → A: HITL gates only unmatched actions; approved fixes saved with scoped condition + timestamp, future matches skip HITL
- Q: Infinite loop fix? → A: Strict recursion_limit (e.g., max 15 steps) in LangGraph
- Q: State bloat fix? → A: Summarize node triggers when len(messages) > 10, compresses older dialogue
- Q: Memory contradiction fix? → A: Timestamp + Update_Memory tool to overwrite stale facts
- Q: Needle-in-haystack fix? → A: Retrieve only top-3 most relevant memories; more context ≠ better
- Q: Episodic memory? → A: Thread-local messages, checkpointed, resolves pronouns, multi-turn flow
- Q: Episodic memory limit? → A: Grows indefinitely → context saturation, exponential cost
- Q: Semantic/Long-term memory? → A: Vector DB, cross-session facts, agents write learnings, autonomous RAG
- Q: Self-healing Day 1 vs Day 30? → A: Day 1 = 15 calls (solve + save); Day 30 = 2 calls (retrieve + apply) = 86% reduction
- Q: Memory write tool? → A: Save_To_Knowledge_Base(fact) after solving novel issue
- Q: Memory read pattern? → A: Triage silent search "similar error?" before planning; if yes, inject solution, bypass expensive agents
- Q: Provenance definition? → A: Map every claim back to specific doc/log/tool/sub-agent; establish chain of custody
- Q: Why citations matter? → A: Verification, hallucination mitigation, accountability; "uncited answer = unusable"
- Q: Inline Prompting? → A: System prompt instructs LLM to append [Doc_1]; fast but highly prone to hallucination
- Q: Structured Tool Forcing? → A: Force JSON {"answer": "...", "citations": [{"doc_id": "...", "exact_quote": "..."}]}; verify exact quote in source
- Q: Post-Hoc Attribution? → A: Fast LLM generates, then secondary NLI model injects citations retroactively; highest accuracy, doubles cost
- Q: Agentic State Provenance? → A: Citations from execution graph logs, not LLM text; 100% deterministic, literal API call log
- Q: Citation verification step? → A: Lightweight code checks every cited ID exists in State message history (never trust LLM blindly)
- Q: Multi-agent citation example? → A: "OOM error [Infra: Splunk-pod-1, Line 452]. Yesterday's commit [Codebase: Git 8f4b2a]."
- Q: Citation UI/UX? → A: Hyperlink directly to live Splunk/GitHub PRs for immediate verification, not just [1]
- Q: State bloat + memory contradiction diagnostic? → A: Check failures vs memory age, message-length growth over time (exam Q)
- Q: 30 tools, correct memory, still inefficient cause? → A: Structural (tool hallucination), needs multi-agent or Planner/Executor split (exam Q)
- Q: Checkout API slowing, flag changed, no deploy routing? → A: Infra (latency) + config/flag owner, then synthesize; flat agent more likely to mis-select (exam Q)
- Q: Inline Prompting only, two risks, minimal fix? → A: Citation hallucination + stale memory cited; fix: verify IDs vs message history + timestamp/prune (exam Q)
