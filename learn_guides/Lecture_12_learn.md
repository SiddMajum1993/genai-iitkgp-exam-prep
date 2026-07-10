# Lecture 12 — Agentic RAG: From Pipelines to Autonomous Problem Solving
**Estimated read time:** 30–40 min · **Difficulty:** Core to Advanced

## What this lecture is about

This lecture transforms your understanding of RAG from a passive retrieval pipeline into an active, autonomous problem-solving system. In Lecture 11, you learned how to make RAG fast and secure at scale. But those systems are still fundamentally linear: embed query → retrieve once → generate answer. They cannot handle multi-hop questions like "Find the error in the payment container, check if it is related to yesterday's auth module commit, and if so, draft a Slack update for the QA team." Linear RAG would try to embed this entire prompt at once and return a confused blend of results.

Agentic RAG solves this by giving the LLM control. Instead of hardcoding the steps, you equip the LLM with tools (functions to query databases, call APIs, search logs) and let it decide autonomously: what tool to call, when to call it, and what to do with the results. The system enters an Observe-Think-Act loop — exactly how a human engineer debugs an incident. This lecture covers the ReAct framework, multi-agent architectures, LangGraph for orchestration, memory management, human-in-the-loop safety gates, and provenance/citations for trust. By the end, you will understand how to build a DevOps agent that can autonomously investigate incidents, remember past solutions, and cite its sources.

## Prerequisites

Before diving in, you should understand:
- The linear RAG pipeline (Lecture 11)
- What vector search does
- Basic Python functions
- What an API call is
- The concept of a state machine (you will learn more here)

If you have ever used ChatGPT's Code Interpreter or tried to build a chatbot that calls external APIs, you have seen glimpses of agentic behavior. This lecture formalizes it.

---

## Concept 1: The Failure of Linear RAG at Multi-Hop Queries

### The problem it solves

Linear RAG assumes that one retrieval step gives you all the context you need. But real-world problems often require chaining multiple searches. "Did the latency spike happen in both Mumbai and US-East Vertex AI deployments?" This question requires two separate retrieval calls (Mumbai logs + US-East logs), then a comparison. A linear pipeline cannot loop back and retrieve again after seeing the first result.

### The intuition (analogy first)

Imagine you are a detective investigating a crime. You find a receipt at the scene. A linear pipeline would stop there and try to solve the case with just that receipt. But you, a human, take the receipt, look up the store, retrieve the purchase time, then cross-reference that time with security footage. That is a multi-hop investigation. Linear RAG cannot do this because it cannot pause, evaluate intermediate results, and decide what to search next.

### Worked example (the complex task from the lecture)

**User query:** "Find the error in the payment container, check if it is related to yesterday's auth module commit, and if so, draft a Slack update for the QA team."

**What linear RAG tries to do:**
1. Embed the entire query: "Find error payment container check auth module commit draft Slack QA"
2. Vector search returns a blend of results from logs, codebase, and HR docs (because "QA team" might match HR org charts)
3. LLM tries to generate an answer from this hodgepodge and hallucinates

**Why it fails:**
- **Vector dilution:** The embedding mixes keywords from three different domains (logs, code, communication), so the vector DB is confused
- **No sequential logic:** The system does not know it needs to find the error FIRST, then search commits, then draft a message. It tries to do everything at once.

### The necessary shift

You cannot hardcode all possible sequences of steps with if/else trees. Instead, hand control over to the LLM. Make retrieval a tool the LLM can choose to call, rather than a fixed stage in a pipeline.

### Exam angle

Expect questions that describe a multi-hop scenario (e.g., "payment pod crashed, then check auth commit, then notify team") and ask why linear RAG fails. The answer involves "cannot loop back," "no intermediate planning," or "vector dilution."

---

## Concept 2: What Makes a System "Agentic" — The Four Components

### The definition

An agent is an LLM wrapped in an architectural pattern that gives it agency — the ability to act on its environment. The four components are:

1. **LLM (the brain):** Decides what to do next based on instructions and observations.
2. **Tools (the hands):** Functions the LLM can call (query a database, call an API, search logs, send a Slack message).
3. **Memory (the notebook):** Stores past conversations and learned facts so the agent does not start blank every time.
4. **Planning (the strategy):** The process by which the LLM breaks down a complex goal into steps.

Without tools, the LLM is just a chatbot. Without memory, it forgets context between turns. Without planning, it cannot handle multi-step tasks.

### The core loop: Observe → Think → Act

This is the heartbeat of an agent:

1. **Observe:** The agent receives input (user query, or the result of the previous tool call).
2. **Think:** The LLM generates a thought (internal reasoning trace) about what to do next.
3. **Act:** The LLM calls a tool (action). The tool executes and returns an observation.
4. Loop back to step 1 with the new observation.

This loop continues until the agent decides it has enough information to generate a final answer.

### Worked example (DevOps agent debugging a failing auth service)

**User input:** "The auth-service is failing."

**Turn 1:**
- Observe: "auth-service is failing"
- Think (t₁): "I need to check the error logs for auth-service to see why it is failing."
- Act (a₁): Call `fetch_server_logs(service="auth-service", timeframe="1h")`
- Observation (o₁): API returns `"ConnectionTimeout: Cannot reach db-primary"`

**Turn 2:**
- Observe: o₁ (the timeout error)
- Think (t₂): "The issue is not the auth-service code, it is the database connection. I need to check db-primary status."
- Act (a₂): Call `query_infrastructure(target="db-primary")`
- Observation (o₂): "db-primary is down, failover in progress"

**Turn 3:**
- Observe: o₂
- Think (t₃): "The root cause is identified: db-primary is down. I can now answer the user."
- Act (a₃): Generate final answer: "The auth-service is failing because db-primary is down. Failover is in progress."

Notice how each observation informs the next thought, which informs the next action. This is autonomous problem-solving.

### The context vector (formal definition)

The agent maintains a context vector c that grows with each turn:

c = (x, t₁, a₁, o₁, t₂, a₂, o₂, ..., tₙ, aₙ, oₙ)

Where:
- x = initial user input
- tᵢ = thought at turn i
- aᵢ = action at turn i
- oᵢ = observation at turn i

This entire history is passed to the LLM at each turn, so it can reason about what has happened so far.

### Common misunderstandings

- **"The LLM runs the tools directly."** No. The LLM generates a request to call a tool (with parameters), and the application code executes it. The LLM just decides WHAT to call and WITH WHAT ARGUMENTS.
- **"Agents are new."** The concept is not new (ReAct paper is from 2023, but tool use is older). What is new is the reliability and scale at which modern LLMs can do this.

### Exam angle

Expect a question like: "What are the four components of an agentic system?" Answer: LLM, Tools, Memory, Planning. Or: "What is the core loop?" Answer: Observe-Think-Act.

---

## Concept 3: Tool Calling — How the LLM Requests Functions

### The problem it solves

The LLM is a text model. It cannot make HTTP requests, query databases, or run Python code directly. Tool calling is the bridge: the LLM outputs a structured request in JSON, the application code parses it, executes the function, and feeds the result back to the LLM.

### How it works (step by step)

**Step 1: Define tools as JSON schemas**

You define each tool as a function with a name, description, and parameter schema:

```json
{
  "name": "fetch_server_logs",
  "description": "Retrieve recent logs to diagnose failures, errors, or performance issues.",
  "parameters": {
    "type": "object",
    "properties": {
      "server_id": {"type": "string", "description": "ID of the server"},
      "minutes_ago": {"type": "integer", "description": "How many minutes back to search"}
    },
    "required": ["server_id", "minutes_ago"]
  }
}
```

**Step 2: Pass the tool schemas to the LLM**

When you call the LLM, you include these schemas in the prompt (or via the API's tools parameter for models like GPT-4, Claude, Gemini).

**Step 3: The LLM generates a tool call**

Instead of generating text, the LLM outputs:

```json
{
  "tool": "fetch_server_logs",
  "arguments": {
    "server_id": "db-12",
    "minutes_ago": 10
  }
}
```

**Step 4: Your application code executes the tool**

You parse the JSON, call the actual Python function `fetch_server_logs("db-12", 10)`, which queries your Splunk API or log store.

**Step 5: Feed the result back to the LLM**

The function returns: `"Database connection pool exhausted"`

You add this as an observation to the context and call the LLM again. It now sees:

- Previous thought: "I need logs"
- Previous action: `fetch_server_logs(db-12, 10)`
- Observation: "Database connection pool exhausted"

The LLM now thinks: "Aha, it is a connectivity issue, not a code failure."

### Worked example (troubleshooting a 500 error)

**User:** "Investigate 500 errors on db-12"

**LLM thinks:** "I should fetch recent logs to see what is happening."

**LLM outputs:**

```json
{"tool": "fetch_server_logs", "arguments": {"server_id": "db-12", "minutes_ago": 10}}
```

**App executes:** Calls Splunk API, gets back `"Error: Database connection pool exhausted"`

**App feeds back to LLM:** "Observation: Database connection pool exhausted"

**LLM thinks:** "This is a resource issue. I should check if the connection pool limit was recently changed."

**LLM outputs:**

```json
{"tool": "query_config_history", "arguments": {"service": "db-12", "config_key": "max_connections"}}
```

...and so on.

### Why descriptions matter

The LLM chooses tools based on their descriptions, not just names. A vague description like "get logs" is less useful than "Retrieve recent logs to diagnose failures, errors, or performance issues, including stack traces and HTTP status codes."

### Common misunderstandings

- **"The LLM executes the code."** No. The LLM generates a request. Your application code does the execution.
- **"Tool schemas must be perfect."** They should be clear, but the LLM is quite good at inferring intent even with imperfect schemas.

### Exam angle

Expect a question about how tool calling works: "What does the LLM output when it wants to call a tool?" Answer: A JSON object with the tool name and arguments.

---

## Concept 4: The Single-Agent Bottleneck — Why One Agent Cannot Handle 30+ Tools

### The problem it solves

As your enterprise system scales, you might have 30, 50, or 100 tools (Splunk, GitHub, Datadog, Jira, Slack, HR database, etc.). If you give all of them to a single agent, the agent suffers from tool hallucination: it confuses tool schemas, mixes up parameters, or calls the wrong tool entirely. This is because the LLM's context window is crowded with 50 tool definitions, and the cognitive load is too high.

### The intuition (analogy first)

Imagine you are a general contractor, and every morning you give your single worker a list of 50 different tasks: plumbing, electrical, carpentry, painting, roofing, HVAC, landscaping, etc. The worker will get confused, use the wrong tools, and make mistakes. Instead, you hire specialists: a plumber, an electrician, a carpenter. Each specialist knows 5–10 tools deeply. When a job comes in, you (the general contractor) route it to the right specialist.

### The solution: Multi-agent systems

Instead of one agent with 50 tools, you create:
- **Triage Agent (the router):** Receives the user query, decides which specialist to call, coordinates the workflow.
- **Specialist Agents:** Each has 5–10 tools for a narrow domain:
  - **Infrastructure Agent:** Splunk, Datadog, Kubernetes API
  - **Codebase Agent:** GitHub API, code search
  - **HR/Policy Agent:** HR database, runbook vector search

The Triage Agent does NOT execute tools directly. It delegates to specialists.

### Worked example (payment gateway failing after last night's update)

**User query:** "Payment gateway failing after last night's update."

**Triage Agent (the planner):**
- Think: "This involves code changes and live logs. I need the Codebase Agent to fetch the latest commit, and the Infrastructure Agent to pull error logs, then I will synthesize."
- Act (delegation): "Delegate to Codebase Agent: 'Fetch yesterday's commit diff for payment-gateway repo.'"

**Codebase Agent (the executor):**
- Act: Call `github.get_commit_diff(repo="payment-gateway", sha="latest")`
- Observation: Returns commit a1b2c3d: "Added strict rate-limiting logic"
- Return to Triage Agent: "Commit a1b2c3d added rate-limiting."

**Triage Agent:**
- Think: "Got the code change. Now I need the error logs."
- Act (delegation): "Delegate to Infrastructure Agent: 'Pull error logs for pay-pod-1 over the last 12 hours.'"

**Infrastructure Agent:**
- Act: Call `splunk.query_logs(service="pay-pod-1", level="ERROR")`
- Observation: "TimeoutException at Auth.py line 42"
- Return to Triage Agent: "Timeout at Auth.py:42"

**Triage Agent:**
- Think: "The timeout correlates with the rate-limiting commit. I now have enough to answer."
- Generate final answer: "The payment gateway is failing because last night's commit (a1b2c3d) added strict rate-limiting, which is causing timeouts in Auth.py line 42."

Notice: The Triage Agent never touched raw APIs. It delegated to specialists, received synthesized observations, and coordinated the answer.

### The formal framing (hierarchical ReAct)

The Triage Agent operates at a meta-level. Its action space A is not raw tools but delegations to specialist agents:

A = {DelegateTo(InfraAgent, task), DelegateTo(CodebaseAgent, task), DelegateTo(HRAgent, task), GenerateFinalAnswer}

Each specialist runs its own micro-ReAct loop (Observe-Think-Act) to accomplish its narrow task, then returns a synthesized observation to the Triage Agent.

### Common misunderstandings

- **"Multi-agent systems are more complex, so they are worse."** Not true. They are more complex to build, but they are MORE RELIABLE at scale because each agent has lower cognitive load.
- **"The Triage Agent is smarter than the specialists."** No. The Triage Agent just coordinates. The specialists do the actual work.

### Exam angle

Expect a question like: "A single flat ReAct agent with 30+ tools is experiencing tool hallucination. What is the structural fix?" Answer: Split into a Triage Agent + specialist agents. Memory alone does not fix action-selection confusion.

---

## Concept 5: Planner vs. Executor — Separating Thinking from Doing

### The problem it solves

Even within an agent, mixing planning (deciding what to do) and execution (actually doing it) in the same LLM call leads to poor reasoning. Planners need to think broadly and strategically. Executors need to format precise tool calls and handle errors.

### The intuition (analogy first)

A project manager (Planner) decides: "We need to query the database, analyze the results, then email the report." They do not write the SQL query themselves. They hand that step to an engineer (Executor), who writes the exact SQL, runs it, handles errors, and reports back: "Query returned 150 rows" or "Query failed: syntax error." The manager then decides the next step based on that feedback.

### How it works (step by step)

**Planner (expensive reasoning model like GPT-4):**
1. Receives user goal: "Investigate payment gateway failure after last night's update."
2. Generates a high-level plan (a DAG of steps):
   - Step 1: "Fetch latest commit diff from payment-gateway repo"
   - Step 2: "Pull error logs for pay-pod-1 over last 12 hours"
   - Step 3: "Synthesize findings to identify root cause"

**Executor (cheaper, faster model like GPT-3.5 or Haiku):**
1. Receives Step 1 from the Planner.
2. Formats the tool call: `github.get_commit_diff(repo="payment-gateway", sha="latest")`
3. Executes the tool, gets the result: "Commit a1b2c3d: Added strict rate-limiting"
4. Returns observation to the orchestrator.

**Orchestrator (the middleware):**
1. Receives observation from Executor.
2. Feeds it back to the Planner: "Step 1 complete. Result: commit a1b2c3d added rate-limiting."
3. Planner updates the plan: "Now proceed to Step 2."
4. Orchestrator dispatches Step 2 to the Executor.

**If Step 2 fails (e.g., network timeout):**
- Executor returns: "Error: network timeout when querying Splunk"
- Orchestrator feeds this back to the Planner
- Planner re-plans: "Retry Step 2 with a shorter time window" or "Skip logs and proceed with what we have"

### Why separate them

1. **Accuracy:** Planners can focus on strategy without worrying about exact API syntax. Executors focus on safe, precise tool calls.
2. **Cost efficiency:** Use expensive models (GPT-4, Opus) only for planning. Use cheap models (Haiku, GPT-3.5) for execution.
3. **Fault tolerance:** If Step 2 fails, you do not re-plan Steps 1 and 3. Just retry Step 2.

### Worked example (network failure during log retrieval)

**Planner generates:**
- Step 1: Fetch code changes
- Step 2: Fetch logs
- Step 3: Synthesize

**Executor completes Step 1 successfully.**

**Executor tries Step 2, fails:** "Splunk API unreachable"

**Orchestrator feeds failure back to Planner.**

**Planner re-plans:**
- Step 2a: "Check network status"
- Step 2b: "If network is up, retry Splunk with shorter timeout"
- Step 3: Synthesize

Notice the dynamic adaptation. A pure linear pipeline would crash and require manual intervention.

### Common misunderstandings

- **"Planner-Executor is the same as multi-agent."** Not exactly. Planner-Executor is a within-agent pattern. Multi-agent is about splitting domains (Infra vs. Codebase). You can combine both: a Triage Agent (Planner) delegates to specialist agents (Executors).

### Exam angle

Expect a question like: "Why separate planning from execution?" Answer: Accuracy (reduce cognitive load), cost (use expensive models only for planning), fault tolerance (retry failed steps without re-planning everything).

---

## Concept 6: LangGraph — State Machines for Agent Orchestration

### The problem it solves

Writing agent loops with raw while loops and if/else logic is brittle, hard to debug, and impossible to pause or resume. State machines give you a formal framework to model the flow: nodes are functions (agents or tools), edges are transitions, and the graph is compiled into an executable app with built-in persistence, checkpointing, and HITL (human-in-the-loop) support.

### The intuition (analogy first)

Imagine you are building a complex workflow in a no-code tool like Zapier or n8n. You drag boxes (nodes) onto a canvas, connect them with arrows (edges), and when you hit "run," the system executes the workflow step by step. If one step fails, you see exactly where it failed and can retry from there. LangGraph is the code equivalent: you define nodes (Python functions), edges (transitions), and compile it into an app.

### What is a state machine

A state machine is a formal model where:
1. The system is in exactly one state at any given time.
2. Transitions between states are triggered by inputs (e.g., "agent decided to call this tool").
3. Each state can have side effects (e.g., call an API, update a database).

For agentic systems, states might be: "Planning," "Executing Tool," "Waiting for Human Approval," "Generating Final Answer."

### Why state machines for AI

LLM outputs are non-deterministic (you cannot predict exactly what the LLM will say). State machines impose deterministic pathways on top of that: "If the LLM calls tool X, go to node Y. If it says FINISH, exit the graph." This makes the system controllable and debuggable.

### LangGraph vs. LangChain

- **LangChain:** Linear pipelines (DAGs, Directed Acyclic Graphs). Good for: Embed → Retrieve → Generate. Cannot loop.
- **LangGraph:** Cyclic graphs. Good for: Observe → Think → Act → Observe → Think → Act → ... Can loop indefinitely (with guards).

LangGraph is built on top of LangChain. You use LangChain to define tools and prompts, then wire them into a LangGraph graph.

### The three core concepts

1. **State:** A shared data structure (Python TypedDict) passed between every node. Nodes read from it and write updates to it.
2. **Nodes:** Python functions that receive the state, do work (call LLM, execute tool), and return a state update.
3. **Edges:** Define which node runs next. Can be unconditional ("always go to node B after node A") or conditional ("if state.next_agent == 'infrastructure', go to InfraNode").

### Worked example (multi-agent DevOps graph)

**State schema:**

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]  # conversation history
    next_agent: str  # "infrastructure", "codebase", "hr", or "FINISH"
    sender: str  # which agent last spoke
```

**Nodes:**

1. `triage_node(state)`: The Triage Agent reads the latest message, decides which specialist to call, updates `state["next_agent"]`.
2. `infrastructure_node(state)`: The Infrastructure Agent calls Splunk tools, appends results to `state["messages"]`, sets `state["next_agent"] = "triage"` (hand control back).
3. `codebase_node(state)`: The Codebase Agent calls GitHub tools, appends results, hands back to Triage.

**Edges:**

- Conditional edge after `triage_node`: Read `state["next_agent"]`. If "infrastructure", route to `infrastructure_node`. If "codebase", route to `codebase_node`. If "FINISH", exit.
- Unconditional edges: `infrastructure_node` → `triage_node`, `codebase_node` → `triage_node` (specialists always return control to Triage).

**Compile:**

```python
graph = StateGraph(AgentState)
graph.add_node("triage_agent", triage_node)
graph.add_node("infrastructure_agent", infra_node)
graph.add_node("codebase_agent", code_node)
graph.add_conditional_edges("triage_agent", router_function)
graph.add_edge("infrastructure_agent", "triage_agent")
graph.add_edge("codebase_agent", "triage_agent")
app = graph.compile()
```

Now you can run: `app.invoke({"messages": [HumanMessage("Check logs")]})`

The graph executes: Triage → (routes to) Infrastructure → (returns to) Triage → (generates answer) → FINISH.

### The power of compile()

Compiling the graph gives you:
- **Thread checkpointing:** The state is saved after every node execution. You can pause, serialize to a database, and resume days later without losing context.
- **Execution streaming:** You can stream intermediate updates to the UI in real-time.
- **HITL support:** Insert approval nodes where the graph pauses and waits for a human to click "Approve" before continuing.

### Common misunderstandings

- **"LangGraph is just a wrapper around while loops."** Technically yes, but it also adds persistence, error handling, HITL, and a declarative API that makes complex flows easier to reason about.
- **"I need to learn graph theory."** No. The "graph" is just a flowchart. Nodes = boxes, edges = arrows.

### Exam angle

Expect a question like: "What does LangGraph provide over raw Python loops?" Answer: State persistence, HITL checkpoints, conditional routing, error handling. Or: "What is the difference between LangChain and LangGraph?" Answer: LangChain is for linear DAGs, LangGraph supports cycles.

---

## Concept 7: Memory — Episodic vs. Semantic

### The problem it solves

LLMs are stateless. Every API call starts with a blank slate. If your agent debugs an issue over 10 turns, and on turn 11 it forgets what it did on turn 3, the system is useless. Memory is how you persist context across turns (episodic) and across sessions (semantic).

### The two types

1. **Episodic memory (short-term):** The current conversation thread. Stored in the State as a list of messages. When the user says "Is it related to the commit?", the agent resolves "it" to "the error we just found in the logs" because the logs are in episodic memory.
2. **Semantic memory (long-term):** Facts learned across different sessions, stored in a vector database. "We fixed this exact OOM error 3 months ago by increasing the JVM heap size" is semantic memory.

### Episodic memory: The conversation thread

**How it works:**

The State contains a field like `messages: List[BaseMessage]`. Each message has:
- `content`: the text
- `name`: who said it (user, triage_agent, infra_agent)

When the Triage Agent calls the LLM, it passes the entire `messages` list as context. The LLM sees the full history and can reason about it.

**Worked example (multi-turn debugging):**

**Turn 1:**
- User: "Check DB server."
- Triage → Infra Agent → Infra returns: "DB server CPU at 90%"
- Episodic memory now: [User("Check DB"), InfraAgent("CPU at 90%")]

**Turn 2:**
- User: "Why is it so high?"
- Triage Agent sees episodic memory, resolves "it" = "DB server CPU"
- Triage → Codebase Agent: "Check recent commits to DB service"
- Codebase returns: "Commit xyz added a heavy query"
- Episodic memory: [User("Check DB"), InfraAgent("CPU 90%"), User("Why high"), CodebaseAgent("Commit xyz")]

**Turn 3:**
- User: "Revert it."
- Triage resolves "it" = "commit xyz" (from episodic memory)
- Triage → Codebase Agent: "Revert commit xyz"

This is natural, fluid conversation. It only works because the agent remembers the thread.

### The limit of episodic memory

After 50 turns, the message list is huge. Passing all of it to the LLM every time is expensive (token cost) and slow (context window saturation). At some point, you need to summarize older messages or prune them.

### Semantic memory: The institutional knowledge base

**How it works:**

When the agent team solves a novel issue, the Triage Agent synthesizes the solution and writes it to a vector database using a tool like:

```python
save_to_knowledge_base(fact="OOM error in payment pod is resolved by increasing JVM heap from 2GB to 4GB in deployment.yaml")
```

This fact is now stored forever. On day 30, when the same OOM error occurs, the Triage Agent runs a silent query before planning:

```python
search_knowledge_base(query="OOM error payment pod")
```

Returns: "OOM error in payment pod is resolved by increasing JVM heap..."

The agent applies the fix directly, skipping the expensive multi-agent investigation.

### Worked example (self-healing system)

**Day 1 (novel issue):**
- User: "Payment pod is OOMKilled."
- Triage → Infra Agent → checks logs → "OutOfMemory heap space"
- Triage → Codebase Agent → checks config → "JVM heap set to 2GB"
- Triage → Infra Agent → increases heap to 4GB → pod restarts successfully
- Triage writes to semantic memory: "OOMKilled in payment pod → increase heap 2GB → 4GB"
- Total cost: 15 LLM API calls

**Day 30 (recurrence):**
- User: "Payment pod is OOMKilled."
- Triage searches semantic memory → finds exact solution
- Triage → Infra Agent: "Increase heap to 4GB"
- Total cost: 2 LLM API calls (86% reduction)

This is autonomous learning. The system gets smarter over time.

### Common misunderstandings

- **"Episodic and semantic are the same."** No. Episodic = current thread (RAM). Semantic = long-term facts (hard drive).
- **"More context is always better."** No. Past a certain point, long context dilutes attention (the "needle in a haystack" problem). Always retrieve only the top 3–5 most relevant memories.

### Exam angle

Expect a question like: "What is the difference between episodic and semantic memory?" Answer: Episodic = short-term, current conversation thread. Semantic = long-term, facts stored in vector DB across sessions. Or: "Why not just pass all past conversations to the LLM?" Answer: Token cost, context saturation, needle-in-haystack.

---

## Concept 8: Human-in-the-Loop (HITL) — Pausing for Approval

### The problem it solves

You do not want an autonomous agent to silently delete a Kubernetes pod, drop a production database table, or send an email to 10,000 customers without human approval. HITL inserts a pause in the graph where the agent must wait for a human to click "Approve" before the dangerous action executes.

### The intuition (analogy first)

Imagine a self-driving car. When it encounters an ambiguous situation (construction zone, unclear signage), it does not guess. It stops and asks the human driver: "Should I proceed?" The human looks, decides, and tells the car to continue. HITL is the same for agents.

### How it works (step by step)

1. The graph reaches an `approve_action` node.
2. The node suspends execution and serializes the state to a database.
3. The system sends a notification to a human (Slack message, email, dashboard alert): "Agent wants to restart pod pay-pod-1. Approve?"
4. The human reviews the context (what the agent found, why it wants to restart).
5. The human clicks "Approve" or "Reject."
6. The graph resumes from the exact point it paused, with the human's decision injected into the state.

### Worked example (Infrastructure Agent detects a memory leak)

**State:**

```python
{
  "messages": [
    InfraAgent("pay-pod-1 has a memory leak"),
    InfraAgent("I propose restarting the pod")
  ],
  "pending_action": "kubectl_restart_pod",
  "target": "pay-pod-1",
  "approval_status": None
}
```

**Graph execution:**

1. Reaches `await_approval` node.
2. Graph pauses, saves state to Postgres.
3. Sends Slack message to admin: "Agent detected memory leak in pay-pod-1 and wants to restart it. [Approve] [Reject]"
4. Admin clicks "Approve" (3 hours later).
5. Graph resumes, sees `approval_status = "approved"`, proceeds to `execute_restart` node.
6. Pod restarts.

**Key insight:** The graph waited 3 hours without losing any context. This is only possible because LangGraph checkpoints the state.

### When to use HITL

For destructive actions:
- Deleting resources (pods, databases, files)
- Modifying production config
- Sending external communications (emails, Slack broadcasts)

For read-only actions (querying logs, searching code), HITL is usually not needed.

### Common misunderstandings

- **"HITL makes agents slow."** Only for destructive actions. Read-only flows run at full speed.
- **"HITL defeats the purpose of automation."** No. It automates the investigation and proposes a solution. The human just approves or rejects, rather than doing the investigation themselves.

### Exam angle

Expect a question like: "Your agent proposes restarting a production pod. What design pattern ensures safety?" Answer: HITL (Human-in-the-Loop) approval node. Or: "Can LangGraph pause execution for hours?" Answer: Yes, via thread checkpointing.

---

## Concept 9: Caveats in Production — The Three Traps

### Trap 1: Multi-agent infinite loops

**What happens:**

Triage Agent asks Codebase Agent for a file. The file does not exist. Codebase Agent asks Triage for clarification. Triage asks Codebase again. Loop continues indefinitely, burning thousands of tokens per minute.

**The fix:**

Always set a `recursion_limit` parameter in LangGraph (e.g., max 15 steps). If the graph hits 15 nodes without reaching FINISH, it terminates with an error: "Max recursion limit reached."

### Trap 2: State bloat (context window saturation)

**What happens:**

Every time agents talk, messages are appended to the State. After 20 turns, the message list is 10,000 tokens. Passing this to the LLM on every call is slow and expensive.

**The fix:**

Implement a `summarize` node that triggers when `len(messages) > 10`. It compresses the older messages into a short summary (e.g., "Infra Agent found CPU spike, Codebase Agent identified commit xyz as cause") and replaces them in the State. Now the LLM sees the summary + the last few messages, keeping the context window lean.

### Trap 3: Memory contradiction (stale facts)

**What happens:**

The agent saves a fact to semantic memory: "Server X is the primary DB." Two months later, infrastructure is upgraded, and Server Y is now primary. The agent retrieves the old fact and hallucinates that Server X is still primary.

**The fix:**

Every fact in semantic memory must include a timestamp. Implement an `update_memory` tool that can overwrite old facts. Optionally, prune facts older than a threshold (e.g., 6 months) unless they are explicitly marked as evergreen.

### Exam angle

Expect a question like: "Your agent system is experiencing infinite loops. What is the safeguard?" Answer: Set a recursion_limit. Or: "Resolution time was falling but now is rising, and token costs are spiking. What are two likely causes?" Answer: State bloat (message growth) and memory contradiction (stale facts causing failures).

---

## Concept 10: Provenance and Citations — Making Agent Decisions Auditable

### The problem it solves

When an agent says "The payment container failed due to an OOM error, related to yesterday's commit 8f4b2a," how do you know it did not hallucinate? You need citations: pointers back to the exact log line (Splunk-pod-1, line 452) and the exact commit (Git commit 8f4b2a). This is provenance: the chain of custody for every claim.

### Why it matters

1. **Verification:** Humans must be able to audit the agent's logic by clicking through to the source.
2. **Hallucination mitigation:** Forcing agents to cite sources drastically reduces fabricated answers.
3. **Accountability:** If the agent made a mistake, you need to know which sub-agent (Infra or Codebase) was responsible.

In enterprise support, an uncited answer is an unusable answer.

### The four citation architectures

1. **Inline prompting (baseline, flawed):** Instruct the LLM in the prompt: "After every sentence, cite the source like [Doc_1]." The LLM tries its best, but often hallucinates citations that do not exist.
2. **Structured tool forcing (enterprise standard):** Force the LLM to output JSON with separate fields for answer and citations. Example:

```json
{
  "answer": "The server failed due to an OOM error.",
  "citations": [
    {"doc_id": "splunk_log_42", "exact_quote": "OutOfMemory: Java heap space"}
  ]
}
```

Then programmatically verify that the exact_quote actually appears in splunk_log_42. If not, flag as invalid.

3. **Post-hoc attribution (the fact-checker):** Let the LLM generate the answer first (no citation formatting), then use a second LLM or NLI model to retroactively inject citations by comparing the answer against source docs. Highest accuracy but doubles compute cost.
4. **Agentic state provenance (LangGraph approach):** Citations are not generated by the LLM's text output but derived from the execution graph itself. Example:

```python
state["messages"] = [
  InfraMessage("Fetched logs from Splunk-pod-1"),
  CodebaseMessage("Retrieved commit 8f4b2a from GitHub")
]
```

The UI renders this as a visual audit trail: "Infra Agent → Splunk-pod-1 | Codebase Agent → commit 8f4b2a." No LLM hallucination risk because the citations are literal logs of API calls that occurred.

### Worked example (multi-agent final answer with citations)

**Final answer to user:**

"The payment container failed due to an Out of Memory error [Infra Agent: Splunk-pod-1, Line 452]. This aligns with yesterday's commit increasing the cache size [Codebase Agent: Git Commit 8f4b2a]."

The user can click:
- "Splunk-pod-1, Line 452" → opens Splunk dashboard at that exact log line
- "Git Commit 8f4b2a" → opens GitHub commit page

This is enterprise-grade transparency.

### The trade-off matrix

| Method | Implementation Cost | Latency | Accuracy |
|--------|-------------------|---------|----------|
| Inline Prompting | Low (text only) | Very Fast | Low (hallucinations) |
| Structured Output (JSON) | Medium (JSON parsing) | Moderate | High (exact quote verification) |
| Post-Hoc Attribution | High (multi-model) | Slow | Very High (strict entailment) |
| Agentic State | High (LangGraph setup) | Graph-dependent | Absolute (deterministic tool logs) |

For a consumer chatbot, inline is often enough. For enterprise DevOps, use structured output for RAG documents + agentic state for tool executions.

### Common misunderstandings

- **"Citations are just a nice-to-have."** No. In production, they are mandatory for trust and debugging.
- **"The LLM can always cite correctly if I prompt it well."** No. LLMs hallucinate citations even with good prompts. You must verify programmatically.

### Exam angle

Expect a question like: "Your agent uses inline prompting for citations. What are the two risks, and what is a minimal fix?" Answer: Risks = citation hallucination (citing docs that do not exist) and stale memory (citing old facts as current). Fix = verify cited IDs against message history + timestamp/prune memory.

---

## Concept 11: The Hierarchical ReAct Architecture (Putting It All Together)

### The global context vector

In a multi-agent system, there is a global context c_global that tracks the entire conversation, including which agent said what:

c_global = (x, t₁_Triage, a₁_Triage→Infra, o₁_Infra, t₂_Triage, a₂_Triage→Codebase, o₂_Codebase, ...)

Each specialist agent runs its own isolated micro-ReAct loop to complete its narrow task, then returns a synthesized observation to the global context. This shields the Triage Agent from raw system logs and prevents context overload.

### Worked example (the checkout API slowing down)

**Scenario:** The checkout API has been slowing down for two days. No deployment occurred, no errors in logs. But a feature flag controlling a third-party shipping-rate integration was changed three days ago.

**Question:** How should the Triage Agent proceed, and why would a single flat ReAct agent be riskier?

**Correct approach:**

1. Triage Agent delegates to Infra Agent: "Check latency metrics for checkout API over last 3 days."
2. Infra Agent returns: "Latency increased by 200ms starting 3 days ago."
3. Triage Agent thinks: "No deployment, no code change. Check config/flag changes."
4. Triage Agent delegates to HR/Policy Agent (or a config agent): "Search for config or flag changes 3 days ago related to checkout."
5. Agent returns: "Feature flag ENABLE_SHIPPING_RATE_INTEGRATION flipped to true."
6. Triage synthesizes: "The latency spike is caused by the new feature flag calling a slow third-party API."

**Why a flat agent with 30+ tools is riskier:**

Even though this query only needs 2–3 tools, the agent has access to 30+ tools in production. Tool hallucination risk is a property of the agent's total tool count, not the number of tools this query needs. A flat agent might:
- Mis-select a tool (e.g., call the GitHub API instead of the config API)
- Hallucinate parameters (e.g., query the wrong time range)
- Mix up schemas (e.g., confuse a Kubernetes API with a Splunk API)

By splitting into specialists, each agent sees only 5–10 relevant tools, drastically reducing confusion.

### Exam angle

This is a multi-part reasoning question testing:
1. Understanding the causal chain (flag change, not code change)
2. Recognizing that tool hallucination is a standing property of the agent's tool count, not per-query
3. Knowing that hierarchical multi-agent architecture mitigates this

---

## Putting it together

Here is how the lecture concepts stack to build a full enterprise agentic RAG system:

1. **Linear RAG fails at multi-hop queries** because it cannot loop back and refine.
2. **Agentic RAG gives the LLM tools** and lets it decide what to retrieve, when, and how many times.
3. **The ReAct loop (Observe-Think-Act)** is the core pattern. Each thought informs the next action, each action produces an observation.
4. **Tool calling** is the bridge: the LLM outputs JSON, the app executes the function, feeds back the result.
5. **Multi-agent systems split domains** to avoid tool hallucination. Triage Agent coordinates, specialists execute.
6. **Planner-Executor separation** keeps reasoning clean. Planners decide strategy, executors handle precise tool calls.
7. **LangGraph formalizes this** as a state machine: nodes = agents/tools, edges = transitions, compile = runnable app with checkpointing.
8. **Episodic memory** keeps the current thread in State. **Semantic memory** stores long-term facts in a vector DB.
9. **HITL gates destructive actions.** The graph pauses, waits for human approval, resumes.
10. **Recursion limits, state summarization, and memory pruning** prevent infinite loops, context bloat, and stale facts.
11. **Citations provide provenance.** Structured output + agentic state logs = deterministic, auditable citations.

The result: a DevOps agent that can autonomously investigate incidents, remember past solutions, and cite its sources — all while respecting human oversight for dangerous actions.

---

## Check yourself (5 questions)

**Q1:** What are the four components of an agentic system?

<details>
<summary>Answer</summary>
LLM (brain), Tools (hands), Memory (notebook), Planning (strategy).
</details>

**Q2:** A single ReAct agent has 30 tools and working memory. It still inefficiently resolves recurring incidents. What is structurally missing?

<details>
<summary>Answer</summary>
Hierarchical multi-agent architecture (Triage + specialists) or Planner/Executor split. Past ~20 tools, a single agent suffers tool hallucination regardless of memory quality. Memory governs what the agent knows; tool count governs whether it can reliably act.
</details>

**Q3:** Your agent proposes restarting a production pod. What design pattern ensures the action does not execute without approval? Where is the state stored during the wait?

<details>
<summary>Answer</summary>
HITL (Human-in-the-Loop) approval node. The graph pauses at an approval node and checkpoints the state to a database (e.g., Postgres). The system waits for a human to approve, which can take hours or days, then resumes from the exact point with no context loss.
</details>

**Q4:** Three weeks in, average resolution time (which had been falling) starts rising, and API costs spike even though incident volume is flat. What are two plausible causes, and how do you diagnose them?

<details>
<summary>Answer</summary>
Cause 1: Memory contradiction (stale facts causing repeated failures). Diagnostic: check failure rate vs. memory age.
Cause 2: State bloat (message list growing, inflating token costs). Diagnostic: check message-length growth over time.
Both are named failure modes from the lecture with falsifiable diagnostics.
</details>

**Q5:** Your agent uses inline prompting for citations. What are the two risks?

<details>
<summary>Answer</summary>
Risk 1: Citation hallucination (agent cites documents that do not exist or do not support the claim).
Risk 2: Stale memory cited as current (agent retrieves an old fact and presents it as up-to-date).
Minimal fix: verify cited IDs vs. message history + timestamp/prune memory.
</details>

---

## Quick reference (for revision)

**Linear RAG vs. Agentic RAG:**
- Linear = fixed pipeline (retrieve once, generate)
- Agentic = LLM in a loop, retrieval is a tool call, can loop indefinitely

**Agentic system components:**
1. LLM (brain)
2. Tools (hands)
3. Memory (notebook)
4. Planning (strategy)

**Core loop: Observe → Think → Act**
- Observe: receive input or tool result
- Think: LLM decides next step
- Act: execute a tool
- Loop back with observation

**Tool calling:**
- LLM outputs JSON: `{"tool": "fetch_logs", "arguments": {"server_id": "db-12"}}`
- App executes function, feeds result back to LLM

**Single-agent bottleneck:**
- Past ~20 tools → tool hallucination (confused schemas, wrong parameters)
- Fix: Multi-agent (Triage + specialists) or Planner/Executor split

**Multi-agent pattern:**
- Triage Agent (router, meta-planner): Delegates tasks, coordinates workflow
- Specialist Agents (executors): Infra Agent (logs, metrics), Codebase Agent (GitHub, code search), HR/Policy Agent (runbooks, HR DB)
- Each specialist has 5–10 tools, returns synthesized observations to Triage

**Planner vs. Executor:**
- Planner (expensive model): Generates high-level plan (DAG of steps)
- Executor (cheap model): Formats precise tool calls, handles errors
- Orchestrator: Feeds observations from Executor back to Planner, dispatches next step
- Benefits: Accuracy, cost efficiency, fault tolerance (retry failed steps without re-planning)

**LangGraph:**
- State machine framework for agents
- State: shared data structure (TypedDict), passed between nodes
- Nodes: Python functions (agents, tools), receive state, return updates
- Edges: transitions (unconditional or conditional)
- Compile: turns graph into executable app with checkpointing, HITL, streaming

**Memory:**
- Episodic (short-term): current conversation thread, stored in State as message list. Resolves "it" and "that" via context.
- Semantic (long-term): facts across sessions, stored in vector DB. Agent writes facts after solving novel issues, retrieves them later (self-healing).

**HITL (Human-in-the-Loop):**
- Pauses graph at approval node for destructive actions (restart pod, delete resource)
- Checkpoints state to DB, waits for human approval (can wait hours), resumes

**Production caveats:**
1. Infinite loops: Set recursion_limit (e.g., max 15 steps)
2. State bloat: Summarize node triggers when len(messages) > 10, compresses old messages
3. Memory contradiction: Store timestamps, implement update_memory tool, prune stale facts

**Citations (provenance):**
1. Inline prompting: instruct LLM to cite in text. Fast but hallucination-prone.
2. Structured output (JSON): force LLM to emit answer + citations separately. Verify exact_quote exists in source.
3. Post-hoc attribution: second model retroactively injects citations. Highest accuracy, doubles cost.
4. Agentic state: derive citations from graph execution logs (deterministic, no hallucination).

**Hierarchical ReAct:**
- Triage Agent operates at meta-level: action space = {DelegateTo(InfraAgent), DelegateTo(CodebaseAgent), ...}
- Each specialist runs micro-ReAct loop, returns synthesized observation
- Global context c_global tracks full conversation with agent provenance

**Self-healing pattern:**
- Day 1: Novel issue, agent investigates (15 API calls), solves, writes to semantic memory
- Day 30: Same issue, agent searches semantic memory, finds solution, applies directly (2 API calls, 86% reduction)

**Key exam insights:**
- Tool hallucination is a property of total tool count, not per-query needs
- Memory ≠ action-selection. Memory stores what the agent knows; hierarchical structure governs whether it can reliably act.
- HITL + self-healing are not contradictory: unapproved new actions gated, previously-approved recurring fixes bypass gate.
- Rising resolution time + rising costs despite flat incident volume → likely state bloat or memory contradiction.
- Inline citations + uncurated long-term memory → citation hallucination + stale facts presented as current.
