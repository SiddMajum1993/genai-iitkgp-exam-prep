// Practice questions for Chapter 10 · From RAG to assistants and agents.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch10"] = {
  "title": "From RAG to assistants and agents",
  "questions": [
    {
      "id": "q1",
      "type": "Concept",
      "stem": "According to the course, what is the defining difference between a tool-using assistant (workflow) and an agent?",
      "options": {
        "A": "An agent uses a larger, more capable LLM than a workflow.",
        "B": "In a workflow the developer fixes the order of steps in advance; in an agent the model chooses the next step at run time based on the latest result.",
        "C": "A workflow cannot call external tools, while an agent can.",
        "D": "An agent always uses retrieval, while a workflow never does."
      },
      "correct": "B",
      "why": {
        "A": "The same model can power either. The dividing line is control of the order, not model size.",
        "B": "Correct. Control flow is the definition: developer-decided means workflow, model-decided at run time means agent.",
        "C": "A tool-using assistant does call tools. It just calls them in a fixed order you wrote.",
        "D": "Agents and RAG are independent. An agent can run without retrieval, and a workflow can include it."
      },
      "takeaway": "Who decides the next step? You → workflow. The model, at run time → agent."
    },
    {
      "id": "q2",
      "type": "Scenario",
      "stem": "A finance team wants to automate monthly expense-report generation. The steps are identical every month, auditors must see the same path each time, and costs must be tightly bounded. What should they build?",
      "options": {
        "A": "A multi-agent system with a supervisor and four specialists.",
        "B": "A fully autonomous agent so it can adapt each month.",
        "C": "A deterministic workflow (with an LLM only where language handling is needed, or no LLM if inputs are clean).",
        "D": "A Reflection loop so every report is critiqued and revised."
      },
      "correct": "C",
      "why": {
        "A": "Multi-agent adds more calls, handoffs and failure points. It's for complex, decomposable goals.",
        "B": "An agent adds unpredictability and cost with no benefit when the path is fixed and must be auditable.",
        "C": "Correct. This is the course's own example of when a workflow wins: same steps, auditable path, bounded cost.",
        "D": "Reflection doubles or triples token spend and doesn't address the need for a fixed, auditable path."
      },
      "takeaway": "If you can draw the flowchart before the request arrives, build a workflow, not an agent."
    },
    {
      "id": "q3",
      "type": "Concept",
      "stem": "In function (tool) calling, what actually happens when an LLM 'calls' search_flights?",
      "options": {
        "A": "The MCP server decides on its own which tool the model needs and runs it.",
        "B": "The LLM looks up fares stored in its weights from training.",
        "C": "The LLM emits a structured request (tool name + arguments); the application code executes the real API call and returns the result to the model.",
        "D": "The LLM opens a network connection and calls the flight API directly."
      },
      "correct": "C",
      "why": {
        "A": "An MCP server exposes and executes tools when asked. It doesn't decide what the model needs.",
        "B": "Fares change hourly and can't come from training data. That's why tools are needed at all.",
        "C": "Correct. The model decides and emits the request; your code acts and feeds the result back as new context.",
        "D": "The slide is explicit: the model never touches the API."
      },
      "takeaway": "The model decides; your code acts. That boundary is where validation and safety checks live."
    },
    {
      "id": "q4",
      "type": "Which is NOT",
      "stem": "Which of the following statements about MCP (Model Context Protocol) is NOT correct?",
      "options": {
        "A": "The MCP server owns the tool schema, so new tools can appear without editing the agent's prompt.",
        "B": "MCP is the standard protocol that agents use to delegate tasks to other agents.",
        "C": "One MCP server can expose many tools, for example a Flights server with several flight-related tools.",
        "D": "An MCP client discovers available tools with list_tools() and runs them with call_tool()."
      },
      "correct": "B",
      "why": {
        "A": "True. This is the main benefit over hand-written schemas.",
        "B": "This is the incorrect statement. MCP connects agents to tools and data. Agent-to-agent delegation uses A2A.",
        "C": "True. Servers aren't one-to-one with tools; group tools by domain.",
        "D": "True. These are the two core operations: discover, then call."
      },
      "takeaway": "MCP = agent ↔ tools/data. A2A = agent ↔ agent."
    },
    {
      "id": "q5",
      "type": "Trace the loop",
      "stem": "In the Paris trace, the agent found a fare and a hotel that together fit the budget, then called check_policy and got: 'FAIL: departure is 5 days away, policy needs 7.' What does a well-designed agent do next?",
      "options": {
        "A": "Return the itinerary anyway, because it is within budget.",
        "B": "Search for an even cheaper hotel to create more budget headroom.",
        "C": "Crash with an error so a developer can investigate.",
        "D": "Treat the failure as an observation, identify the real cause (the date), and search flights for a later departure."
      },
      "correct": "D",
      "why": {
        "A": "That's exactly the mistake the fixed workflow made: within budget but non-compliant.",
        "B": "'The budget was never the problem.' A cheaper hotel doesn't fix the booking-window violation.",
        "C": "The loop must be recoverable: a failed check feeds the next decision, it doesn't crash the run.",
        "D": "Correct. A failed check is an observation; the agent re-plans around the cause, which is why agents exist."
      },
      "takeaway": "Step 4 of the Paris trace is the reason agents exist: observe the failure, fix the cause, re-check."
    },
    {
      "id": "q6",
      "type": "Compare",
      "stem": "Which pairing of loop shape and trade-off is correct?",
      "options": {
        "A": "ReAct: cheapest and easiest to debug, but can wander on long tasks.",
        "B": "Reflection: lowest token cost, because it works on only one draft.",
        "C": "ReAct: writes the entire plan before seeing any evidence.",
        "D": "Plan-and-Execute: cheapest shape, because it never re-plans."
      },
      "correct": "A",
      "why": {
        "A": "Correct. ReAct interleaves one thought and one action; it's cheap and debuggable but can drift over long horizons.",
        "B": "Reflection repeatedly drafts, critiques and revises, so it doubles or triples token spend.",
        "C": "Planning everything before evidence arrives is Plan-and-Execute's weakness, not ReAct's.",
        "D": "Plan-and-Execute does re-plan on failure. Its strengths are predictable cost and parallel steps."
      },
      "takeaway": "ReAct = cheap, step by step. Plan-and-Execute = plan first, predictable. Reflection = critique and revise, costly."
    },
    {
      "id": "q7",
      "type": "Classify",
      "stem": "During one run, the agent notes: 'policy retrieved; a flight is on hold; compliance re-check still needed.' Where should this information live?",
      "options": {
        "A": "The model's weights, via fine-tuning after the trip.",
        "B": "Long-term memory, so it is available next month.",
        "C": "Conversation state, because the user said it.",
        "D": "Task state (the scratchpad for this run)."
      },
      "correct": "D",
      "why": {
        "A": "Weights don't store per-run progress. That's what external state is for.",
        "B": "Intermediate results are temporary. Long-term memory is for durable facts such as the employee's grade or home airport.",
        "C": "Conversation state records what the user and assistant said. These are the agent's own progress notes.",
        "D": "Correct. Task state tracks what's done and what's left in one run. Without it, there's no loop."
      },
      "takeaway": "Conversation = one thread; task state = one run; long-term = across sessions."
    },
    {
      "id": "q8",
      "type": "Code reading",
      "stem": "Look at this simplified agent runner. Which property of a good agent loop does the highlighted check mainly provide?",
      "code": "for i in range(max_steps):\n    plan = think(state)\n    if plan.get(\"done\"):\n        # >>> highlighted check <<<\n        if not any(h.tool == \"check_policy\" and h.obs[\"compliant\"]\n                   for h in state[\"history\"]):\n            state[\"history\"].append(note(\"check_policy has not passed yet\"))\n            continue\n        return plan[\"answer\"]\n    obs = act(plan)          # errors come back as {'error': ...}\n    remember(state, plan, obs)",
      "options": {
        "A": "Long-term memory: it saves facts across sessions.",
        "B": "Parallelism: it runs tools at the same time.",
        "C": "Verified success: it accepts 'done' only if the history shows a passing check, not just the model's claim.",
        "D": "Query rewriting: it improves retrieval queries."
      },
      "correct": "C",
      "why": {
        "A": "The history is task state for this run, not cross-session memory.",
        "B": "Nothing here runs in parallel. It's a sequential loop.",
        "C": "Correct. The guard reads the history, not the thought. A false claim of success is caught in code.",
        "D": "Nothing here touches retrieval queries."
      },
      "takeaway": "You can't fix a false 'done' with a better prompt. The guardrail lives in your code. (range(max_steps) is the 'bounded' guard.)"
    },
    {
      "id": "q9",
      "type": "Concept",
      "stem": "How does retrieval change when it moves from a RAG pipeline into an agent (Agentic RAG)?",
      "options": {
        "A": "Retrieval becomes one tool among several; the agent decides whether to retrieve, with what query, and how many times.",
        "B": "Retrieval runs exactly once at the start, as in pipeline RAG.",
        "C": "Retrieval is replaced by a larger context window holding all documents.",
        "D": "Retrieval is removed, because agents rely only on tools."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The agent can rewrite the query, retrieve more than once, skip retrieval, or re-retrieve to verify.",
        "B": "'Always exactly once' describes pipeline RAG, not an agent.",
        "C": "The Q&A was clear: large context doesn't replace RAG for large, changing, private data.",
        "D": "Retrieval stays. It just becomes a tool the agent can choose."
      },
      "takeaway": "Agentic RAG = decision-making + iteration around retrieval."
    },
    {
      "id": "q10",
      "type": "Scenario",
      "stem": "An agent has a book_hotel tool. Sometimes it books the same hotel twice after a retry. Which fix does the course recommend?",
      "options": {
        "A": "Put a guard in the execution layer around the booking tool: a unique booking key, a check for existing bookings, and human approval before the write.",
        "B": "Add 'never book twice' to the system prompt and nothing else.",
        "C": "Remove the booking tool and let the model describe the booking in text.",
        "D": "Switch the loop shape from ReAct to Reflection."
      },
      "correct": "A",
      "why": {
        "A": "Correct. Write tools change the real world, so they need idempotency checks and human confirmation at the execution boundary.",
        "B": "Prompts help but aren't enforcement. The course says guards for write actions belong in code.",
        "C": "That avoids the problem by removing the capability; the task can no longer be completed.",
        "D": "Changing the loop shape doesn't stop duplicate writes."
      },
      "takeaway": "Read tools can run freely; write tools need code-level guards (idempotency) and human approval."
    }
  ]
};
