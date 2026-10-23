// Practice questions for Chapter 12 · LangChain & LangGraph.
window.PRACTICE_QUIZZES = window.PRACTICE_QUIZZES || {};
window.PRACTICE_QUIZZES["ch12"] = {
  "title": "LangChain & LangGraph",
  "questions": [
    {
      "id": "q1",
      "type": "Scenario",
      "stem": "A booking agent sends a 'book flight AI142' request. The airline books the ticket, but the reply is lost and the agent sees a timeout. The agent cannot tell whether the airline processed the request. What is the safest way to retry?",
      "options": {
        "A": "Retry immediately up to three times without any key, because timeouts are temporary.",
        "B": "Retry with an idempotency key derived from the trip and the flight, so the airline recognises a repeat and returns the existing ticket.",
        "C": "Don't retry. Report the booking as failed so the traveller can book manually.",
        "D": "Retry with a new random UUID as the request ID, so the airline treats it as a fresh request."
      },
      "correct": "B",
      "why": {
        "A": "The timeout is temporary, but the booking is a write. Retrying a write without a key can buy two tickets.",
        "B": "Correct. A derived key (thread + intent) is the same on every retry, so a repeat returns 'already booked' with the same ticket.",
        "C": "This avoids a duplicate, but it may wrongly report failure when the ticket was actually issued. The idempotency key gives a safe retry instead.",
        "D": "A fresh UUID per attempt makes every retry look new, which is exactly how a second ticket gets booked."
      },
      "takeaway": "Every write needs an idempotency key, a human confirmation, or both. The key must be derived, never random per attempt."
    },
    {
      "id": "q2",
      "type": "Code reading",
      "stem": "Two nodes run in parallel in the same super-step. Each returns a list of hotel candidates in the field shown. What happens?",
      "code": "class TripState(TypedDict, total=False):\n    candidates: list[dict]     # no Annotated reducer\n    log: Annotated[list[str], add]\n\n# node_search_paris  returns {\"candidates\": [...], \"log\": [\"paris\"]}\n# node_search_roissy returns {\"candidates\": [...], \"log\": [\"roissy\"]}",
      "options": {
        "A": "Both candidate lists are automatically concatenated, because lists always append.",
        "B": "The graph waits for the next super-step and merges the lists then.",
        "C": "The second node's list silently replaces the first, and 'log' also keeps only one entry.",
        "D": "LangGraph raises INVALID_CONCURRENT_GRAPH_UPDATE for 'candidates', because both parallel nodes write an overwrite-only field."
      },
      "correct": "D",
      "why": {
        "A": "Lists don't append by default. You need Annotated[list, add] for that, as 'log' has.",
        "B": "Merging happens at the end of each super-step using the reducers. There's no deferred merge.",
        "C": "LangGraph doesn't silently drop one write, and 'log' has an add reducer, so both entries would be kept.",
        "D": "Correct. With no reducer the field is overwrite. Two concurrent writes conflict, and LangGraph raises an error instead of guessing."
      },
      "takeaway": "Default = overwrite. Fields written by parallel nodes need an accumulating reducer such as Annotated[list, add]."
    },
    {
      "id": "q3",
      "type": "Compare",
      "stem": "A travel agent should (1) resume a half-finished trip plan after a server crash, and (2) remember on every future trip that the employee wants to avoid a certain airline. Which pairing is correct?",
      "options": {
        "A": "(1) Checkpointer with the same thread_id, (2) Store, which persists facts across threads.",
        "B": "Both are handled by the checkpointer, as long as the recursion_limit is high enough.",
        "C": "Both are handled by the conversation summary in middleware.",
        "D": "(1) Store, (2) Checkpointer."
      },
      "correct": "A",
      "why": {
        "A": "Correct. The checkpointer saves state per thread for resumption; the Store holds application facts for every future run.",
        "B": "The checkpointer is scoped to one thread_id, so a new trip (new thread) wouldn't see it. recursion_limit is unrelated.",
        "C": "A summary compresses conversation to save tokens. It doesn't give crash recovery or reliable cross-session facts.",
        "D": "This is reversed. The Store is long-term memory across threads; the checkpointer is the per-thread state used to resume."
      },
      "takeaway": "Checkpointer = short-term, per thread_id (resume, HITL, time travel). Store = long-term, across threads."
    },
    {
      "id": "q4",
      "type": "Code reading",
      "stem": "This node pauses for manager approval. The developer notices that, after the manager approves, the confirmation email went out twice. Why?",
      "code": "def node_approval(state):\n    send_email(state[\"manager\"], state[\"itinerary\"])   # side effect\n    reply = interrupt({\"itinerary\": state[\"itinerary\"]})\n    return {\"manager_reply\": reply}\n\n# later:\napp.invoke(Command(resume=\"Approved\"), cfg)",
      "options": {
        "A": "On resume, the node re-runs from its first line, so send_email runs a second time before interrupt() returns the resume value.",
        "B": "interrupt() sends the payload as an email automatically, in addition to send_email.",
        "C": "Command(resume=...) always runs the whole graph again from START.",
        "D": "The thread_id was missing, so LangGraph created a new run."
      },
      "correct": "A",
      "why": {
        "A": "Correct. interrupt() doesn't freeze the function mid-way. The node restarts from the top on resume, so side effects before it execute twice.",
        "B": "interrupt() surfaces the payload to your application. It doesn't send emails.",
        "C": "Resume continues from the paused node using the checkpoint. It doesn't restart the whole graph.",
        "D": "Without a thread_id there would be nothing to resume. The node wouldn't have reached interrupt() with the approval."
      },
      "takeaway": "Never place side effects before interrupt() in the same node. Put them in a separate node after the approval."
    },
    {
      "id": "q5",
      "type": "Symptom → fix",
      "stem": "A flight-search tool returns: 'No economy fare under ₹50,000 on 15 Aug.' What should the agent framework do with this?",
      "options": {
        "A": "Retry the same search with exponential backoff until a fare appears.",
        "B": "Return it to the model as an observation, so it can try another date or say honestly that no fare fits.",
        "C": "Let the model estimate a likely fare so the plan can continue.",
        "D": "Raise an exception so the graph stops immediately and no wrong booking can happen."
      },
      "correct": "B",
      "why": {
        "A": "This isn't a temporary infrastructure failure. Retrying returns the same fact again and wastes calls.",
        "B": "Correct. A meaning-based 'no results' is a fact for the model to act on, returned as a clear tool message.",
        "C": "A confident fare that no tool returned is worse than an outage, because it looks like a real answer.",
        "D": "Crashing the run means the model is never told. An exception escaping a tool aborts the graph and loses the chance to replan."
      },
      "takeaway": "Retry what infrastructure can fix (timeouts, 503, 429). Surface meaning-based results to the model."
    },
    {
      "id": "q6",
      "type": "Concept",
      "stem": "Which statement about LangChain and LangGraph is correct?",
      "options": {
        "A": "They are competing frameworks, so a project must choose one or the other.",
        "B": "LangGraph is a no-code visual tool, while LangChain is code-first.",
        "C": "LangChain supplies standard components (model I/O, tools, typed output); LangGraph is the control-flow layer. Since 1.0, LangChain's create_agent is itself built as a LangGraph graph.",
        "D": "LangGraph replaces MCP, because it also connects agents to tools."
      },
      "correct": "C",
      "why": {
        "A": "The course explicitly calls this a misconception.",
        "B": "LangGraph is code-first. Tools like n8n are the low-code ones.",
        "C": "Correct. Toolbox vs flowchart: they work together, and create_agent is a small LangGraph graph (model node + tool node loop).",
        "D": "MCP connects an agent to tools and data; LangGraph orchestrates the agent. They are used together."
      },
      "takeaway": "LangChain standardises components; LangGraph orchestrates them. They're layers, not rivals."
    },
    {
      "id": "q7",
      "type": "Which is NOT",
      "stem": "Which of the following is NOT a correct statement about how a LangGraph run executes?",
      "options": {
        "A": "Exceeding the recursion_limit raises GraphRecursionError.",
        "B": "recursion_limit caps the number of tool calls the agent may make.",
        "C": "Nodes running in the same super-step read the previous checkpoint and don't see each other's writes.",
        "D": "Each super-step runs the active nodes, merges their updates with the reducers, saves a checkpoint, then follows the edges."
      },
      "correct": "B",
      "why": {
        "A": "This is correct. That's what the limit raises when hit.",
        "B": "This is the false statement, so it's the answer. recursion_limit counts super-steps, not tool calls (one tool round trip is usually two super-steps).",
        "C": "This is correct. If one node needs another's output, it must come in the next super-step.",
        "D": "This is correct. It's the super-step cycle from the slides."
      },
      "takeaway": "recursion_limit counts super-steps. Checkpoints are saved at each super-step boundary."
    },
    {
      "id": "q8",
      "type": "Scenario",
      "stem": "A planning agent keeps alternating between two fares and replanning without ever finishing, though each run stays well under the recursion_limit. Which addition targets this specific problem?",
      "options": {
        "A": "An oscillation guard and a step budget kept in state, with escalation that hands back the checkpointed state.",
        "B": "Add an idempotency key to the fare-search tool.",
        "C": "Switch the checkpointer from SQLite to Postgres.",
        "D": "Increase the recursion_limit so the agent has more room to converge."
      },
      "correct": "A",
      "why": {
        "A": "Correct. An oscillation guard catches A→B→A behaviour, a step budget catches slow thrashing, and escalating with the saved state keeps the work done so far.",
        "B": "Idempotency keys protect writes from duplication. A search is a read and isn't the problem here.",
        "C": "The storage backend doesn't affect whether the loop converges.",
        "D": "More room means more wasted steps. The loop needs an end condition that doesn't depend on the model choosing to stop."
      },
      "takeaway": "Use several independent bounds: recursion_limit, a step budget in state, a cost ceiling and an oscillation guard."
    },
    {
      "id": "q9",
      "type": "Concept",
      "stem": "Why does the course recommend asking the model for a typed verdict (e.g. a Pydantic PolicyVerdict with compliant: bool and violated_clause) instead of free-text answers?",
      "options": {
        "A": "It guarantees the model will always return a valid object.",
        "B": "It lets the model run the policy-checking tool by itself.",
        "C": "A conditional edge can route reliably on a typed field, and a malformed answer fails at the boundary instead of several steps later.",
        "D": "It removes the need for a checkpointer."
      },
      "correct": "C",
      "why": {
        "A": "Structured output doesn't guarantee validity. Its value is that invalid output fails loudly and early.",
        "B": "The model never runs tools itself. The application does.",
        "C": "Correct. A graph can't branch reliably on prose. Typed output also gives loud, early failure before bad data reaches booking.",
        "D": "Typed output and persistence solve unrelated problems."
      },
      "takeaway": "Structured output = data your edges can route on, plus loud, early failure when it's malformed."
    },
    {
      "id": "q10",
      "type": "Concept",
      "stem": "A deployed agent has a 5-second tool timeout, a 30-second node timeout and a 90-second run deadline. Why should these be nested so that the innermost clock fires first?",
      "options": {
        "A": "Because LangGraph refuses to compile a graph with equal timeouts.",
        "B": "Because the run deadline is only checked once the run has finished.",
        "C": "So that the agent retries the whole run instead of a single tool.",
        "D": "So that the trace names the exact dependency that stalled. If the outer clock fires first, you only learn that the run timed out."
      },
      "correct": "D",
      "why": {
        "A": "There's no such compile rule. The nesting is a design principle.",
        "B": "The run deadline is enforced during the run; past it, the agent checkpoints and returns what's settled.",
        "C": "The goal is a small, targeted retry or fallback, not rerunning everything.",
        "D": "Correct. The inner tool clock fires first and identifies the stalled dependency. An outer clock firing first destroys that diagnosis."
      },
      "takeaway": "Tool < node < run: nested timeouts, with the inner one firing first so you know what stalled."
    }
  ]
};
