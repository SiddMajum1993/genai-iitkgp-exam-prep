window.EXAM2_MOCK_6 = {
  title: "Mock Exam 6 — Numbers and formulas",
  durationMinutes: 90,
  marksPerQuestion: 4,
  sections: [
    {
      name: "Section 1 — Fine-tuning & alignment",
      questions: [
        {
          id: "q1",
          stem: "A team fully fine-tunes a 30B-parameter model with mixed precision. For every trained parameter they count 2 bytes (FP16 weight), 2 bytes (FP16 gradient), 8 bytes (Adam's two FP32 states) and 4 bytes (an FP32 master copy of the weight). Before activations, roughly how much GPU memory is that?",
          options: {
            A: "About 360 GB",
            B: "About 60 GB",
            C: "About 480 GB",
            D: "About 120 GB"
          },
          correct: "C",
          explanation: "Correct: (C). Bytes per parameter = 2 + 2 + 8 + 4 = 16. Memory = 30 billion × 16 bytes = 480 billion bytes ≈ 480 GB. (A) 360 GB uses 12 bytes per parameter (30 × 12). That is the common rule without the FP32 master copy, but this question says to count the master copy. (B) 60 GB is only the FP16 weights (30 × 2). It ignores gradients and optimizer states, which are bigger than the weights. (D) 120 GB is 4 bytes per parameter (weight plus gradient). It leaves out Adam's 8 bytes and the master copy. Review: Chapter 04, Concept 6 — Where the GPU memory actually goes."
        },
        {
          id: "q2",
          stem: "LoRA with rank r = 8 is added to the query and value matrices (2 matrices per layer) of a 24-layer model. Every one of those matrices is 2048 × 2048. Using L × M × r × (d + k), how many trainable LoRA parameters are there in total?",
          options: {
            A: "1,572,864",
            B: "786,432",
            C: "3,145,728",
            D: "65,536"
          },
          correct: "A",
          explanation: "Correct: (A). Per matrix: r × (d + k) = 8 × (2048 + 2048) = 8 × 4096 = 32,768. Per layer, with 2 matrices: 32,768 × 2 = 65,536. All 24 layers: 65,536 × 24 = 1,572,864. (B) 786,432 is 32,768 × 24. It counts only one matrix per layer, forgetting M = 2. (C) 3,145,728 is what you get with 4 matrices per layer (Q, K, V and O), but this question uses only 2. (D) 65,536 is the count for a single layer; the question asks for all 24. Review: Chapter 04, Concept 3 — Worked example: the size of a LoRA adapter."
        },
        {
          id: "q3",
          stem: "A bottleneck adapter has hidden size d = 512 and bottleneck size m = 16. It has a down-projection and an up-projection, each with a bias. Using the course formula for one adapter, how many parameters does it add?",
          options: {
            A: "16,384",
            B: "8,192",
            C: "528",
            D: "16,912"
          },
          correct: "D",
          explanation: "Correct: (D). The formula is 2md + m + d. Weights: 2 × 16 × 512 = 16,384. Biases: m + d = 16 + 512 = 528. Total: 16,384 + 528 = 16,912. (A) 16,384 counts only the weights and leaves out the biases. (B) 8,192 is m × d, one projection only; there are two (down and up). (C) 528 is just the biases (m + d) and misses the weights, which are most of the parameters. Review: Chapter 03, Concept 4 — What an adapter looks like; Numbers & formulas, section 9."
        },
        {
          id: "q4",
          stem: "A retailer's support bot must quote the current returns policy, which changes every month, and always answer in the brand's short, friendly format. The team has little GPU budget. Which plan fits the course's guidance best?",
          options: {
            A: "Fine-tune on each month's policy text so the facts sit in the weights, retraining monthly",
            B: "Use RAG to fetch the current policy at question time, and set tone and format in the prompt",
            C: "Paste the entire policy library into every prompt, which counts as RAG because the facts are supplied",
            D: "Fine-tune with PEFT only, because fine-tuning has the highest capacity for external knowledge"
          },
          correct: "B",
          explanation: "Correct: (B). RAG wins on knowledge that changes (temporary knowledge), and prompting is the cheapest way to set tone and format. Move to fine-tuning only if prompting fails to hold the style. (A) Fine-tuning gives permanent behaviour change, not fresh facts; retraining monthly is costly and slow. (C) Pasting whole documents is prompt engineering with in-context information. It is not RAG, because no retrieval step chooses what to include. (D) Fine-tuning is low to medium on external knowledge; RAG is the extremely high one. Review: Chapter 01, Concept 2 — The rules of thumb from the Q&A."
        },
        {
          id: "q5",
          stem: "You quantize X = [0.6, −2.0, 5.0] to 8 bits with absmax quantization: c = 127 ÷ max|X|, stored integer q = round(c · x), and dequantized value = q ÷ c. After the round trip, roughly what does the value 0.6 come back as?",
          options: {
            A: "15",
            B: "0.591",
            C: "0.600",
            D: "0.630"
          },
          correct: "B",
          explanation: "Correct: (B). The largest absolute value is 5.0, so c = 127 ÷ 5.0 = 25.4. Quantize: 0.6 × 25.4 = 15.24, which rounds to q = 15. Dequantize: 15 ÷ 25.4 ≈ 0.591. The small error (about 0.009) is the price of rounding. (A) 15 is the stored integer; it has not been divided by c yet. (C) 0.600 assumes the round trip is lossless, but rounding to an integer loses a little. (D) 0.630 comes from rounding 15.24 up to 16 (16 ÷ 25.4 ≈ 0.630); 15.24 rounds to 15, not 16. Review: Chapter 04, Concept 7 — Quantization: storing numbers with fewer bits; Numbers & formulas, section 12."
        },
        {
          id: "q6",
          stem: "A LoRA run was stable at an effective batch of 32 with learning rate 1e-4. You move to 4 GPUs with a micro-batch of 4 per GPU and gradient accumulation of 4. Following the linear scaling rule from the new effective batch, which starting learning rate is best?",
          options: {
            A: "5e-5",
            B: "About 1e-4",
            C: "4e-4 or higher",
            D: "About 2e-4"
          },
          correct: "D",
          explanation: "Correct: (D). Effective batch = micro-batch × accumulation × GPUs = 4 × 4 × 4 = 64. The batch doubled (64 ÷ 32 = 2), so the learning rate roughly doubles: 1e-4 × 2 = 2e-4. (A) 5e-5 forgets the GPUs: 4 × 4 = 16 would halve the batch, which would halve the rate. (B) 1e-4 ignores the scaling rule; the effective batch changed, so the rate should too. (C) 4e-4 would need an effective batch of 128, which double-counts something. Review: Chapter 02, Concept 11 — Batch size; Numbers & formulas, sections 5 and 6."
        },
        {
          id: "q7",
          stem: "In RLHF, the policy receives r_total = r_φ − β · KL(π_θ ‖ π_ref). For one response, the reward model gives 9, the KL divergence from the reference is 3, and β = 0.4. What is r_total?",
          options: {
            A: "About 7.8",
            B: "6.0",
            C: "8.6",
            D: "About 10.2"
          },
          correct: "A",
          explanation: "Correct: (A). The penalty is β × KL = 0.4 × 3 = 1.2. Then r_total = 9 − 1.2 = 7.8. A larger β would pull the policy closer to the reference model by charging more for drifting. (B) 6.0 is 9 − 3: it forgets to multiply the KL by β. (C) 8.6 is 9 − 0.4: it subtracts β alone and never multiplies by the KL. (D) 10.2 adds the penalty instead of subtracting it; the KL term is a cost. Review: Chapter 05, Concept 6 — The fix: KL penalty against a frozen reference."
        },
        {
          id: "q8",
          stem: "A teammate summarises LoRA for a review meeting. Which of their statements is NOT correct?",
          options: {
            A: "Matrix B starts at zero, so at the start of training the model behaves exactly like the original",
            B: "The frozen weights need no gradients or Adam states, which is the main memory saving",
            C: "Doubling the rank r leaves the trainable parameter count unchanged, because α/r rescales the update",
            D: "After training, the low-rank update can be merged into the original weights, so inference adds no extra latency"
          },
          correct: "C",
          explanation: "Correct: (C). This is the incorrect statement, which is why it is the answer. The parameter count is r × (d + k), which grows linearly with r, so doubling r doubles it. The α/r factor only rescales the size of the update; it does not change how many parameters exist. (A) True: if B = 0 then B·A = 0, so training starts from the original model's behaviour. (B) True: W₀ is frozen, so no gradient or optimizer state is kept for it. That is the main saving, not the small file size. (D) True: the update can be folded into W₀, which is why LoRA adds zero latency, unlike sequential adapters. Review: Chapter 04, Concept 4 — The two dials: rank r and alpha α."
        },
        {
          id: "q9",
          stem: "After fine-tuning, a model has lost some general skills. You apply WiSE-FT with α = 0.3 using θ_merged = (1 − α) · θ_base + α · θ_ft. One weight is 1.5 in the base model and 2.5 after fine-tuning. What is its merged value?",
          options: {
            A: "About 1.8",
            B: "2.2",
            C: "Exactly 2.0",
            D: "0.75"
          },
          correct: "A",
          explanation: "Correct: (A). (1 − α) = 0.7. Merged = 0.7 × 1.5 + 0.3 × 2.5 = 1.05 + 0.75 = 1.8. A small α keeps the merged weight close to the base model, which protects general skills. (B) 2.2 swaps the two weights (0.3 × 1.5 + 0.7 × 2.5), which favours the fine-tuned model. (C) 2.0 uses α = 0.5, an equal blend. (D) 0.75 is only the fine-tuned part (0.3 × 2.5); the base-model part is missing. Review: Chapter 02, Concept 12 — Catastrophic forgetting; Numbers & formulas, section 8."
        }
      ]
    },
    {
      name: "Section 2 — Vision-language models & multimodal RAG",
      questions: [
        {
          id: "q10",
          stem: "A ViT receives a 512 × 512 image and splits it into 16 × 16 patches. How many patches does it produce, not counting any [CLS] token?",
          options: {
            A: "32",
            B: "256",
            C: "1,024",
            D: "16,384"
          },
          correct: "C",
          explanation: "Correct: (C). Patches = (H × W) ÷ P² = (512 × 512) ÷ (16 × 16) = 262,144 ÷ 256 = 1,024. Same as (512 ÷ 16)² = 32² = 1,024. (A) 32 is the number of patches along one side, not the total. (B) 256 would come from using patch size 32 ((512 ÷ 32)² = 16² = 256). (D) 16,384 divides by P instead of P²: 262,144 ÷ 16. Review: Chapter 07, Concept 1 — The Vision Transformer (ViT); Numbers & formulas, section 19."
        },
        {
          id: "q11",
          stem: "A shop has 50,000 product photos and no labelled training data. They need each photo placed in one of 12 categories. Which CLIP-based approach fits?",
          options: {
            A: "Train a linear classifier on top of the frozen image encoder using the 50,000 photos",
            B: "Embed sentences such as \"a photo of a {category}\" with the text encoder, then pick the most similar for each image",
            C: "Embed the 12 category names with the image encoder and compare them with the photo embeddings",
            D: "Have CLIP write a caption for each photo, then match the caption text against the 12 categories using keyword search"
          },
          correct: "B",
          explanation: "Correct: (B). This is zero-shot classification. Class names become short sentences, the text encoder embeds them, and the highest similarity to the image embedding wins. No training examples are needed. (A) A linear probe needs labelled target data, which this shop does not have. (C) Names are text, so they go through the text encoder, not the image encoder. (D) CLIP cannot generate text; it only scores image-text matches. A generative captioner would be needed, and it is slower and costlier for a simple tag. Review: Chapter 07, Concept 3 — Zero-shot classification with CLIP."
        },
        {
          id: "q12",
          stem: "SAM's image encoder takes about 0.15 s and runs once per image. After that, each prompt costs about 50 ms in the lightweight mask decoder. An annotator clicks 8 points on one image. About how long does the model spend in total?",
          options: {
            A: "About 1.6 seconds",
            B: "About 0.40 s",
            C: "About 0.2 s",
            D: "About 0.55 s"
          },
          correct: "D",
          explanation: "Correct: (D). The encoder runs once: 0.15 s. The decoder runs 8 times: 8 × 0.05 = 0.40 s. Total = 0.15 + 0.40 = 0.55 s. (A) 1.6 s is 8 × (0.15 + 0.05) = 8 × 0.2, which wrongly re-runs the encoder for every click. (B) 0.40 s counts only the decoder runs and leaves out the one-time encoder cost. (C) 0.20 s is the time for a single prompt (0.15 + 0.05), not eight. Review: Chapter 07, Concept 6 — The three parts; Numbers & formulas, section 21."
        },
        {
          id: "q13",
          stem: "A VLM reads scanned invoices at 224 px. It misreads small totals and sometimes invents a PO number that is not on the page. Which pair of changes best targets both problems?",
          options: {
            A: "Feed the pages at higher resolution, and use a JSON schema in which missing fields may be null",
            B: "Keep 224 px but ask for step-by-step reasoning, and require a value in every schema field",
            C: "Raise the temperature to explore other readings, and add a larger tokenizer vocabulary",
            D: "Run external OCR and send only plain text, dropping the image, so the layout is preserved"
          },
          correct: "A",
          explanation: "Correct: (A). Resolution is the biggest lever for documents, because the text is small and dense. Allowing null or \"not found\" stops the model from inventing a plausible value for a missing field. (B) At 224 px the small text is still unreadable, and forcing a value for every field makes invention more likely. (C) Higher temperature adds randomness, and tokenizer size does not fix unreadable pixels. (D) Plain OCR text loses the spatial layout, which is what links labels to values. Review: Chapter 08, Concept 4 — Document intelligence (Why documents need special care; What goes wrong)."
        },
        {
          id: "q14",
          stem: "ColPali scores a page with MaxSim: for each query token, take its best-matching page patch, then add those best scores. The similarities below show 3 query tokens against 4 page patches. What is the page's MaxSim score?",
          code: "         patch1  patch2  patch3  patch4\ntoken1    0.1     0.8     0.3     0.2\ntoken2    0.6     0.2     0.5     0.4\ntoken3    0.3     0.1     0.9     0.7",
          options: {
            A: "3.0",
            B: "2.3",
            C: "5.1",
            D: "0.77"
          },
          correct: "B",
          explanation: "Correct: (B). Take the maximum of each query token's row: token1 → 0.8, token2 → 0.6, token3 → 0.9. Add them: 0.8 + 0.6 + 0.9 = 2.3. (A) 3.0 takes the best score in each patch column (0.6 + 0.8 + 0.9 + 0.7) instead of each query-token row. (C) 5.1 adds every number in the grid, with no maximum at all. (D) 0.77 averages the three maxima (2.3 ÷ 3); MaxSim sums them. Review: Chapter 09, Concept 6 — Late interaction and MaxSim; Numbers & formulas, section 22."
        },
        {
          id: "q15",
          stem: "A team reviews their multimodal RAG assistant, which cites page numbers with every answer. A reviewer lists statements about its weak points. Which statement is NOT correct?",
          options: {
            A: "If the right page is never retrieved, no amount of prompt wording can repair the answer",
            B: "A page that is on topic but unhelpful can still distract the generator and pull the answer off course",
            C: "A cited page number does not by itself prove that the answer actually came from that page",
            D: "Grounding the answer in retrieved pages removes hallucination, so citations no longer need checking"
          },
          correct: "D",
          explanation: "Correct: (D). This is the incorrect statement. Grounding reduces hallucination but does not eliminate it, so citations still need to be verified. (A) True: this is a retrieval failure. It is like answering from the wrong textbook. (B) True: irrelevant context, such as a Q2 page when the question is about Q3, can distract the generator. (C) True: over-trusting citations is a recognised pitfall; the footnote may not say what the essay claims. Review: Chapter 09, Concept 9 — Pitfalls (high exam value)."
        }
      ]
    },
    {
      name: "Section 3 — Agents, multi-agent systems & LangGraph",
      questions: [
        {
          id: "q16",
          stem: "A finance team's monthly refund process always runs the same four steps in the same order: verify the order, check the return window, issue the refund, send a confirmation email. Auditors want an identical path every time. The team wants an LLM to read messy customer emails and write the reply. Which design fits?",
          options: {
            A: "An autonomous agent that chooses the order of the four steps at run time, to stay flexible",
            B: "A supervisor with four worker agents, one per step, so that each step has its own specialist",
            C: "A fixed workflow with the order set in code, using the LLM only to read the email and draft the reply",
            D: "A ReAct loop with no step limit, so the model keeps going until it decides it is finished"
          },
          correct: "C",
          explanation: "Correct: (C). If you can draw the flowchart before the request arrives, build a workflow. It is cheaper, bounded and auditable, and the LLM is still useful for language. (A) An agent picks the next step at run time, which is exactly what auditors do not want here. (B) Splitting a fixed process into agents adds cost, latency and failure points with no benefit. (D) An unbounded loop breaks the rule that every agent loop needs an end condition, and the path is not even unknown. Review: Chapter 10, Concept 5 — Workflow vs agent: who controls the order?"
        },
        {
          id: "q17",
          stem: "A company has 15 internal tools (HR, ticketing, finance). Three different agent applications must use them, and the tool list changes often. The team is tired of re-writing JSON schemas into each application's prompt. Which approach fits best?",
          options: {
            A: "Expose the tools via MCP servers; agents discover them with list_tools() and run them with call_tool()",
            B: "Adopt A2A so each application can reach the tools by delegating work to other agents",
            C: "Replace the company's REST APIs with MCP, because MCP is meant to substitute for APIs",
            D: "Keep hand-writing schemas, because MCP is mandatory only for agents using more than one model"
          },
          correct: "A",
          explanation: "Correct: (A). MCP is a standard protocol for discovering and using tools. The server owns the schema, so new tools appear without editing prompts or redeploying agents. It pays off with many tools, changing tools and several applications. (B) A2A is for agent-to-agent communication, not for connecting agents to tools and data. (C) MCP sits on top of APIs; it does not replace them. (D) MCP is not mandatory at all; plain function calling is fine for a few tools, and there is no rule about the number of models. Review: Chapter 10, Concept 4 — MCP: tools you connect to, not re-declare."
        },
        {
          id: "q18",
          stem: "In a trip-planning system, the flight agent keeps switching to an earlier, pricier flight to fix check-in time, and the budget agent keeps rejecting the total. The coordinator has a round limit of 3 and a deadlock check. In round 2 the same conflict appears for the second time. What should happen?",
          options: {
            A: "Continue to round 3, then quietly pick the cheapest flight and report that the trip is planned and approved",
            B: "Stop early on the repeated conflict and escalate to a human with the best partial plan and the open conflict",
            C: "Ask both agents to try again, passing them the full conversation history so that they resolve it themselves",
            D: "Wait for the token budget to run out, since that is the only guard that measures whether progress is being made"
          },
          correct: "B",
          explanation: "Correct: (B). The deadlock check fires when the same conflict is seen twice, so the loop stops early without waiting for the round limit. Escalation hands the best partial plan and the unresolved conflict to a person instead of failing silently. (A) The check exists precisely so the loop does not burn another round, and a silent pick hides the conflict. (C) Giving both agents more context does not make a conflict resolve itself; reconciliation must be explicit. (D) The token budget guards against runaway spending; it is a different guard, not the only one. Review: Chapter 11, Concept 6 — Bounding the loop: three guards plus escalation."
        },
        {
          id: "q19",
          stem: "A coordinator hands the hotel agent the whole 60-message conversation, including flight discussions. The hotel agent misses the budget cap buried in the middle and books an expensive room. What is the best fix?",
          options: {
            A: "Keep sending the full history, because trimming context risks losing something important that is needed later",
            B: "Give every specialist the same context so that conflicts between them resolve themselves",
            C: "Send a small structured handoff with only what the hotel agent needs, such as dates, city and budget cap",
            D: "Let the hotel agent edit the shared state directly, so it can look up the cap by itself"
          },
          correct: "C",
          explanation: "Correct: (C). Handoffs should carry the minimum sufficient context as a small structured message. Full history burns tokens and buries key constraints. (A) This is the trap: full history buries the constraint that was missed. (B) Identical context does not resolve conflicts; reconciliation has to be an explicit step. (D) Agents should receive slices of state; the coordinator holds the shared state, and unrestricted edits increase risk. Review: Chapter 11, Concept 4 — Handoffs: pass only what's needed."
        },
        {
          id: "q20",
          stem: "In a LangGraph app, a flight-search node and a hotel-search node run in parallel in the same super-step. Both return {\"results\": [...]} for a plain list field. The run fails with INVALID_CONCURRENT_GRAPH_UPDATE. What is the right fix?",
          options: {
            A: "Raise recursion_limit so that the graph has enough super-steps to finish both parallel searches properly",
            B: "Move the results into the Store, which is the place for parallel writes",
            C: "Add a checkpointer so that each node's write is saved separately",
            D: "Give the field a reducer, for example Annotated[list, add], so both updates are combined"
          },
          correct: "D",
          explanation: "Correct: (D). With no reducer a field is overwritten, so two parallel writes would silently erase each other and LangGraph raises an error instead. An accumulating reducer combines both lists. (A) recursion_limit caps super-steps; it has nothing to do with merging writes. (B) The Store is long-term memory across threads, not a merge rule for state. (C) A checkpointer saves state per thread for resume and time travel; it does not decide how concurrent writes merge. Review: Chapter 12, Concept 5 — Why reducers matter: parallel nodes."
        },
        {
          id: "q21",
          stem: "An agent calls a booking API. The airline books the ticket, but the reply is lost and the agent sees a timeout, so it retries and the traveller ends up with two tickets. Which fix addresses the root cause?",
          options: {
            A: "Send an idempotency key built from the thread and itinerary, so every retry reuses it and gets the same ticket",
            B: "Send a fresh random UUID with each attempt, so that the airline can tell the separate attempts apart",
            C: "Increase the number of retries, so that one of them is more likely to see the reply from the airline",
            D: "Ask the model to judge from its own reasoning whether the first call probably worked or not"
          },
          correct: "A",
          explanation: "Correct: (A). Idempotent means repeating the operation gives the same final result. A key derived from the trip and intent makes each retry identical, so the airline replies \"already booked\" with the same ticket. (B) A fresh UUID per attempt makes every retry look like a new request, so it is not an idempotency key. (C) More retries make duplicate bookings more likely, not less. (D) The agent cannot tell whether the airline processed the request, and a model's guess is not a control; writes need an idempotency key, a human confirmation, or both. Review: Chapter 12, Concept 12 — Idempotency: never book twice."
        }
      ]
    },
    {
      name: "Section 4 — Responsible AI",
      questions: [
        {
          id: "q22",
          stem: "A parcel company updated the model behind its support chatbot. Afterwards the bot swore at a customer, even though the system prompt still said \"never use rude language\". Which lesson from the course applies most directly?",
          options: {
            A: "Make the system prompt instruction longer and more emphatic, because wording is the main control",
            B: "Add a disclaimer under the chat window explaining that the bot can make mistakes",
            C: "Use guardrails the system enforces, and re-test them after every model, prompt or vendor change",
            D: "Send an email alert to the support lead whenever a customer reports rude language"
          },
          correct: "C",
          explanation: "Correct: (C). An instruction (\"don't do X\") is not a control; a guardrail the system enforces is. Guardrails must also be re-tested after every change, as the DPD chatbot case showed after a system update. (A) A stronger instruction is still only an instruction and can be ignored or bypassed. (B) A disclaimer does not prevent harm and is not a control. (D) An alert tells you after the event; it does not stop the behaviour. Review: Chapter 13, Concept 6 — The five essential controls (Guardrails)."
        },
        {
          id: "q23",
          stem: "An AI agent runs in a loop overnight after an API key is stolen. The company had a billing alert that emailed the finance team at US$500, but nobody saw it and the bill reached tens of thousands of dollars by morning. Which control would have limited the damage?",
          options: {
            A: "A second alert at a lower amount, so that more people receive an email when spending starts to climb",
            B: "A longer system prompt telling the agent to be careful with spending",
            C: "A monthly review meeting where finance explains the AI bill",
            D: "A hard spending cap on the key or project, with a named owner of the AI bill"
          },
          correct: "D",
          explanation: "Correct: (D). A cap stops the spending; an alert only reports it. The course also asks for loop limits, key security and a named owner. (A) More alerts are still alerts, and nobody may be awake to act on them. (B) A prompt is an instruction, not a control, and it would not stop a stolen key. (C) A monthly review happens long after the money is gone. Review: Chapter 13, Concept 6 — The five essential controls (Cost control)."
        },
        {
          id: "q24",
          stem: "A bank uses a vendor's AI tool to summarise customer files. The tool mishandles personal data, and the vendor says the mistake belongs to its model, not to the bank. Under the course's view of India's DPDP Act, who stays responsible?",
          options: {
            A: "Only the vendor, because it built and runs the model and the bank only bought a finished product from it",
            B: "The bank as well, because the organisation remains responsible even when a vendor processes the data",
            C: "No one, because existing laws do not cover AI systems at all",
            D: "The customer, who accepted the terms of use of the bank's website"
          },
          correct: "B",
          explanation: "Correct: (B). Most organisations are deployers, and a deployer still answers for outcomes. Under the DPDP Act, the organisation stays responsible even when a vendor processes the data. (A) Passing blame to the vendor does not remove the deployer's responsibility. (C) Existing rules do not cover every AI action, but that does not mean no one is accountable; it means the organisation needs its own controls. (D) The customer does not carry the organisation's duty for how data is handled. Review: Chapter 13, Concept 4 — The limits of existing rules: deployers still answer."
        }
      ]
    }
  ]
};
