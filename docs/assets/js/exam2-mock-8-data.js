window.EXAM2_MOCK_8 = {
  "title": "Mock Exam 8 — Curveballs (challenge round)",
  "durationMinutes": 90,
  "marksPerQuestion": 4,
  "sections": [
    {
      "name": "Section 1 — Fine-tuning & alignment",
      "questions": [
        {
          "id": "q1",
          "stem": "Which of the following is NOT a recognised way to reduce catastrophic forgetting during fine-tuning?",
          "options": {
            "A": "Mixing a small share (about 1 to 10%) of the original training data back into the new data",
            "B": "Raising the learning rate so the new skill is absorbed in fewer steps",
            "C": "Penalising movement of important weights, as in EWC or L2-SP regularisation",
            "D": "Averaging the fine-tuned weights with the original weights, as in WiSE-FT"
          },
          "correct": "B",
          "explanation": "Correct: (B). Forgetting happens because updates overwrite weights the old skills depended on. A higher learning rate makes bigger overwrites, so it makes forgetting worse. The usual advice is the opposite: a conservative learning rate and few epochs. (A) Replay is a standard fix, because the model keeps seeing old examples. (C) EWC and L2-SP add a penalty so important weights stay close to where they were. (D) WiSE-FT blends the old and new weights so some of the old behaviour returns. Review: Chapter 02, Concept 12, Catastrophic forgetting."
        },
        {
          "id": "q2",
          "stem": "A team fine-tunes on 48,000 training examples with a micro-batch of 8 per GPU, 4 gradient-accumulation steps and 3 GPUs. They train for 2 epochs with a warm-up ratio of 0.06 of all optimizer steps. How many warm-up steps is that?",
          "options": {
            "A": "About 30 steps",
            "B": "240 steps",
            "C": "180 steps",
            "D": "60 steps"
          },
          "correct": "D",
          "explanation": "Correct: (D). Effective batch = 8 × 4 × 3 = 96. Steps per epoch = 48,000 / 96 = 500. Two epochs give 1,000 optimizer steps, and 0.06 × 1,000 = 60 warm-up steps. (A) 30 uses only one epoch of steps (0.06 × 500). (B) 240 forgets the accumulation steps, so the batch is 24 and there are 4,000 steps in total. (C) 180 forgets the 3 GPUs, so the batch is 32 and there are 3,000 steps in total. Always count micro-batch, accumulation and GPUs together. Review: Chapter 02, Concept 11, Batch size, and Numbers & formulas, section 4."
        },
        {
          "id": "q3",
          "stem": "Statement 1: Gradient checkpointing lowers activation memory by recomputing some activations during the backward pass. Statement 2: Because of this, it also removes sudden memory spikes, so a paged optimizer is no longer useful. Which option is correct?",
          "options": {
            "A": "Statement 2 is true and Statement 1 is false",
            "B": "Both statements are true",
            "C": "Statement 1 is true; Statement 2 is false",
            "D": "Neither statement is true"
          },
          "correct": "C",
          "explanation": "Correct: (C). Statement 1 is true: checkpointing trades extra compute for lower activation memory. Statement 2 is false: it does not remove spikes. The paged optimizer still helps, because it moves Adam's states to CPU RAM when GPU memory suddenly peaks, which is why QLoRA uses both together. (A) This has the truth values the wrong way round. (B) Statement 2 is the false one. (D) Statement 1 is true, so neither is not right. Review: Chapter 04, Concepts 11 and 12."
        },
        {
          "id": "q4",
          "stem": "Which list gives the order of one QLoRA training step on a single layer?",
          "options": {
            "A": "Dequantize the layer to BF16, run the forward pass, delete the temporary copy, backpropagate into A and B only, then apply the Adam update",
            "B": "Run the forward pass on the 4-bit weights, backpropagate into the base weights, update them in 4-bit, then merge A and B",
            "C": "Apply the Adam update to A and B, dequantize the layer, run the forward pass, then backpropagate into the base weights",
            "D": "Dequantize the layer, backpropagate into the base weights, run the forward pass, then page the optimizer to the disk"
          },
          "correct": "A",
          "explanation": "Correct: (A). The 4-bit layer is first expanded to BF16 so it can be multiplied. After the forward pass the temporary BF16 copy is deleted to save memory. The backward pass then sends gradients only into the small matrices A and B, and Adam updates only them, with its states paged if memory is short. (B) The base weights are frozen, so no gradient is applied to them and nothing is updated in 4-bit. (C) An update cannot come before the forward and backward passes that produce its gradients, and the base never trains. (D) A backward pass cannot come before the forward pass, and the optimizer pages to CPU RAM, not to disk. Review: Chapter 04, Putting it together, What QLoRA actually does."
        },
        {
          "id": "q5",
          "stem": "With 4-bit NF4 weights, each block of 128 parameters stores one 32-bit absmax constant. What is the average storage per parameter, in bits?",
          "options": {
            "A": "4.0 bits",
            "B": "4.125 bits",
            "C": "4.25 bits",
            "D": "4.5 bits"
          },
          "correct": "C",
          "explanation": "Correct: (C). Each parameter costs 4 bits, plus its share of the block constant: 32 / 128 = 0.25 bits. So the total is 4 + 0.25 = 4.25 bits. (A) 4.00 ignores the constants altogether. (B) 4.125 would be a 256-parameter block, since 32 / 256 = 0.125. (D) 4.50 is the figure for blocks of 64, where 32 / 64 = 0.5. The block size here is 128, so the overhead is smaller. Review: Chapter 04, Concepts 8 and 10, and Numbers & formulas, section 14."
        },
        {
          "id": "q6",
          "stem": "Which statement most precisely separates prompt tuning from prefix tuning?",
          "options": {
            "A": "Prompt tuning trains the whole embedding table of the model, while prefix tuning trains only the output layer of the model",
            "B": "Prompt tuning learns vectors only at the input embedding layer, while prefix tuning learns key and value vectors at every layer",
            "C": "Prompt tuning learns key and value vectors at every layer, while prefix tuning learns vectors only at the input",
            "D": "Prompt tuning needs the model to be unfrozen, while prefix tuning keeps the model frozen"
          },
          "correct": "B",
          "explanation": "Correct: (B). Prompt tuning adds t trainable vectors of size d at the input embedding layer only, so it has t × d parameters. Prefix tuning adds trainable keys and values at all L layers, which is 2 × L × t × d parameters. In both, the base model stays frozen. (A) Neither method trains the whole embedding table or only the output layer. (C) This swaps the two definitions. (D) Both keep the model frozen, so that is not a difference. Review: Chapter 03, Concept 8, Prompt tuning vs prefix tuning."
        },
        {
          "id": "q7",
          "stem": "A team has almost no spare GPU memory because Adam states for the whole model do not fit. They train one layer at a time and re-freeze that layer before moving to the next, so only one layer ever has optimizer state. Which method is this?",
          "options": {
            "A": "Progressive fine-tuning, because it adds each new layer to the trainable set",
            "B": "Prompt tuning, because only a small block of virtual tokens is trained",
            "C": "QLoRA, because the base model is stored in 4-bit form",
            "D": "Layer-wise fine-tuning, because it re-freezes the previous layer each time it moves on to the next one"
          },
          "correct": "D",
          "explanation": "Correct: (D). In layer-wise fine-tuning each layer is trained alone and then frozen again, so only one layer holds Adam states at any moment. (A) Progressive fine-tuning keeps the earlier layers unfrozen as new ones are added, so optimizer state keeps growing. (B) Prompt tuning trains virtual tokens at the input and does not go layer by layer. (C) QLoRA saves memory by quantising the base and training adapters, not by training one layer at a time. Review: Chapter 01, Layer-wise vs block-wise vs progressive fine-tuning."
        },
        {
          "id": "q8",
          "stem": "Which statement about the models used in PPO-based RLHF is NOT true?",
          "options": {
            "A": "The reference model is updated along with the policy so the KL penalty stays small",
            "B": "The value (critic) model is trained alongside the policy",
            "C": "The reward model is kept frozen while PPO runs",
            "D": "DPO removes the reward and value models and keeps only the policy and a frozen reference"
          },
          "correct": "A",
          "explanation": "Correct: (A). The reference model is a frozen copy of the starting policy. The KL penalty measures how far the policy has drifted from it, so if it moved too, the penalty would stop meaning anything. (B) True: PPO trains both the policy and the value model, which gives 2 trained and 2 frozen models in total. (C) True: the reward model was trained earlier and is frozen during PPO. (D) True: DPO needs only 2 models, the policy and the frozen reference. Review: Chapter 05, Concept 7, The four models in the PPO loop."
        },
        {
          "id": "q9",
          "stem": "A red-team run on a customer chatbot reports 96% of attack prompts blocked overall, but only 70% in the prompt-injection category. The team adds an input filter for injection. What should they do next?",
          "options": {
            "A": "Stop, because the overall pass rate was already above 95%",
            "B": "Re-run the full attack set and a set of harmless prompts, then compare pass rates overall and per category",
            "C": "Remove the output guardrails, because the new input filter now covers every risk",
            "D": "Test only the prompt-injection category, because the other categories already passed"
          },
          "correct": "B",
          "explanation": "Correct: (B). Red-teaming is a loop: attack, find the gap, fix it, re-test. The re-test has to check the fixed category and the rest, and it must include harmless prompts so that the new filter is not blocking normal users (over-blocking). An overall number can hide a weak category, so rates are read per category as well. (A) A high overall rate hid a 70% weak spot. (C) Guardrails work on both the input and the output, and one filter does not replace the other. (D) Testing one category misses side effects of the fix elsewhere, and misses over-blocking. Review: Chapter 06, Concept 6, Red-teaming."
        }
      ]
    },
    {
      "name": "Section 2 — Vision-language models & multimodal RAG",
      "questions": [
        {
          "id": "q10",
          "stem": "Which statement about the Segment Anything Model (SAM) is NOT correct?",
          "options": {
            "A": "The heavy image encoder runs once per image, and its embedding is reused for many prompts",
            "B": "The prompt decoder is light enough to give a mask quickly after each click",
            "C": "For an ambiguous prompt it can return several candidate masks with predicted quality scores",
            "D": "Each new click re-runs the full image encoder, so every prompt costs as much as the first"
          },
          "correct": "D",
          "explanation": "Correct: (D). SAM splits the work: the heavy encoder runs once, and the light decoder handles each new prompt. If the encoder re-ran on every click, interactive use would be slow, so this statement is the false one. (A) True: that is the main speed idea. (B) True: the decoder is small and fast. (C) True: it returns 3 candidate masks, each with a predicted IoU score. Review: Chapter 07, Concept 6, SAM."
        },
        {
          "id": "q11",
          "stem": "Which option lists the stages of SAM's data engine in the right order?",
          "options": {
            "A": "Fully automatic, then semi-automatic, then assisted-manual",
            "B": "Semi-automatic first, then assisted-manual, then fully automatic with a 32 × 32 grid of points",
            "C": "Assisted-manual, then semi-automatic, then fully automatic with a 32 × 32 grid of points",
            "D": "Assisted-manual, then fully automatic (32 × 32 grid), then semi-automatic"
          },
          "correct": "C",
          "explanation": "Correct: (C). The data engine bootstraps itself. First, people label masks with help from the model. Then the model proposes masks for confident objects and people fill in what is missing. In the last stage the model labels everything itself by prompting a 32 × 32 grid of points. (A) This runs the stages backwards, and the model cannot label automatically before it has learned from human labels. (B) Assisted-manual comes first, since the model has no labels yet. (D) The fully automatic stage must be last. Review: Chapter 07, Concept 6, The data engine."
        },
        {
          "id": "q12",
          "stem": "A vision encoder uses 224 × 224 images and 16 × 16 patches, so one view gives 196 visual tokens. A LLaVA-NeXT style AnyRes setup splits the image into a 2 × 2 grid of tiles and also adds one global view of the whole image. How many visual tokens go to the LLM?",
          "options": {
            "A": "980 tokens in total",
            "B": "784 tokens, counting only the tiles",
            "C": "196 tokens, a single view",
            "D": "3,920 tokens"
          },
          "correct": "A",
          "explanation": "Correct: (A). There are 4 tiles plus 1 global view, which is 5 views. Each view gives 196 tokens, so 5 × 196 = 980. AnyRes keeps more detail but pays for it in more tokens. (B) 784 counts only the 4 tiles and forgets the global view. (C) 196 is a single view, which is what a plain fixed-resolution model would use. (D) 3,920 multiplies by 20 views, which has no basis in the setup. Review: Chapter 07, Concept 4, LLaVA-NeXT AnyRes."
        },
        {
          "id": "q13",
          "stem": "Statement 1: ColPali, which scores page images with late interaction (MaxSim), indexes documents much faster than an OCR-based pipeline. Statement 2: Its index is smaller than a single-vector index, and it needs less GPU memory. Which option is correct?",
          "options": {
            "A": "Both statements are true, because both describe the same trade-off",
            "B": "Only Statement 1 is true",
            "C": "Only Statement 2 is true, and Statement 1 is false",
            "D": "Neither statement is true, since both are wrong"
          },
          "correct": "B",
          "explanation": "Correct: (B). ColPali skips OCR and layout steps, so indexing is about 18 times faster. The cost is that it keeps many vectors per page, so the index is larger and it needs more GPU memory, while query speed is comparable. Statement 2 therefore has the trade-off the wrong way round. (A) Statement 2 is false. (C) Statement 1 is true, so only 2 cannot be right. (D) Statement 1 is true. Review: Chapter 09, Concept 6, Late interaction and MaxSim."
        },
        {
          "id": "q14",
          "stem": "Which statement most precisely compares the Qwen-VL adapter with the DocVLM query-compression module?",
          "options": {
            "A": "Both compress the same visual features, but Qwen-VL uses 64 queries and DocVLM uses 256",
            "B": "Qwen-VL compresses OCR text and layout, while DocVLM compresses only raw image patches",
            "C": "Both use a fixed number of queries, so neither one can depend on the question that is asked",
            "D": "Qwen-VL compresses visual features with a fixed 256 queries, while DocVLM uses 64 learnable queries on OCR and layout signal and keeps the base VLM frozen"
          },
          "correct": "D",
          "explanation": "Correct: (D). Qwen-VL's adapter turns visual features into a fixed 256 tokens. DocVLM compresses OCR text with its layout positions into 64 learnable queries, makes the result instruction-aware, projects it to the VLM's hidden size and appends it to the visual tokens, with the base VLM frozen. (A) This swaps the query counts and the inputs. (B) This swaps what each one compresses. (C) DocVLM's output is instruction-aware, so it does depend on the question. Review: Chapter 08, Concept 4, Document intelligence."
        },
        {
          "id": "q15",
          "stem": "A company must search 2 million scanned report pages. Their OCR pipeline takes too long to index, tables and charts lose meaning after OCR, and extra storage is acceptable. Which choice fits best?",
          "options": {
            "A": "Keep the OCR pipeline and add a bigger text embedding model to get better matches",
            "B": "Use CLIP with one global vector per page and drop the re-ranking step",
            "C": "Index page images with ColPali, which uses late interaction and skips OCR",
            "D": "Fine-tune SAM to cut each page into regions before search"
          },
          "correct": "C",
          "explanation": "Correct: (C). ColPali indexes page images directly, so there is no OCR step, and it indexes about 18 times faster. Its larger index and extra GPU memory are acceptable here, and query speed stays comparable. (A) A bigger text model does not fix slow OCR or lost table and chart meaning. (B) One global vector per page loses fine detail that late interaction keeps. (D) SAM makes masks for objects. It is not a retriever for document pages. Review: Chapter 09, Concept 6, Late interaction and MaxSim."
        }
      ]
    },
    {
      "name": "Section 3 — Agents, multi-agent systems & LangGraph",
      "questions": [
        {
          "id": "q16",
          "stem": "Which statement about LangGraph is NOT correct?",
          "options": {
            "A": "The checkpointer is shared across every thread_id and is the long-term memory",
            "B": "recursion_limit counts super-steps, not individual tool calls",
            "C": "A reducer decides how a node's update is merged into a state key",
            "D": "After interrupt() is resumed, the node starts again from its first line, not from the interrupt"
          },
          "correct": "A",
          "explanation": "Correct: (A). The checkpointer saves state per thread_id, so it is short-term memory for one conversation. Long-term memory across threads belongs to the Store. (B) True: each super-step counts once towards the limit. (C) True: reducers merge updates, for example by appending to a list. (D) True: resuming re-runs the node from the top, so side effects before interrupt() can repeat. Review: Chapter 12, Concept 8, Memory: checkpointer vs Store."
        },
        {
          "id": "q17",
          "stem": "Which list gives the order of one LangGraph super-step?",
          "options": {
            "A": "Follow edges, run nodes, checkpoint, then merge with reducers",
            "B": "Save a checkpoint, run the nodes, follow the edges, then merge updates with reducers, then repeat",
            "C": "Run the nodes, checkpoint, merge updates with reducers, then follow the edges",
            "D": "Run the nodes, merge updates with reducers, save a checkpoint, then follow the edges"
          },
          "correct": "D",
          "explanation": "Correct: (D). In a super-step the active nodes run, their updates are merged into the State by reducers, the merged State is saved as a checkpoint, and only then are the edges followed to choose the next nodes. (A) Edges cannot be followed before any node has run. (B) A checkpoint before the nodes run would save a State that has no results yet. (C) Saving before the merge would store a State without the updates. Review: Chapter 12, Concept 7, How a graph runs."
        },
        {
          "id": "q18",
          "stem": "A node sends a confirmation email and then calls interrupt() to wait for a manager's approval. After the manager approves, the customer gets the email twice. What is the best explanation and fix?",
          "options": {
            "A": "The checkpointer saved the state twice, so it should be turned off",
            "B": "Resuming re-runs the node from the top, so move the email after interrupt() or guard it with a derived idempotency key",
            "C": "interrupt() sends a second copy by design, so the email should be sent by a different graph",
            "D": "The recursion_limit was reached, so it should be raised"
          },
          "correct": "B",
          "explanation": "Correct: (B). With interrupt() plus Command(resume=...), the node is executed again from its first line, so anything before interrupt() runs twice. The fix is to place side effects after the interrupt, or to protect them with an idempotency key that is derived from the operation, not generated randomly. (A) The checkpointer is what makes resuming possible, so turning it off breaks the pause. (C) interrupt() does not send anything. The repeat comes from the re-run. (D) The recursion limit stops runaway loops and has nothing to do with a repeated email. Review: Chapter 12, Concept 9, Human-in-the-loop."
        },
        {
          "id": "q19",
          "stem": "Which of the following is NOT a reliable loop guard for a multi-agent system?",
          "options": {
            "A": "A hard round limit that ends the run after N exchanges",
            "B": "A token budget that stops the run once spending crosses a set value",
            "C": "Writing 'stop after three rounds' in the system prompt and trusting the agents to obey",
            "D": "A deadlock check that escalates to a human reviewer when two agents keep disagreeing for several rounds"
          },
          "correct": "C",
          "explanation": "Correct: (C). A prompt is only a request, and a model can ignore it or lose track of the count. Loop guards must be enforced in code by the runtime. (A) A round limit counted in code is reliable. (B) A token budget checked in code stops cost from running away. (D) A deadlock check with escalation to a human catches stuck disagreements. Review: Chapter 11, Concept 6, Bounding the loop."
        },
        {
          "id": "q20",
          "stem": "Statement 1: MCP connects an agent to tools and data, whereas A2A connects one agent to another agent, using Agent Cards and Tasks. Statement 2: RetryPolicy in LangGraph is the right tool for fixing a model's wrong or badly formatted answer. Which option is correct?",
          "options": {
            "A": "Only Statement 1 is true",
            "B": "Both statements are true, as each describes a real feature",
            "C": "Only Statement 2 is true, and Statement 1 is false",
            "D": "Neither statement is true"
          },
          "correct": "A",
          "explanation": "Correct: (A). Statement 1 is true: MCP is agent to tools and data, A2A is agent to agent. Statement 2 is false: RetryPolicy is meant for infrastructure failures such as a timeout or a flaky service, where trying again helps. A wrong answer from the model needs validation, feedback or a re-plan. (B) Statement 2 is false. (C) This has the truth values the wrong way round. (D) Statement 1 is true. Review: Chapter 11, Concept 12, MCP vs A2A, and Chapter 12, Concept 11, Reliability."
        },
        {
          "id": "q21",
          "stem": "Which statement about idempotency keys for a payment tool is most precise?",
          "options": {
            "A": "A random key made fresh on each retry is best, because every attempt is then unique and easy to trace",
            "B": "The key should be derived from the operation (thread, step, payment details), so a retry reuses it",
            "C": "The key is needed only for read-only tools, since reads can repeat safely",
            "D": "One fixed key shared by all payments is enough, as long as it is kept secret"
          },
          "correct": "B",
          "explanation": "Correct: (B). The point is that a retry of the same operation must be recognised as the same operation, so the key is built from stable details of that operation and the payment happens once. (A) A fresh random key on every retry makes each attempt look new, so a duplicate payment goes through. (C) Reads are already safe to repeat, and write actions are the ones that need the key. (D) One shared key would make all payments look like one operation, so later payments would be dropped. Review: Chapter 12, Concept 12, Idempotency."
        }
      ]
    },
    {
      "name": "Section 4 — Responsible AI",
      "questions": [
        {
          "id": "q22",
          "stem": "Which statement about cost control for an AI application is NOT true?",
          "options": {
            "A": "A hard quota can refuse requests once a limit is reached",
            "B": "A budget alert only tells people that spending is rising",
            "C": "A cap per user can stop one heavy user from using up the budget",
            "D": "A budget alert at 80% guarantees that spending stops at the budget"
          },
          "correct": "D",
          "explanation": "Correct: (D). An alert is a message, not a brake. If nobody acts on it, the spend continues, so an alert is not a cap. A control that actually stops spend must be enforced, for example as a hard quota or limit. (A) True: a quota enforced in the system stops the spend. (B) True: that is all an alert does. (C) True: per-user limits are a normal cost control. Review: Chapter 13, Concept 6, The five essential controls."
        },
        {
          "id": "q23",
          "stem": "A bank's chatbot tells a customer the wrong refund rule. The bank argues that the chatbot is a separate system, so the customer should take the issue up with the AI vendor. What does the course's responsible-AI lesson say?",
          "options": {
            "A": "The bank, as the deployer, stays accountable",
            "B": "The AI vendor alone is accountable, because it built the model",
            "C": "Nobody is accountable, because the answer was generated by a model",
            "D": "The customer is accountable, because they should have checked the answer"
          },
          "correct": "A",
          "explanation": "Correct: (A). The lesson from the Air Canada case, and from India's DPDP law, is that the organisation that deploys the system remains accountable for its outputs. A chatbot is not a separate legal party. (B) The vendor may share some duties by contract, but the deployer cannot hand over its own accountability. (C) Generated text still has an owner. (D) Customers are not expected to audit a company's own chatbot. Review: Chapter 13, Concept 4, The limits of existing rules."
        },
        {
          "id": "q24",
          "stem": "Statement 1: In responsible AI, 'responsible' is the promise to behave well, and 'trustworthy' is the proof you can show. Statement 2: Publishing an ethics policy is enough proof that a system is trustworthy. Which option is correct?",
          "options": {
            "A": "Both statements are true",
            "B": "Statement 2 is true and Statement 1 is false",
            "C": "Statement 1 is true and Statement 2 is false",
            "D": "Neither statement is true, because both claims fail"
          },
          "correct": "C",
          "explanation": "Correct: (C). Statement 1 is true: a promise is what you say, and proof is evidence such as test results, logs, documentation and monitoring. Statement 2 is false because a policy is only a promise. Trust has to be earned with evidence. (A) Statement 2 is false. (B) This has the truth values the wrong way round. (D) Statement 1 is true. Review: Chapter 13, Concept 2, Responsible AI vs trustworthy AI."
        }
      ]
    }
  ]
};
