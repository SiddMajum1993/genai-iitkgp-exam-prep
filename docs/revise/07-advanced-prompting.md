# Lecture 7 — Advanced Prompt Engineering
**Instructor:** Prof. Koustav Rudra

## TL;DR (5 bullets)
- Zero-shot prompting gives direct instructions with no examples; few-shot provides input-output pairs; few-shot CoT adds intermediate reasoning steps.
- Chain-of-Thought (CoT) prompting guides LLMs to break complex tasks into logical steps, improving accuracy on reasoning problems.
- Self-Consistency samples multiple CoT reasoning paths and selects the most consistent answer via majority vote (not identifiable from a single prompt).
- ReAct interleaves Thought-Action-Observation cycles, enabling LLMs to use external tools and refine answers iteratively.
- Tree-of-Thoughts explores multiple reasoning branches in parallel, evaluates them, and selects the best path for complex decision-making.

## Exam-relevant concepts

### Zero-Shot Prompting
- **Definition:** Direct instructions to the LLM without providing any examples. The model relies purely on pre-trained knowledge.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** If you see a prompt with NO examples and NO reasoning chain, it is zero-shot. Larger models perform better at zero-shot tasks.
- **Common confusion:** Mistaking zero-shot CoT (which uses "Let's think step by step") for regular zero-shot.
- **How to tell it apart from similar concepts:** Zero-shot has NO examples. Few-shot has examples. Zero-shot CoT has "Let's think step by step" but no examples.

### Few-Shot Prompting
- **Definition:** Providing a small number of input-output examples to help the model understand intent, format, and style.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** If you see examples with direct Q → A pairs (no intermediate reasoning), it is few-shot WITHOUT CoT.
- **Common confusion:** Few-shot with CoT reasoning is NOT the same as few-shot without CoT.
- **How to tell it apart from similar concepts:**
  - Few-shot (no CoT): Q → A directly
  - Few-shot CoT: Q → reasoning steps → A

### Few-Shot Chain-of-Thought (CoT)
- **Definition:** Provides examples that include intermediate reasoning steps before reaching the answer.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** The presence of explicit reasoning steps in examples (e.g., "Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.") signals few-shot CoT.
- **Common confusion:** Confusing with few-shot without CoT (which just shows Q → A).
- **How to tell it apart from similar concepts:**
  - Few-shot (no CoT): "Q: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1. A: The answer is False."
  - Few-shot CoT: "Q: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1. A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False."

### Zero-Shot CoT
- **Definition:** Appending "Let's think step by step" to a zero-shot prompt to elicit reasoning without providing examples.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** The phrase "Let's think step by step" is the signature of zero-shot CoT.
- **Common confusion:** It is NOT few-shot because there are no examples.
- **How to tell it apart from similar concepts:**
  - Zero-shot: Direct question, no examples, no reasoning prompt.
  - Zero-shot CoT: Direct question + "Let's think step by step," no examples.
  - Few-shot CoT: Examples with reasoning steps included.

### Auto-CoT (Automatic Chain-of-Thought)
- **Definition:** Automates reasoning chain creation via question clustering and demonstration sampling. Uses zero-shot CoT to generate reasoning for representative questions.
- **Formula / numbers:** Two stages: (1) Question clustering, (2) Demonstration sampling.
- **When it matters (exam trap):** Auto-CoT is an automated approach to generate CoT examples, not manually crafted.
- **Common confusion:** Confusing Auto-CoT with manually written few-shot CoT.
- **How to tell it apart from similar concepts:** Auto-CoT uses clustering and heuristics; few-shot CoT uses manually curated examples.

### Self-Consistency
- **Definition:** An inference-time strategy that samples multiple CoT reasoning paths (using temperature > 0) and selects the most frequent answer via majority voting.
- **Formula / numbers:** Sample N reasoning paths, pick the answer that appears most often.
- **When it matters (exam trap):** Self-consistency is NOT identifiable from a single prompt. It requires multiple sampled outputs.
- **Common confusion:** Self-consistency is NOT a prompting technique you can identify from one example. It is a decoding strategy.
- **How to tell it apart from similar concepts:**
  - CoT: Single reasoning chain.
  - Self-Consistency: Multiple reasoning chains → majority vote.

### ReAct Prompting
- **Definition:** Interleaves Thought (reasoning), Action (tool use), and Observation (result from action) in a loop until the final answer is reached.
- **Formula / numbers:** Thought → Action → Observation → repeat.
- **When it matters (exam trap):** If a prompt shows cycles of "Thought N," "Act N," "Obs N," it is ReAct.
- **Common confusion:** Confusing ReAct with CoT. ReAct includes actions (e.g., Search, API calls); CoT is reasoning only.
- **How to tell it apart from similar concepts:**
  - CoT: reasoning steps only.
  - ReAct: reasoning + actions (tool calls) + observations.

### Tree-of-Thoughts (ToT)
- **Definition:** Explores multiple reasoning paths in parallel, evaluates each branch, and selects the best approach. Uses search algorithms (breadth-first, depth-first).
- **Formula / numbers:** N/A
- **When it matters (exam trap):** ToT involves multiple experts or approaches evaluated in parallel. Look for keywords like "imagine three different experts," "evaluate approaches," "branch out."
- **Common confusion:** Confusing ToT with self-consistency. ToT evaluates reasoning paths explicitly; self-consistency uses majority voting.
- **How to tell it apart from similar concepts:**
  - CoT: single linear reasoning path.
  - Self-Consistency: multiple paths, majority vote.
  - ToT: multiple paths, explicit evaluation of each path, selection based on quality/confidence.

### Meta Prompting
- **Definition:** Uses abstracted examples and structural templates (e.g., "Let's think step by step," LaTeX formatting) to guide LLM responses systematically.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** Meta prompting focuses on response structure and syntax (e.g., $\boxed{}$), not content.
- **Common confusion:** Confusing with CoT. Meta prompting emphasizes format; CoT emphasizes reasoning.
- **How to tell it apart from similar concepts:** Meta prompting includes explicit formatting instructions (e.g., "End the solution with the final answer encapsulated in a LaTeX-formatted box").

### Directional Stimulus Prompting (DSP)
- **Definition:** Uses a tuneable policy LM to generate "hints" or "clues" (directional stimulus) to guide a frozen LLM toward the desired output.
- **Formula / numbers:** Two steps: (1) Supervised fine-tuning to generate stimuli, (2) RL training with reward (e.g., ROUGE score).
- **When it matters (exam trap):** DSP generates keywords or hints from the input context to steer the LLM.
- **Common confusion:** Confusing DSP with RAG (retrieval-augmented generation). DSP generates hints internally; RAG retrieves external knowledge.
- **How to tell it apart from similar concepts:** DSP generates hints/keywords from the input itself; RAG retrieves from external sources.

### Prompt Chaining
- **Definition:** Decomposes one large task into a sequence of smaller, focused LLM calls, where the output of one step becomes the input to the next.
- **Formula / numbers:** N/A
- **When it matters (exam trap):** Look for sequential subtasks with intermediate checkpoints.
- **Common confusion:** Confusing chaining with a single complex prompt.
- **How to tell it apart from similar concepts:** Prompt chaining involves multiple LLM calls in sequence; a complex prompt is one LLM call.

### Temperature
- **Definition:** Controls sharpness of the next-token distribution. Lower temperature → sharper (more repetitive); higher temperature → flatter (more diverse).
- **Formula / numbers:** T = 0.5 gives low entropy (0.110); T = 1 gives medium entropy (0.594); T = 2 gives high entropy (1.109).
- **When it matters (exam trap):** Temperature = 0 is not feasible; it is thresholded to 0.000001 or greedy decoding is used.
- **Common confusion:** Thinking T = 0 is allowed. It is not; greedy decoding is used instead.
- **How to tell it apart from similar concepts:** Temperature affects probability distribution sharpness; Top-k and Top-p affect candidate token selection.

### Top-k Sampling
- **Definition:** Selects the top K most likely words and samples the next word only from this subset.
- **Formula / numbers:** k = 5 means consider only the top 5 tokens.
- **When it matters (exam trap):** Top-k is fixed; top-p is dynamic (adapts based on cumulative probability).
- **Common confusion:** Confusing top-k (fixed number) with top-p (cumulative probability threshold).
- **How to tell it apart from similar concepts:** Top-k = fixed K tokens; top-p = dynamic subset based on cumulative probability.

### Top-p (Nucleus Sampling)
- **Definition:** Selects a subset of words with cumulative probability exceeding threshold P.
- **Formula / numbers:** p = 0.65 means select tokens until cumulative probability reaches 65%.
- **When it matters (exam trap):** Top-p adapts the subset size; top-k is fixed.
- **Common confusion:** Confusing with top-k.
- **How to tell it apart from similar concepts:** Top-p is dynamic and adapts to the probability distribution; top-k is fixed.

### Hallucination
- **Definition:** LLM generates false or misleading information presented as fact (e.g., fabricated species, non-existent events).
- **Formula / numbers:** N/A
- **When it matters (exam trap):** Hallucination is a failure mode, not a prompting technique. Mitigations include guardrails and system prompts.
- **Common confusion:** Confusing hallucination with incorrect reasoning.
- **How to tell it apart from similar concepts:** Hallucination = fabricated facts; incorrect reasoning = logical errors.

## Prompt patterns / examples shown in slides

### Zero-Shot Example
```
User: Classify the test into neutral, negative or positive. Just give me the classification.
Text: I think the vacation is okay.
Sentiment:

Mistral: Neutral
```

### Few-Shot Example (no CoT)
```
User: This is awesome // Positive
This is bad! // Negative
Wow that movie was rad! // Positive
What a horrible show! //

Mistral: Negative
```

### Few-Shot Example (no CoT, fails on reasoning)
```
User: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: The answer is False.
The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.
A: The answer is True.
The odd numbers in this group add up to an even number: 16, 11, 14, 4, 8, 13, 24.
A: The answer is True.
The odd numbers in this group add up to an even number: 17, 9, 10, 12, 13, 4, 2.
A: The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A:

Model: The answer is True   (WRONG)
```

### Few-Shot CoT Example (correct)
```
User: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.
The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.
The odd numbers in this group add up to an even number: 16, 11, 14, 4, 8, 13, 24.
A: Adding all the odd numbers (11, 13) gives 24. The answer is True.
The odd numbers in this group add up to an even number: 17, 9, 10, 12, 13, 4, 2.
A: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A:

Mistral: The odd numbers in this group are: 15, 5, 13, 7, 1.
Adding these together gives: 15+5+13+7+1=41.
Since 41 is an odd number, the answer is False   (CORRECT)
```

### Zero-Shot CoT Example
```
Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
A: Let's think step by step

Output: There are 16 balls in total. Half of the balls are golf balls. That means that there are 8 golf balls. Half of the golf balls are blue. That means that there are 4 blue golf balls.  (CORRECT)
```

(Without "Let's think step by step," the zero-shot output was 8, which is WRONG.)

### ReAct Example
```
Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?

ITERATION 1:
Thought 1: I need to search Apple Remote and find the program it was originally designed to interact with.
Act 1: Search [Apple Remote]
Obs 1: The Apple Remote is a remote control introduced in October 2005 by Apple _ originally designed to control the Front Row media center program …

ITERATION 2:
Thought 2: Apple Remote was originally designed to control the Front Row media center program. I need to search Front Row next and find what other device can control it.
Act 2: Search [Front Row]
Obs 2: Could not find [Front Row]. Similar: ['Front Row Seat to Earth', 'Front Row Motorsports', 'Front Row (software)', ...]

ITERATION 3 & 4:
Thought 3: Front Row is not found. I need to search Front Row (software)
Act 3: Search [Front Row (software)]
Obs 3: Front Row is a discontinued media center software
Thought 4: Front Row (software) is controlled by an Apple Remote or the keyboard function keys. So the answer is keyboard function keys.
Act 4: Finish [keyboard function keys]
```

### Meta Prompting Example
```
Meta Prompt:
Problem Statement:
• Problem: [question to be answered]
Solution Structure:
1. Begin the response with "Let's think step by step."
2. Follow with the reasoning steps, ensuring the solution process is broken down clearly and logically.
3. End the solution with the final answer encapsulated in a LaTeX-formatted box, ... , for clarity and emphasis.
4. Finally, state "The answer is [final answer to the problem].", with the final answer presented in LaTeX notation

Few Shot prompt:
Problem: Find the domain of the expression √(x−2)/ √(5−x).
Solution: The expressions inside each square root must be non-negative. Therefore, x − 2 ≥ 0, so x ≥ 2, and 5 − x ≥ 0, so x ≤ 5. Also, the denominator cannot be equal to zero, so 5 − x > 0, which gives x < 5. Therefore, the domain of the expression is [2, 5). Final Answer: The final answer is [2, 5). I hope it is correct.

Problem: If det A = 2 and det B = 12, then find det(AB).

Response from LLM:
Let's think step by step.
1. We know that the determinant of a product of matrices (AB) is equal to the product of their determinants (det A * det B). This is given by the formula: det(AB) = det A * det B.
2. We are given that det A = 2 and det B = 12.
3. Substitute these values into the formula: det(AB) = 2 * 12.
4. Calculate the product: det(AB) = 24.
So, the final answer is $\boxed{24}$.
```

### Tree-of-Thoughts Example
```
Puzzle: Bob is in the living room. He walks to the kitchen, carrying a cup. He puts a ball in the cup, turns the cup upside down, then carries the cup to the bedroom. Where is the ball?

Usual output [ZERO SHOT]: The ball is in the bedroom.

Tree of thoughts:
Imagine three different experts are answering this question. They will track the ball step-by-step, evaluate the physics of each action, and branch out to cover different possible interpretations.

EXPERT 1 (Physics)
Cup upside down, gravity makes ball fall out in kitchen
Conclusion: Kitchen floor

EXPERT 2 (Language)
Story never says ball stayed inside the cup. "Upside down" implies the ball dropped before moving rooms.
Conclusion: Kitchen

EXPERT 3 (Skeptic)
The text does not specify the cup type or the ball size. If the ball is small and lodged in the rim, stuck by suction, or the cup has a hidden lid, it could remain inside while Bob carries it to the bedroom. The narrative leaves that possibility open.
Conclusion: Bedroom (ball still in cup)

FINAL CONSENSUS [Response]
The ball is in the kitchen.
```

### Prompt Chaining Example
```
Single Prompt (complex):
Consider the given text in Spanish. Translate it into English. Find all the statistics and facts used in this text and list them as bullet points. Translate them again into Spanish.

Prompt Chaining (sequential):
1. Read the given Spanish text.
2. Translate the text into English language.
3. Fetch the statistics and facts from the text.
4. Create a bullet point list of all these facts.
5. Translate them into Spanish language.
```

## Likely MCQ angles
- Given a single prompt example, identify the prompting technique (zero-shot, few-shot, few-shot CoT, zero-shot CoT).
- Distinguish CoT from self-consistency (self-consistency requires multiple samples, not identifiable from one prompt).
- Recognize ReAct by the presence of Thought-Action-Observation cycles.
- Distinguish Tree-of-Thoughts from self-consistency (ToT explicitly evaluates branches; self-consistency uses majority vote).
- Identify meta prompting by structural/formatting instructions.
- Distinguish prompt chaining from a single complex prompt.
- Temperature = 0 is not feasible (use greedy decoding or threshold to ~0.000001).
- Top-k is fixed; top-p is dynamic.
- Hallucination vs. incorrect reasoning.
- Auto-CoT uses clustering and heuristics; few-shot CoT uses manual examples.

## One-line flashcards (20)
- Q: What is zero-shot prompting? → A: Direct instructions to the LLM with no examples.
- Q: What is few-shot prompting? → A: Providing a small number of input-output examples to guide the model.
- Q: What is few-shot CoT? → A: Few-shot prompting with intermediate reasoning steps in the examples.
- Q: How do you identify few-shot CoT from an example? → A: The examples include reasoning steps (e.g., "Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.").
- Q: What is zero-shot CoT? → A: Appending "Let's think step by step" to a zero-shot prompt.
- Q: Can you identify self-consistency from a single prompt? → A: No. Self-consistency is an inference-time strategy that samples multiple CoT paths and majority-votes.
- Q: What is ReAct prompting? → A: Interleaving Thought, Action, and Observation in a loop to solve tasks using tools.
- Q: How do you distinguish ReAct from CoT? → A: ReAct includes actions (tool calls) and observations; CoT is reasoning only.
- Q: What is Tree-of-Thoughts? → A: Exploring multiple reasoning branches in parallel, evaluating them, and selecting the best path.
- Q: How do you distinguish ToT from self-consistency? → A: ToT explicitly evaluates branches; self-consistency uses majority voting without explicit evaluation.
- Q: What is meta prompting? → A: Using abstracted examples and structural templates (e.g., LaTeX formatting) to guide responses.
- Q: What is directional stimulus prompting? → A: Using a policy LM to generate hints/clues to guide a frozen LLM.
- Q: What is prompt chaining? → A: Breaking one large task into sequential smaller LLM calls, where output of one step is input to the next.
- Q: What happens when temperature = 0? → A: Not feasible; thresholded to ~0.000001 or greedy decoding is used.
- Q: What is top-k sampling? → A: Selecting the top K most likely tokens and sampling from this fixed subset.
- Q: What is top-p sampling? → A: Selecting tokens with cumulative probability exceeding threshold P (dynamic subset).
- Q: What is hallucination? → A: LLM generating false or misleading information presented as fact.
- Q: What is Auto-CoT? → A: Automating CoT example creation via question clustering and demonstration sampling.
- Q: What is the key phrase for zero-shot CoT? → A: "Let's think step by step".
- Q: Larger models are better at what? → A: Zero-shot tasks and CoT reasoning.
